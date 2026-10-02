# localize/localize_utils/batch.py
"""Serial and parallel batch runners for localize_plot_withorig.

Both runners checkpoint results periodically and resume from existing output files.
"""
from __future__ import annotations
import glob, os, signal, time, traceback, itertools
from datetime import datetime
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from typing import Optional

import pandas as pd


def _shard_dir(out_dir: str, out_basename: str) -> str:
    return os.path.join(out_dir, f"{out_basename}_shards")


def _write_shard(buffer: list, shard_dir: str, kind: str, seq: int, ext: str) -> None:
    """Write one checkpoint's worth of new rows as its own small file,
    named by an increasing sequence number.

    Cost is O(len(buffer)), independent of how much has accumulated so
    far -- unlike the old approach of merging into and rewriting the
    entire accumulated output every checkpoint, whose cost grows with
    total progress. That growth was confirmed as a real bottleneck on a
    1.4M-candidate run: the results parquet reached 436MB (errors CSV
    158MB) on Dropbox-synced storage, and the periodic full rewrite grew
    slow enough to stall every worker waiting on the main process between
    checkpoints. Shards are merged into the single canonical output file
    once, in _consolidate_shards -- paid once per run instead of once per
    checkpoint.
    """
    if not buffer:
        return
    os.makedirs(shard_dir, exist_ok=True)
    df = pd.DataFrame(buffer)
    path = os.path.join(shard_dir, f"{kind}_{seq:07d}.{ext}")
    _atomic_write(df, path, ext)


def _consolidate_shards(shard_dir: str, kind: str, existing: Optional[pd.DataFrame],
                        dedup_col: Optional[str], out_path: str, out_fmt: str) -> pd.DataFrame:
    """Merge all shard files for *kind* ('results' or 'errors') plus any
    pre-existing consolidated data into the single canonical file at
    *out_path*, then remove the shards.

    Called once at the end of a run, and once at the start of the next
    invocation (via _load_resume_state) to recover from a run that
    crashed before it got to consolidate -- either way, this full
    read+concat+write happens O(1) times per run, not once per checkpoint.
    """
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, f"{kind}_*")))
    parts = [existing] if existing is not None and len(existing) > 0 else []
    for p in shard_paths:
        parts.append(pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p))

    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if dedup_col and len(merged) > 0 and dedup_col in merged.columns:
        merged = merged.drop_duplicates(subset=[dedup_col], keep="last")

    if len(merged) > 0 or os.path.exists(out_path) or shard_paths:
        _atomic_write(merged, out_path, out_fmt)

    for p in shard_paths:
        os.remove(p)
    if os.path.isdir(shard_dir) and not os.listdir(shard_dir):
        os.rmdir(shard_dir)

    return merged


def _atomic_write(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Write *df* to *path* via a temp file + atomic rename.

    to_parquet/to_csv write directly to the target path; a kill or crash
    mid-write leaves a truncated, unrecoverable file (confirmed on a real
    run: a killed process left a parquet with a valid header but no
    footer, unreadable by any parquet reader). os.replace is atomic on the
    same filesystem, so a checkpoint file is always either the previous
    complete version or the new complete version, never a partial one.
    """
    tmp_path = path + ".tmp"
    if fmt == "parquet":
        df.to_parquet(tmp_path, index=False)
    else:
        df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _load_resume_state(results_path: str, errors_path: str, key_col: Optional[str],
                       save_format: str, shard_dir: Optional[str] = None):
    """Load existing results and the set of keys already attempted (success
    OR error) for resuming a checkpointed run.

    Resuming used to only skip already-*succeeded* keys, so any candidate
    that had errored (e.g. a real, reproducible geometry failure like "no
    valid recorder triplets") got silently retried -- and retried again on
    every subsequent resume -- accumulating duplicate rows in the errors
    file without making any real progress on the actual remaining work.

    If *shard_dir* is given and holds leftover per-checkpoint shard files
    (from a previous run that crashed before it got to consolidate them --
    see _consolidate_shards), they're merged in now. This is the one other
    point (besides end-of-run) where that full read+concat+write happens;
    it's paid once at process startup, not once per checkpoint.
    """
    key = key_col or "row_key"

    existing = None
    if os.path.exists(results_path):
        existing = pd.read_parquet(results_path) if save_format == "parquet" else pd.read_csv(results_path)
    err_existing = pd.read_csv(errors_path) if os.path.exists(errors_path) else None

    if shard_dir and os.path.isdir(shard_dir):
        results_ext = "parquet" if save_format == "parquet" else "csv"
        if glob.glob(os.path.join(shard_dir, "results_*")):
            existing = _consolidate_shards(shard_dir, "results", existing, key_col,
                                           results_path, results_ext)
        if glob.glob(os.path.join(shard_dir, "errors_*")):
            err_existing = _consolidate_shards(shard_dir, "errors", err_existing, key,
                                               errors_path, "csv")

    done_keys: set[str] = set()
    if existing is not None and key_col and key_col in existing.columns:
        done_keys |= set(existing[key_col].astype(str))
    if err_existing is not None and key in err_existing.columns:
        done_keys |= set(err_existing[key].astype(str))

    return existing, done_keys


# ---------- Serial runner (with checkpoints) ----------

def run_localizations_batch(
    df: pd.DataFrame,
    *,
    func,                       # localize_plot_withorig
    func_kwargs: dict,          # constant kwargs (bl_ref, wf, etc.)
    key_col: Optional[str] = None, # unique ID column; if None use index
    # runner settings
    checkpoint_every: int = 100,
    out_dir: str = "localization/results",
    out_basename: str = "localize",
    resume: bool = True,
    save_format: str = "parquet" # 'parquet' or 'csv'
):
    """
    Runs `func(row, **func_kwargs)` over all rows, collects results, checkpoints
    every `checkpoint_every` rows, and logs per-row exceptions without stopping.

    Returns (results_df, errors_df).
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{out_basename}.parquet" if save_format=="parquet" else f"{out_basename}.csv")
    errors_path  = os.path.join(out_dir, f"{out_basename}_errors.csv")
    log_path     = os.path.join(out_dir, f"{out_basename}.log")

    # Force plot_mode default to "none" unless caller overrides
    func_kwargs = dict(func_kwargs)
    func_kwargs.setdefault("plot_mode", "none")

    # Load existing (resume)
    if resume:
        existing, done_keys = _load_resume_state(results_path, errors_path, key_col, save_format)
    else:
        existing = None
        done_keys = set()

    results, errors = [], []
    processed_since_ckpt = 0

    def _checkpoint():
        nonlocal existing, results, errors
        if not results and not errors:
            return

        new_res_df = pd.DataFrame(results) if results else pd.DataFrame()
        if existing is not None and len(existing) > 0:
            if key_col and key_col in new_res_df.columns and key_col in existing.columns:
                merged = pd.concat([existing, new_res_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=[key_col], keep="last")
            else:
                merged = pd.concat([existing, new_res_df], ignore_index=True)
        else:
            merged = new_res_df

        _atomic_write(merged, results_path, save_format)
        existing = merged

        if errors:
            err_df = pd.DataFrame(errors)
            if os.path.exists(errors_path):
                old_err = pd.read_csv(errors_path)
                err_df = pd.concat([old_err, err_df], ignore_index=True)
            _atomic_write(err_df, errors_path, "csv")

        results, errors = [], []

    # Iteration helpers
    if key_col is None:
        df_iter = df.itertuples(index=True, name="Row")
        get_key = lambda t: t.Index
        get_row = lambda k: df.loc[k]
    else:
        df_iter = df.itertuples(index=False, name="Row")
        get_key = lambda t: getattr(t, key_col)
        get_row = lambda k: df[df[key_col] == k].iloc[0]

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"=== Batch start {datetime.now().isoformat()} ===\n")

        for t in df_iter:
            key = get_key(t)
            if done_keys and str(key) in done_keys:
                continue

            row = get_row(key)
            try:
                # add unique_id per row for collision-proof filenames
                res = func(row, unique_id=str(key), **func_kwargs)
                res_row = res if isinstance(res, dict) else {"result": res}
                res_row[(key_col or "row_key")] = key
                results.append(res_row)
            except Exception as e:
                errors.append({
                    (key_col or "row_key"): key,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                    "traceback": traceback.format_exc(),
                })
                logf.write(f"[{datetime.now().isoformat()}] ERROR key={key}: {type(e).__name__}: {e}\n")

            processed_since_ckpt += 1
            if processed_since_ckpt >= checkpoint_every:
                _checkpoint()
                processed_since_ckpt = 0

        _checkpoint()
        logf.write(f"=== Batch end {datetime.now().isoformat()} ===\n")

    final_results = existing if existing is not None else pd.DataFrame()
    final_errors  = pd.read_csv(errors_path) if os.path.exists(errors_path) else pd.DataFrame()
    return final_results, final_errors

# ---------- Worker for parallel ----------
#
# func/func_kwargs are loaded once per worker process via _pool_initializer,
# not re-sent with every payload. func_kwargs typically carries large shared
# objects (the wf catalog, deployments table, frequency profiles) -- passing
# them per-task means every one of N tasks pickles its own copy into the
# executor's queue, which for a large N (this batch runner is meant for
# tens/hundreds of thousands of candidates) causes the "N copies of a large
# DataFrame in flight at once" memory/CPU blowup this comment is here to
# avoid re-introducing.
_worker_func = None
_worker_kwargs = None
_worker_timeout = None


def _pool_initializer(func, func_kwargs, task_timeout=None) -> None:
    global _worker_func, _worker_kwargs, _worker_timeout
    _worker_func = func
    _worker_kwargs = func_kwargs
    _worker_timeout = task_timeout
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass


class _TaskTimeout(Exception):
    pass


def _worker_run_one(payload):
    """
    payload: (key, row_dict) -- func/func_kwargs come from _pool_initializer.

    The timeout is enforced HERE, inside the worker, via SIGALRM -- not only in
    the parent. The parent can stop *waiting* on a future but cannot stop the
    worker, so a parent-side-only timeout permanently burns a pool slot. Once
    n_workers tasks had overrun, throughput went to zero and every remaining
    candidate timed out: on 2026-09-20 a 38k-candidate run produced nothing at
    all after its first 90 minutes, and a later run's parent hung for ~3 h
    after writing its results because the wedged workers never exited.

    SIGALRM lands between bytecodes, which is enough here because the slow path
    is the Python-level peak-combination search. The parent keeps a longer
    backstop for anything this cannot interrupt.
    """
    key, row_dict = payload
    armed = False
    if _worker_timeout and hasattr(signal, "SIGALRM"):
        def _fire(signum, frame):
            raise _TaskTimeout(f"exceeded task_timeout={_worker_timeout}s")
        try:
            signal.signal(signal.SIGALRM, _fire)
            signal.setitimer(signal.ITIMER_REAL, float(_worker_timeout))
            armed = True
        except (ValueError, OSError):
            armed = False
    try:
        row = SimpleNamespace(**row_dict)
        res = _worker_func(row, unique_id=str(key), **_worker_kwargs)
        res_row = res if isinstance(res, dict) else {"result": res}
        return {"ok": True, "key": key, "result": res_row}
    except _TaskTimeout as e:
        return {
            "ok": False, "key": key,
            "error_type": "TimeoutError",
            "error_msg": str(e),
            "traceback": "",
        }
    except Exception as e:
        return {
            "ok": False, "key": key,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        if armed:
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except (ValueError, OSError):
                pass

# ---------- Parallel runner (process pool) ----------

def run_localizations_parallel(
    df: pd.DataFrame,
    *,
    func,                       # localize_plot_withorig (module-level for pickling)
    func_kwargs: dict,
    key_col: Optional[str] = None,
    n_workers: Optional[int] = None,
    checkpoint_every: int = 100,
    out_dir: str = "localization/results",
    out_basename: str = "localize_parallel",
    resume: bool = True,
    save_format: str = "parquet", # 'parquet' or 'csv'
    task_timeout: Optional[float] = None,  # seconds; None = no limit
):
    """
    Parallel execution with per-row exception handling and periodic checkpoints.
    Returns (results_df, errors_df).

    task_timeout : if set, a single task running longer than this is recorded
        as a TimeoutError and the runner moves on rather than waiting for it
        indefinitely. Caveat: ProcessPoolExecutor has no way to forcibly stop
        a task already running in a worker -- this stops *waiting* for it,
        it doesn't reclaim that worker slot, so a genuinely hung task costs
        one worker's worth of parallelism for the remainder of the run. Real
        protection against a single task blocking the *entire* batch
        (including at shutdown), not a guarantee every worker stays free.
    """
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, f"{out_basename}.parquet" if save_format=="parquet" else f"{out_basename}.csv")
    errors_path  = os.path.join(out_dir, f"{out_basename}_errors.csv")
    log_path     = os.path.join(out_dir, f"{out_basename}.log")
    shard_dir    = _shard_dir(out_dir, out_basename)
    results_ext  = "parquet" if save_format == "parquet" else "csv"

    # Force plot_mode default to "none" unless caller overrides
    func_kwargs = dict(func_kwargs)
    func_kwargs.setdefault("plot_mode", "none")

    # Load existing (resume). Also reconciles any leftover shards from a
    # previous run that crashed before it got to consolidate (see
    # _consolidate_shards) -- the only other point besides end-of-run
    # where the full read+concat+write happens, so it's still O(1) times
    # per run, not O(1) per checkpoint.
    if resume:
        existing, done_keys = _load_resume_state(results_path, errors_path, key_col, save_format,
                                                  shard_dir=shard_dir)
    else:
        existing = None
        done_keys = set()

    # Each checkpoint writes a new, independent shard file instead of
    # merging into and rewriting the entire accumulated output -- O(1) per
    # checkpoint regardless of total progress so far. See _write_shard's
    # docstring for why this replaced the old merge-and-rewrite approach.
    shard_seq = 0

    def write_checkpoint(results_buffer, errors_buffer):
        nonlocal shard_seq
        if not results_buffer and not errors_buffer:
            return
        _write_shard(results_buffer, shard_dir, "results", shard_seq, results_ext)
        _write_shard(errors_buffer, shard_dir, "errors", shard_seq, "csv")
        shard_seq += 1

    # Build payloads
    if key_col is None:
        iter_rows = df.reset_index().itertuples(index=False, name="Row")
        get_key = lambda t: getattr(t, "index")
        get_row_dict = lambda t: df.loc[getattr(t, "index")].to_dict()
    else:
        iter_rows = df.itertuples(index=False, name="Row")
        get_key = lambda t: getattr(t, key_col)
        get_row_dict = lambda t: t._asdict()

    payloads = []
    for t in iter_rows:
        key = get_key(t)
        if done_keys and str(key) in done_keys:
            continue
        payloads.append((key, get_row_dict(t)))

    if not payloads:
        final_results = existing if existing is not None else pd.DataFrame()
        final_errors  = pd.read_csv(errors_path) if os.path.exists(errors_path) else pd.DataFrame()
        return final_results, final_errors

    # Run pool
    results_buffer, errors_buffer = [], []
    processed_since_ckpt = 0

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"=== Parallel batch start {datetime.now().isoformat()} ===\n")

        # Not using `with ProcessPoolExecutor(...) as ex:` deliberately --
        # its __exit__ always calls shutdown(wait=True), which would block
        # on a hung worker regardless of what we do inside the block. We
        # need to choose wait=False ourselves if a timeout ever fired.
        # task_timeout goes to the workers too: they enforce it themselves via
        # SIGALRM and return a TimeoutError result, which keeps the pool slot
        # alive. See _worker_run_one.
        ex = ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_initializer,
                                 initargs=(func, func_kwargs, task_timeout))
        had_timeout = False
        try:
            key_by_future = {}
            submitted_at = {}

            # Submit a bounded in-flight window rather than the whole payload
            # list up front. submitted_at is used to detect task_timeout
            # overruns, so it must reflect when a task actually got a worker
            # slot -- submitting everything at once (with n_workers << number
            # of payloads) leaves most tasks sitting in the executor's internal
            # queue for far longer than task_timeout before a worker ever
            # touches them, so they'd get flagged as TimeoutError purely for
            # having queued, not for running long.
            backlog = iter(payloads)
            in_flight_target = max(n_workers * 2, 1)

            def _top_up():
                for pl in itertools.islice(backlog, in_flight_target - len(pending)):
                    fut = ex.submit(_worker_run_one, pl)
                    key_by_future[fut] = pl[0]
                    submitted_at[fut] = time.monotonic()
                    pending.add(fut)

            pending = set()
            _top_up()

            # Poll with a short wait instead of a blind as_completed() so we
            # can notice tasks that have overrun task_timeout even while
            # other tasks are still running.
            poll_interval = min(5.0, task_timeout) if task_timeout else 5.0

            while pending:
                done, pending = wait(pending, timeout=poll_interval, return_when=FIRST_COMPLETED)

                for fut in done:
                    key = key_by_future[fut]
                    out = fut.result()
                    if out["ok"]:
                        row_out = out["result"]
                        row_out[(key_col or "row_key")] = key
                        results_buffer.append(row_out)
                    else:
                        errors_buffer.append({
                            (key_col or "row_key"): key,
                            "error_type": out["error_type"],
                            "error_msg": out["error_msg"],
                            "traceback": out["traceback"],
                        })
                        logf.write(f"[{datetime.now().isoformat()}] ERROR key={key}: {out['error_type']}: {out['error_msg']}\n")
                    processed_since_ckpt += 1

                if task_timeout is not None:
                    now = time.monotonic()
                    # Backstop only. The worker's own SIGALRM should have
                    # returned a TimeoutError well before this; reaching here
                    # means the task was stuck somewhere the signal could not
                    # interrupt, and that worker really is lost. Give it a
                    # generous grace margin so we do not abandon (and leak) a
                    # slot that was about to report back on its own.
                    grace = max(30.0, 0.5 * task_timeout)
                    overrun = {f for f in pending
                               if now - submitted_at[f] > task_timeout + grace}
                    for fut in overrun:
                        pending.discard(fut)
                        had_timeout = True
                        key = key_by_future[fut]
                        errors_buffer.append({
                            (key_col or "row_key"): key,
                            "error_type": "TimeoutError",
                            "error_msg": f"exceeded task_timeout={task_timeout}s",
                            "traceback": "",
                        })
                        logf.write(f"[{datetime.now().isoformat()}] TIMEOUT key={key} "
                                  f"(> {task_timeout}s) -- moving on, worker slot may stay busy\n")
                        processed_since_ckpt += 1

                _top_up()

                if processed_since_ckpt >= checkpoint_every:
                    write_checkpoint(results_buffer, errors_buffer)
                    results_buffer, errors_buffer = [], []
                    processed_since_ckpt = 0

            # final checkpoint
            write_checkpoint(results_buffer, errors_buffer)
            logf.write(f"=== Parallel batch end {datetime.now().isoformat()} ===\n")
        finally:
            # Clean full shutdown normally (joins/reaps every worker); but
            # if a task timed out, some worker may be permanently wedged in
            # user code that never checks for a shutdown signal, so don't
            # block returning on it.
            ex.shutdown(wait=not had_timeout, cancel_futures=had_timeout)
            # had_timeout now means the worker-side alarm failed to interrupt
            # a task, which should be rare; the normal timeout path returns a
            # TimeoutError result and leaves the worker healthy.

    # Consolidate this run's shards into the single canonical output file --
    # the one full read+concat+write for this run's new rows, paid once
    # here rather than once per checkpoint. Read each output's pre-run
    # baseline fresh from disk (_load_resume_state already folded any
    # crash-leftover shards into it, but only tracked the baseline for
    # results in-memory, not errors).
    key = key_col or "row_key"
    existing_results = (
        (pd.read_parquet(results_path) if save_format == "parquet" else pd.read_csv(results_path))
        if os.path.exists(results_path) else None
    )
    existing_errors  = pd.read_csv(errors_path) if os.path.exists(errors_path) else None
    final_results = _consolidate_shards(shard_dir, "results", existing_results, key_col,
                                        results_path, results_ext)
    final_errors  = _consolidate_shards(shard_dir, "errors", existing_errors, key,
                                        errors_path, "csv")
    return final_results, final_errors
