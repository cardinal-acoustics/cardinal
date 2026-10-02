#!/usr/bin/env python3
# birdnet/run.py
#
# BirdNET one-pass per-directory extractor with minute-grid alignment.
# Outputs _bn_grid_wprobs/ sidecar parquets alongside the audio files.
#
# Integrated from misc/birdnet_embeddings/birdnet_onepass_perdir.py.
# Changes from original:
#   - parse_filename, GETRECORDER, RETURN_S4_TIME replaced with imports
#     from catalog.parse and config (no stubs / TODOs remain)
#   - RANCH_TZ replaced with SITE_TIMEZONE from config
#   - --out-dirname default set to _bn_grid_wprobs to match existing data
#   - Model path sourced from config.BIRDNET_MODEL_PATH
#
# Everything else (normalization, bandpass, tensor selection, sharding,
# manifest, parallelism) is unchanged from the original.

from __future__ import annotations
import os, math, argparse, json
import sys, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional, Iterable

import numpy as np
import pandas as pd
import librosa
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

from scipy.signal import butter, filtfilt

from catalog.parse import parse_filename, getrecorder, return_s4_time
from config import SITE_TIMEZONE, BIRDNET_MODEL_PATH, AUDIO_TOPDIRS

RANCH_TZ = SITE_TIMEZONE

# ===== Normalization helpers (closer to BirdNET-Analyzer) =====
class NormMode:
    NONE = "none"
    FILE_PEAK = "file-peak"
    FILE_PERCENTILE = "file-percentile"
    WINDOW_PEAK = "window-peak"  # retained for experiments

def log(msg: str) -> None:
    """Write a log message with a timestamp to stdout."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def robust_percentile_abs(y: np.ndarray, p: float = 99.0) -> float:
    """Return the p-th percentile of absolute amplitude."""
    if y.size == 0:
        return 0.0
    return float(np.percentile(np.abs(y), p))

def apply_file_normalization(y: np.ndarray,
                             mode: str = NormMode.FILE_PERCENTILE,
                             target_peak: float = 0.99,
                             percentile: float = 99.0,
                             eps: float = 1e-7) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    if mode == NormMode.NONE:
        return y
    if mode == NormMode.FILE_PEAK:
        m = float(np.max(np.abs(y)))
    elif mode == NormMode.FILE_PERCENTILE:
        m = robust_percentile_abs(y, percentile)
    else:
        return y
    if m < eps:
        return y
    return (y * (target_peak / m)).astype(np.float32, copy=False)

def scale_to_unit(y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    m = float(np.max(np.abs(y)))
    if m < eps:
        return y.astype(np.float32)
    return (y / m).astype(np.float32)


def l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)

# Prefer ai_edge_litert; then TensorFlow Lite; finally tflite_runtime.
BACKEND = None
try:
    import ai_edge_litert.interpreter as tflite
    BACKEND = "ai_edge_litert"
except Exception as _e1:
    try:
        import tensorflow.lite as tflite
        BACKEND = "tensorflow.lite"
    except Exception as _e2:
        try:
            import tflite_runtime.interpreter as tflite
            BACKEND = "tflite_runtime"
        except Exception as _e3:
            raise ImportError(
                "No TFLite runtime found. Please install one of:\n"
                "  pip install ai-edge-litert      # recommended on macOS arm64\n"
                "  pip install tflite-runtime      # fallback\n"
                "  pip install tensorflow-macos    # then use tensorflow.lite\n"
            ) from _e1

print(f"[birdnet.run] TFLite backend: {BACKEND}")


AUDIO_EXTS = (".wav", ".WAV")
OUT_DIRNAME = "_bn_grid_wprobs"

def apply_bandpass(y: np.ndarray, sr: int, hp: float = 150.0, lp: float = 15000.0, order: int = 4) -> np.ndarray:
    if sr <= 0 or y.size == 0:
        return y.astype(np.float32)
    nyq = 0.5 * sr
    low = max(1.0, float(hp)) / nyq
    high = min(float(lp), nyq * 0.99) / nyq
    if not (0.0 < low < high < 1.0):
        return y.astype(np.float32)
    b, a = butter(order, [low, high], btype="band")
    try:
        y_f = filtfilt(b, a, y).astype(np.float32)
    except Exception:
        y_f = y.astype(np.float32)
    return y_f


def grid_offset_seconds(dt: datetime, grid: int = 3) -> float:
    s = dt.second + dt.microsecond / 1e6
    r = s % grid
    return 0.0 if r == 0.0 else (grid - r)

def prepare_interpreter(model_path: str, threads: int = 4,
                        logits_index: int | None = None,
                        emb_index: int | None = None,
                        n_classes: int | None = None):
    # experimental_preserve_all_tensors=True is required to access intermediate
    # tensors (e.g. the embedding at index 545) after invoke(). Without it,
    # TFLite recycles intermediate tensor memory and get_tensor() returns null.
    try:
        interp = tflite.Interpreter(model_path=model_path, num_threads=threads,
                                    experimental_preserve_all_tensors=True)
    except TypeError:
        # Older TFLite versions that don't support this flag preserve tensors by default
        interp = tflite.Interpreter(model_path=model_path, num_threads=threads)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()
    all_tensors = interp.get_tensor_details()

    if logits_index is not None and emb_index is not None:
        return interp, in_det, logits_index, emb_index

    def shp_tuple(d):
        return tuple(d.get("shape", ()))

    def flat_size_from_shape(shp):
        if len(shp) == 1:
            return int(shp[0])
        if len(shp) == 2 and shp[0] == 1:
            return int(shp[1])
        return None

    def is_float_vec(d):
        return d.get("dtype") == np.float32 and flat_size_from_shape(shp_tuple(d)) is not None

    float_outs = [d for d in out_det if d.get("dtype") == np.float32]

    logits_idx = None
    if n_classes is not None:
        for d in float_outs:
            if flat_size_from_shape(shp_tuple(d)) == n_classes:
                logits_idx = d["index"]
                break

    if logits_idx is None:
        candidates = [(flat_size_from_shape(shp_tuple(d)) or -1, d["index"]) for d in float_outs]
        candidates = [(n, idx) for (n, idx) in candidates if n is not None]
        if not candidates:
            raise RuntimeError("No suitable float outputs found in model.")
        logits_idx = max(candidates, key=lambda x: x[0])[1]

    preferred_sizes = (1024, 2048, 1536, 512, 4096, 256)
    preferred_name_substrings = (
        "GLOBAL_AVG_POOL/Mean",
        "global_average_pool",
        "GLOBAL_AVG_POOL",
        "POST_ACT_1",
        "POST_BN_1",
        "Mean",
    )

    emb_idx = None
    for pref in preferred_sizes:
        for d in float_outs:
            if d["index"] == logits_idx:
                continue
            if flat_size_from_shape(shp_tuple(d)) == pref:
                emb_idx = d["index"]
                break
        if emb_idx is not None:
            break

    if emb_idx is None and float_outs:
        others = [
            (flat_size_from_shape(shp_tuple(d)) or -1, d["index"]) for d in float_outs if d["index"] != logits_idx
        ]
        others = [(n, idx) for (n, idx) in others if n is not None]
        if others:
            emb_idx = max(others, key=lambda x: x[0])[1]

    if emb_idx is None:
        logits_shape = None
        for d in all_tensors:
            if d.get("index") == logits_idx:
                logits_shape = shp_tuple(d)
                break
        all_float_vecs = [d for d in all_tensors if is_float_vec(d) and d.get("index") != logits_idx]

        named_pref = [
            d for d in all_float_vecs
            if any(s in d.get("name", "") for s in preferred_name_substrings)
            and flat_size_from_shape(shp_tuple(d)) in preferred_sizes
        ]
        if named_pref:
            emb_idx = max(named_pref, key=lambda d: flat_size_from_shape(shp_tuple(d)) or 0)["index"]
        else:
            named_any = [d for d in all_float_vecs if any(s in d.get("name", "") for s in preferred_name_substrings)]
            if named_any:
                emb_idx = max(named_any, key=lambda d: flat_size_from_shape(shp_tuple(d)) or 0)["index"]
            else:
                sized = [
                    (flat_size_from_shape(shp_tuple(d)) or -1, d["index"]) for d in all_float_vecs
                ]
                sized = [(n, idx) for (n, idx) in sized if n is not None]
                if not sized:
                    raise RuntimeError("No suitable embedding tensor found.")
                emb_idx = max(sized, key=lambda x: x[0])[1]

        try:
            info = next(d for d in all_tensors if d.get("index") == emb_idx)
            print(
                f"      [choose] fallback(all_tensors) emb_idx={emb_idx} name='{info.get('name','')}' shape={shp_tuple(info)}",
                flush=True,
            )
        except Exception:
            pass

    return interp, in_det, logits_idx, emb_idx

def run_windows(interp, in_det, logits_idx, emb_idx, y: np.ndarray, sr: int,
                win: float = 3.0, hop: float = 3.0, start_shift: float = 0.0,
                window_norm_mode: str = NormMode.NONE):
    W = int(round(win * sr))
    H = int(round(hop * sr))
    i0 = int(round(start_shift * sr))

    in_shape = tuple(in_det["shape"])
    want_1xT = (1, W)
    want_1x1xT = (1, 1, W)

    for a in range(i0, max(0, len(y) - W) + 1, H):
        b = a + W
        chunk = y[a:b]
        if len(chunk) < W:
            pad = np.zeros(W, np.float32); pad[:len(chunk)] = chunk
            chunk = pad
        chunk = chunk.astype(np.float32, copy=False)
        if window_norm_mode == NormMode.WINDOW_PEAK:
            mx = float(np.max(np.abs(chunk)))
            if mx > 0.0:
                chunk = 0.99 * (chunk / mx)
        x = np.expand_dims(chunk, 0).astype(np.float32)
        if in_shape == want_1x1xT:
            x = np.expand_dims(x, 1)
        elif in_shape == want_1xT:
            pass
        else:
            x = x.reshape(in_shape)
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        if a == i0:
            e_shp = tuple(interp.get_tensor_details()[emb_idx]["shape"])
            l_shp = tuple(interp.get_tensor_details()[logits_idx]["shape"])
            print(f"      [tensors] emb_idx={emb_idx} shape={e_shp} | logits_idx={logits_idx} shape={l_shp}", flush=True)

        emb = interp.get_tensor(emb_idx).squeeze().astype(np.float32)
        logits = interp.get_tensor(logits_idx).squeeze().astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-logits))
        yield a / sr, b / sr, emb, p


def find_audio_dirs(root: Path | str,
                    *,
                    recursive_exclude: bool = True,
                    exclude_filename: str = "donotindex.txt") -> Iterable[Path]:
    root = Path(root)
    excl_lower = exclude_filename.lower()
    for dp, dn, fn in os.walk(root):
        d = Path(dp)
        if any(f.lower() == excl_lower for f in fn):
            if recursive_exclude:
                dn[:] = []
            continue
        if any(f.endswith(AUDIO_EXTS) for f in fn):
            yield d

def process_dir(dir_path: Path, model_path: str, labels_csv: Optional[Path],
                out_dirname: str, sr: int, threads: int,
                include_probs: bool, shard_rows: int, overwrite: bool,
                logits_index: Optional[int], emb_index: Optional[int], labels_count: Optional[int],
                do_bandpass: bool, hp_cut: float, lp_cut: float, bp_order: int,
                do_gain_norm: bool, l2_norm_emb: bool,
                file_norm_mode: str, file_norm_percentile: float,
                window_norm_mode: str, prob_threshold: float) -> tuple[str, int, int]:
    out_dir = dir_path / out_dirname
    out_dir.mkdir(exist_ok=True)
    manifest = out_dir / "manifest.json"
    done = set()
    if manifest.exists() and not overwrite:
        try:
            done = set(json.loads(manifest.read_text()).get("processed_files", []))
        except Exception:
            pass

    n_classes = None
    if labels_csv is not None and Path(labels_csv).exists():
        try:
            import csv
            with open(labels_csv, "r", encoding="utf-8") as f:
                n_classes = sum(1 for line in f if line.strip())
        except Exception:
            n_classes = None
    if labels_count is not None:
        n_classes = labels_count if labels_count else n_classes

    interp, in_det, logits_idx, emb_idx = prepare_interpreter(
        model_path, threads=threads,
        logits_index=logits_index,
        emb_index=emb_index,
        n_classes=n_classes
    )
    print(f"      [choose] logits_idx={logits_idx} emb_idx={emb_idx} n_classes={n_classes}", flush=True)

    files = sorted([p for p in dir_path.iterdir() if p.suffix in AUDIO_EXTS and p.is_file()])
    if not files:
        return str(dir_path), 0, 0

    norm_desc = f"norm={file_norm_mode if do_gain_norm else 'off'}"
    bp_desc = f"BP={hp_cut:.0f}-{lp_cut:.0f}Hz/order{bp_order}" if do_bandpass else "BP=off"
    print(f"[start] {dir_path} | wavs={len(files)} | backend={BACKEND} | threads={threads} | {bp_desc} | {norm_desc}", flush=True)
    t_dir_start = time.time()

    shard_id = 0
    rows = []
    total_windows = 0
    processed_files = set(done)

    for wav in files:
        print(f"  [file] {dir_path.name}/{wav.name} ...", flush=True)
        t_file_start = time.time()
        if (str(wav) in processed_files) and not overwrite:
            continue

        meta = parse_filename(str(wav))
        dt0 = meta["start_datetime"]
        if pd.isna(dt0):
            continue

        try:
            y, _ = librosa.load(str(wav), sr=sr, mono=True)
        except Exception as exc:
            log(f"[warn] skipping {wav} due to error: {exc}")
            continue

        if do_bandpass:
            y = apply_bandpass(y, sr, hp=hp_cut, lp=lp_cut, order=bp_order)

        if do_gain_norm:
            y = apply_file_normalization(
                y,
                mode=file_norm_mode,
                target_peak=0.99,
                percentile=file_norm_percentile
            )
        shift = grid_offset_seconds(dt0, grid=3)
        wcount = 0

        for ws, we, emb, prob in run_windows(interp, in_det, logits_idx, emb_idx, y, sr,
                                             win=3.0, hop=3.0, start_shift=shift,
                                             window_norm_mode=window_norm_mode):
            if l2_norm_emb:
                emb = l2_normalize_vec(emb)
            row = {
                "file": str(wav),
                "abs_start_iso": dt0.isoformat(),
                "win_start": float(ws),
                "win_end": float(we),
                "embedding": emb.tolist(),
            }
            if include_probs:
                prob_arr = np.asarray(prob, dtype=np.float32)
                mask = prob_arr >= prob_threshold
                if np.any(mask):
                    idx = np.nonzero(mask)[0]
                    vals = prob_arr[mask]
                    row["probs"] = [[int(i), float(v)] for i, v in zip(idx, vals)]
                else:
                    row["probs"] = []
            rows.append(row)
            total_windows += 1
            wcount += 1
            if (wcount % 200) == 0:
                elapsed = time.time() - t_file_start
                rate = wcount / elapsed if elapsed > 0 else float('inf')
                print(f"      windows={wcount} ({rate:.1f}/s)", flush=True)

            if len(rows) >= shard_rows:
                shard_id += 1
                shard_path = out_dir / f"emb_{shard_id:05d}.parquet"
                try:
                    pd.DataFrame(rows).to_parquet(shard_path, index=False)
                except Exception as e:
                    raise RuntimeError(f"Failed writing Parquet shard: {shard_path} ({len(rows)} rows). "
                                       f"Install pyarrow: pip install pyarrow. Original error: {e}")
                print(f"    [write] {shard_path}  rows={len(rows)}", flush=True)
                rows.clear()

        elapsed_file = time.time() - t_file_start
        print(f"  [done] {wav.name} | windows={wcount} | {elapsed_file:.1f}s", flush=True)
        processed_files.add(str(wav))

    elapsed_dir = time.time() - t_dir_start
    print(f"[dir done] {dir_path} | files={len(processed_files - done)} | windows={total_windows} | {elapsed_dir:.1f}s", flush=True)

    if rows:
        shard_id += 1
        shard_path = out_dir / f"emb_{shard_id:05d}.parquet"
        try:
            pd.DataFrame(rows).to_parquet(shard_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed writing Parquet shard: {shard_path} ({len(rows)} rows). "
                               f"Install pyarrow: pip install pyarrow. Original error: {e}")
        print(f"    [write] {shard_path}  rows={len(rows)}", flush=True)
        rows.clear()

    manifest.write_text(json.dumps({"processed_files": sorted(processed_files)}, indent=2))
    return str(dir_path), len(processed_files), total_windows


def main():
    ap = argparse.ArgumentParser("BirdNET one-pass per-directory extractor with minute-grid alignment & donotindex skip.")
    ap.add_argument("roots", nargs="*", metavar="DIR",
                    help="Root directories to process. Defaults to AUDIO_TOPDIRS from config.")
    ap.add_argument("--model", default=BIRDNET_MODEL_PATH, help="Path to BirdNET .tflite model")
    ap.add_argument("--labels", type=Path, default=None, help="(Optional) labels CSV (not required)")
    ap.add_argument("--out-dirname", default=OUT_DIRNAME, help="Per-directory output subfolder name")
    ap.add_argument("--sr", type=int, default=48000, help="Target sample rate (Hz)")
    ap.add_argument("--jobs", type=int, default=6, help="Directories to process in parallel")
    ap.add_argument("--threads", type=int, default=3, help="TFLite threads per process")
    ap.add_argument("--include-probs", action="store_true", help="Store class probabilities vectors")
    ap.add_argument("--prob-threshold", type=float, default=0.10,
                    help="When --include-probs is set, store only class probabilities >= this threshold (default 0.10)")
    ap.add_argument("--shard-rows", type=int, default=200_000, help="Rows per Parquet shard")
    ap.add_argument("--overwrite", action="store_true", help="Reprocess even if manifest lists the file")
    ap.add_argument("--logits-index", type=int, default=None, help="Explicit tensor index for logits")
    ap.add_argument("--emb-index", type=int, default=None, help="Explicit tensor index for embeddings")
    ap.add_argument("--labels-count", type=int, default=None, help="Number of classes")
    ap.add_argument("--no-bandpass", action="store_true", help="Disable BirdNET-style bandpass (default on)")
    ap.add_argument("--hp", type=float, default=150.0, help="High-pass cutoff in Hz (default 150.0)")
    ap.add_argument("--lp", type=float, default=15000.0, help="Low-pass cutoff in Hz (default 15000.0)")
    ap.add_argument("--bp-order", type=int, default=4, help="Butterworth bandpass order (default 4)")
    ap.add_argument("--no-gain-norm", action="store_true",
                    help="Disable amplitude normalization (default on as file-percentile).")
    ap.add_argument("--l2-normalize-emb", action="store_true", help="L2-normalize each embedding vector")
    ap.add_argument("--norm", dest="norm_mode", choices=["none", "file-peak", "file-percentile", "window-peak"],
                    default="file-percentile")
    ap.add_argument("--norm-percentile", type=float, default=99.0)
    ap.add_argument("--list-outputs", action="store_true", help="List model output tensors and exit")
    args = ap.parse_args()

    model_path = str(Path(args.model).resolve())

    if args.list_outputs:
        it = tflite.Interpreter(model_path=model_path, num_threads=max(1, args.threads))
        it.allocate_tensors()
        outs = it.get_output_details()
        dets = it.get_tensor_details()
        print("Model outputs:")
        for d in outs:
            idx = d["index"]; name = d.get("name", f"<unnamed_{idx}>"); dtype = d.get("dtype"); shp = tuple(d.get("shape", ()))
            print(f"  index={idx:4d}  name={name}  dtype={dtype}  shape={shp}")
        print("\nAll float tensors:")
        for d in dets:
            idx = d["index"]; name = d.get("name", f"<unnamed_{idx}>"); shp = tuple(d.get("shape", ())); dtype = d.get("dtype")
            if dtype == np.float32:
                print(f"  index={idx:4d}  name={name}  dtype={dtype}  shape={shp}")
        return

    roots = [Path(r) for r in args.roots] if args.roots else [Path(r) for r in AUDIO_TOPDIRS]
    todo = list(find_audio_dirs_from_roots(roots))
    if not todo:
        print("No eligible audio directories found.")
        return

    print(f"Found {len(todo)} audio directories. Out subfolder: '{args.out_dirname}'", flush=True)
    ok = total_files = total_windows = 0

    logits_idx = args.logits_index
    emb_idx = args.emb_index
    if logits_idx is None and emb_idx is None:
        # Defaults for BirdNET GLOBAL 6K V2.4 FP32
        logits_idx = 546  # (1, 6522) Identity
        emb_idx = 545     # (1, 1024) GLOBAL_AVG_POOL/Mean

    file_norm_mode = args.norm_mode
    file_norm_percentile = float(args.norm_percentile)
    window_norm_mode = NormMode.NONE if args.norm_mode != NormMode.WINDOW_PEAK else NormMode.WINDOW_PEAK

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [
            ex.submit(
                process_dir, d, model_path, args.labels, args.out_dirname,
                args.sr, args.threads, args.include_probs, args.shard_rows, args.overwrite,
                logits_idx, emb_idx, args.labels_count,
                (not args.no_bandpass), args.hp, args.lp, args.bp_order,
                (not args.no_gain_norm), args.l2_normalize_emb,
                file_norm_mode, file_norm_percentile, window_norm_mode,
                args.prob_threshold
            )
            for d in todo
        ]
        for f in as_completed(futs):
            d, nfiles, nwins = f.result()
            ok += 1; total_files += nfiles; total_windows += nwins
            print(f"[Done {ok}/{len(todo)}] {d} -> files:{nfiles} windows:{nwins}", flush=True)

    print(f"\nAll done. Directories: {ok}, Files: {total_files}, Windows: {total_windows}")


def find_audio_dirs_from_roots(roots: list[Path]) -> Iterable[Path]:
    for root in roots:
        yield from find_audio_dirs(root)


if __name__ == "__main__":
    main()
