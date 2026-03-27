#!/usr/bin/env python3
"""
batch_runner.py — Batch processor for the Duckling GIF Animation Engine
=========================================================================

Processes every supported image in an input directory and writes animated GIFs
to an output directory.  Supports parallel processing via a thread pool.

Per-image emotion sidecar
─────────────────────────
Place a plain-text file with the same stem and extension ".emotion" alongside
any image to override the global emotion for that specific file:

  duck_01.png        ← source image
  duck_01.emotion    ← contains the word "angry" (no quotes, no newline needed)

CLI usage
─────────
  python batch_runner.py ./input_images/ ./output_gifs/ [options]

  Options:
    --emotion EMOTION       Force emotion for all images (default: random)
    --frames N              Frames per GIF (default: 80)
    --duration MS           Milliseconds per frame (default: 80)
    --workers N             Parallel worker threads (default: 4)
    --emotions-json PATH    Custom emotions JSON file
    --transition EMOTION    End-emotion for all transitions
    --transition-frames N   Frames for emotion transition (default: 20)
    --loop N                GIF loop count; 0 = infinite (default: 0)
    --no-summary            Suppress the error-detail summary at the end
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image

from emotion_engine import EMOTIONS, list_emotions, load_custom_emotions
from duck_animator import generate_frames, save_gif

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDECAR EMOTION LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _sidecar_emotion(image_path: Path) -> Optional[str]:
    """
    Look for a <stem>.emotion sidecar file next to the image.
    Returns the stripped lowercase emotion string, or None if absent.
    """
    sidecar = image_path.with_suffix(".emotion")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip().lower()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-IMAGE WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _process_one(
    img_path:         Path,
    out_path:         Path,
    emotion:          str,
    n_frames:         int,
    duration:         int,
    transition_to:    Optional[str],
    transition_frames: int,
    custom_emotions:  Optional[dict],
    loop:             int,
) -> dict:
    """
    Process a single image.  Called from the thread pool.
    Returns a result dict suitable for summary reporting.
    """
    result = {
        "input":   str(img_path),
        "output":  str(out_path),
        "emotion": emotion,
        "status":  "pending",
        "error":   None,
        "elapsed": 0.0,
    }
    t0 = time.monotonic()

    try:
        img       = Image.open(img_path)
        emo_dict  = custom_emotions or EMOTIONS

        # Per-image sidecar overrides the global emotion setting
        sidecar = _sidecar_emotion(img_path)
        if sidecar:
            if sidecar in emo_dict:
                emotion = sidecar
                result["emotion"] = emotion
            else:
                print(f"  ⚠️  Unknown sidecar emotion '{sidecar}' — ignoring.")

        # Resolve random *per image* (called inside the worker to be thread-safe)
        if emotion == "random":
            emotion = random.choice(list(emo_dict.keys()))
            result["emotion"] = emotion

        frames = generate_frames(
            base_image=img,
            emotion_name=emotion,
            n_frames=n_frames,
            transition_to=transition_to,
            transition_frames=transition_frames,
            custom_emotions=custom_emotions,
        )
        save_gif(frames, str(out_path), duration=duration, loop=loop)

        result["status"]  = "ok"

    except Exception as exc:
        result["status"] = "error"
        result["error"]  = str(exc)

    result["elapsed"] = time.monotonic() - t0
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    input_dir:         str,
    output_dir:        str,
    emotion:           str           = "random",
    n_frames:          int           = 80,
    duration:          int           = 80,
    transition_to:     Optional[str] = None,
    transition_frames: int           = 20,
    custom_emotions_path: Optional[str] = None,
    workers:           int           = 4,
    loop:              int           = 0,
    print_summary:     bool          = True,
) -> list[dict]:
    """
    Process all supported images in input_dir, writing GIFs to output_dir.

    Uses a ThreadPoolExecutor so multiple images are processed concurrently.
    Returns a list of per-image result dicts.

    Each result dict has keys:
      input, output, emotion, status ('ok'|'error'), error, elapsed
    """
    in_path  = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect image files
    image_files = sorted(
        f for f in in_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_files:
        print(f"⚠️  No supported images found in '{input_dir}'.")
        return []

    # Load custom emotions once (shared across all workers)
    custom_emotions = None
    if custom_emotions_path:
        custom_emotions = load_custom_emotions(custom_emotions_path)
        print(f"📖 Custom emotions loaded from: {custom_emotions_path}")

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n🦆 Duckling Batch Runner")
    print(f"   Input   : {in_path.resolve()}")
    print(f"   Output  : {out_path.resolve()}")
    print(f"   Images  : {len(image_files)}")
    print(f"   Emotion : {emotion}")
    print(f"   Frames  : {n_frames} × {duration} ms/frame")
    print(f"   Workers : {workers}")
    print()

    results: list[dict] = []
    t_batch = time.monotonic()

    # ── Thread pool ──────────────────────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_file = {}
        for img_file in image_files:
            gif_file = out_path / (img_file.stem + ".gif")
            fut = pool.submit(
                _process_one,
                img_file, gif_file,
                emotion, n_frames, duration,
                transition_to, transition_frames,
                custom_emotions, loop,
            )
            future_to_file[fut] = img_file

        for idx, fut in enumerate(as_completed(future_to_file), 1):
            img_name = future_to_file[fut].name
            res      = fut.result()
            results.append(res)
            icon = "✅" if res["status"] == "ok" else "❌"
            print(
                f"  [{idx:3d}/{len(image_files)}] {icon} {img_name:<40}"
                f"  emotion={res['emotion']:<12}  {res['elapsed']:.1f}s"
            )

    total_time = time.monotonic() - t_batch
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    err_count  = len(results) - ok_count

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("── Batch Summary " + "─" * 44)
    print(f"   Processed  : {ok_count}/{len(image_files)} images")
    print(f"   Errors     : {err_count}")
    avg = total_time / len(image_files) if image_files else 0
    print(f"   Total time : {total_time:.1f}s  ({avg:.1f}s avg/image)")
    print(f"   Output dir : {out_path.resolve()}")

    if print_summary and err_count > 0:
        print()
        print("── Errors " + "─" * 51)
        for r in results:
            if r["status"] != "ok":
                print(f"   {Path(r['input']).name}: {r['error']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    emotions_str = ", ".join(list_emotions()) + ", random"
    parser = argparse.ArgumentParser(
        prog="batch_runner",
        description="🦆 Duckling GIF Animation Engine — Batch Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available emotions: {emotions_str}",
    )
    parser.add_argument("input_dir",  help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory for output GIFs")
    parser.add_argument(
        "--emotion", default="random",
        help="Emotion for all images, or 'random' for per-image random (default: random)",
    )
    parser.add_argument("--frames",    type=int, default=80,
                        help="Frames per GIF (default: 80)")
    parser.add_argument("--duration",  type=int, default=80,
                        help="Milliseconds per frame (default: 80)")
    parser.add_argument("--workers",   type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    parser.add_argument("--emotions-json", default=None,
                        help="Path to custom emotions JSON file")
    parser.add_argument("--transition", default=None,
                        help="Emotion to transition toward at end of each GIF")
    parser.add_argument("--transition-frames", type=int, default=20,
                        help="Frames for emotion transition (default: 20)")
    parser.add_argument("--loop",  type=int, default=0,
                        help="GIF loop count, 0=infinite (default: 0)")
    parser.add_argument("--no-summary", action="store_true",
                        help="Suppress error-detail summary at the end")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        emotion=args.emotion,
        n_frames=args.frames,
        duration=args.duration,
        transition_to=args.transition,
        transition_frames=args.transition_frames,
        custom_emotions_path=args.emotions_json,
        workers=args.workers,
        loop=args.loop,
        print_summary=not args.no_summary,
    )
