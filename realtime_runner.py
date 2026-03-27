#!/usr/bin/env python3
"""
realtime_runner.py — Real-time multi-duck animation runner
===========================================================

Runs a WorldSimulation in real-time, saving each rendered frame to disk so it
can be viewed in an external GIF viewer or image displayer that auto-refreshes.

Also supports single-duck real-time mode via the TimelineSequencer for
lightweight testing without the full world overhead.

Usage (multi-duck world)
────────────────────────
    python realtime_runner.py [image1.png] [image2.png] ...

    # With explicit options:
    python realtime_runner.py beetle.png hanshaw.png --fps 12 --output live.gif

Usage (single duck, quick test)
────────────────────────────────
    python realtime_runner.py --single beetle.png

The runner keeps a rolling buffer of frames and writes a short GIF every N
seconds so the output file stays fresh without growing unboundedly.

API
───
    run_realtime(images, num_ducks=3, fps=12, output="live.gif",
                 buffer_seconds=4.0, max_seconds=0)

    run_realtime_single(image, emotion="neutral", fps=12,
                        output="single_live.gif", max_seconds=0)
"""

import sys
import os
import time
import random
import signal
import argparse
from collections import deque

from PIL import Image

from world_simulation   import WorldSimulation, save_gif, _make_placeholder
from timeline_sequencer import TimelineSequencer, save_frames_as_gif


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FPS           = 12
DEFAULT_OUTPUT        = "realtime_output.gif"
DEFAULT_BUFFER_SECS   = 4.0     # write a GIF every N seconds of accumulated frames
DEFAULT_MAX_SECONDS   = 0.0     # 0 = run forever (Ctrl-C to stop)


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-DUCK REAL-TIME RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_realtime(
    images:          dict[str, Image.Image],
    num_ducks:       int   = 3,
    fps:             int   = DEFAULT_FPS,
    output:          str   = DEFAULT_OUTPUT,
    buffer_seconds:  float = DEFAULT_BUFFER_SECS,
    max_seconds:     float = DEFAULT_MAX_SECONDS,
    canvas_size:     tuple[int, int] = (720, 480),
    draw_debug:      bool  = False,
    verbose:         bool  = True,
) -> None:
    """
    Run a multi-duck world in real-time, writing GIF snapshots to `output`.

    Ctrl-C to stop — the current buffer is flushed on exit.
    """
    world = WorldSimulation.create_new(
        num_ducks   = num_ducks,
        images      = images,
        canvas_size = canvas_size,
        draw_shadows = True,
        draw_names   = True,
        draw_debug   = draw_debug,
    )

    dt             = 1.0 / fps
    buffer_frames  = max(1, round(buffer_seconds * fps))
    frame_buffer   = deque(maxlen=buffer_frames)
    elapsed        = 0.0
    wall_start     = time.perf_counter()
    last_gif_t     = 0.0
    frame_count    = 0

    # Inject random events periodically
    event_pool   = ["user_message", "attention", "random_thought",
                    "loud_noise", "idle_timeout", "praise"]
    next_event_t = random.uniform(1.5, 3.5)

    def _flush_gif() -> None:
        nonlocal last_gif_t
        if frame_buffer:
            save_gif(list(frame_buffer), output, fps=fps)
            if verbose:
                print(f"[realtime] Wrote {len(frame_buffer)}-frame GIF → {output}")
        last_gif_t = elapsed

    # Graceful shutdown on Ctrl-C
    def _on_sigint(sig, frame):
        print("\n[realtime] Interrupted — flushing final buffer …")
        _flush_gif()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    if verbose:
        print(f"[realtime] Starting {num_ducks}-duck world at {fps} fps")
        print(f"  Output file: {output}")
        print(f"  Press Ctrl-C to stop.\n")

    while True:
        t0    = time.perf_counter()
        frame = world.step(dt)
        frame_buffer.append(frame)
        frame_count += 1
        elapsed     += dt

        # Periodic random event injection into a random duck
        if elapsed >= next_event_t:
            event     = random.choice(event_pool)
            duck_list = list(world.ducks.values())
            if duck_list:
                target = random.choice(duck_list)
                target.trigger(event)
                if verbose:
                    print(f"[realtime] t={elapsed:.2f}s  event={event!r} → {target.id}")
            next_event_t = elapsed + random.uniform(1.5, 4.0)

        # Write GIF snapshot every buffer_seconds
        if elapsed - last_gif_t >= buffer_seconds:
            _flush_gif()

        # Stop after max_seconds (0 = infinite)
        if max_seconds > 0 and elapsed >= max_seconds:
            _flush_gif()
            break

        # Throttle to target FPS
        spent   = time.perf_counter() - t0
        sleep_t = max(0.0, dt - spent)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if verbose:
        total_wall = time.perf_counter() - wall_start
        print(f"[realtime] Finished. {frame_count} frames in {total_wall:.1f}s wall time.")


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-DUCK REAL-TIME RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_realtime_single(
    image:           Image.Image,
    emotion:         str   = "neutral",
    fps:             int   = DEFAULT_FPS,
    output:          str   = "single_live.gif",
    buffer_seconds:  float = DEFAULT_BUFFER_SECS,
    max_seconds:     float = DEFAULT_MAX_SECONDS,
    verbose:         bool  = True,
) -> None:
    """
    Run a single-duck timeline in real-time, writing rolling GIF snapshots.

    Random events are injected to keep things lively.
    """
    seq = TimelineSequencer(image, emotion=emotion, fps=fps)

    # Rolling event injection
    event_pool      = ["user_message", "random_thought", "attention",
                       "loud_noise", "idle", "praise"]
    script_so_far   = [{"t": 0.0, "event": "idle"}]
    next_event_offset = random.uniform(1.5, 3.0)

    dt             = 1.0 / fps
    buffer_frames  = max(1, round(buffer_seconds * fps))
    frame_buffer   = deque(maxlen=buffer_frames)
    elapsed        = 0.0
    last_gif_t     = 0.0
    frame_count    = 0

    # Reload the sequencer with an evolving script every few seconds
    _reload_interval = 3.0
    last_reload_t    = -_reload_interval   # trigger immediately

    def _maybe_reload():
        nonlocal last_reload_t, script_so_far, next_event_offset
        if elapsed - last_reload_t < _reload_interval:
            return
        # Build a fresh script window starting from now
        script_window = []
        t = 0.0
        while t < _reload_interval + 1.0:
            ev = random.choice(event_pool)
            script_window.append({"t": t, "event": ev})
            t += random.uniform(0.8, 2.5)
        seq.load_script(script_window)
        last_reload_t = elapsed

    def _flush_gif():
        nonlocal last_gif_t
        if frame_buffer:
            save_frames_as_gif(list(frame_buffer), output, fps=fps)
            if verbose:
                print(f"[realtime-single] Wrote {len(frame_buffer)}-frame GIF → {output}")
        last_gif_t = elapsed

    def _on_sigint(sig, frame):
        print("\n[realtime-single] Interrupted — flushing …")
        _flush_gif()
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    if verbose:
        print(f"[realtime-single] Single-duck mode, emotion={emotion!r}")
        print(f"  Output: {output}  |  Ctrl-C to stop\n")

    while True:
        t0 = time.perf_counter()

        _maybe_reload()
        frame = seq._render_frame(dt)
        frame_buffer.append(frame)
        frame_count += 1
        elapsed     += dt

        if elapsed - last_gif_t >= buffer_seconds:
            _flush_gif()

        if max_seconds > 0 and elapsed >= max_seconds:
            _flush_gif()
            break

        spent   = time.perf_counter() - t0
        sleep_t = max(0.0, dt - spent)
        if sleep_t > 0:
            time.sleep(sleep_t)

    if verbose:
        print(f"[realtime-single] Done. {frame_count} frames rendered.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time duck animation runner"
    )
    p.add_argument("images", nargs="*",
                   help="Input image paths (multiple = multi-duck world)")
    p.add_argument("--single",  action="store_true",
                   help="Single-duck mode (uses first image only)")
    p.add_argument("--emotion", default="neutral",
                   help="Starting emotion for single-duck mode")
    p.add_argument("--fps",     type=int,   default=DEFAULT_FPS)
    p.add_argument("--output",  default=DEFAULT_OUTPUT)
    p.add_argument("--buffer",  type=float, default=DEFAULT_BUFFER_SECS,
                   help="GIF buffer window in seconds")
    p.add_argument("--max",     type=float, default=DEFAULT_MAX_SECONDS,
                   help="Stop after N seconds (0 = infinite)")
    p.add_argument("--canvas",  default="720x480",
                   help="Canvas size WxH (default: 720x480)")
    p.add_argument("--debug",   action="store_true",
                   help="Show debug overlay on ducks")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse canvas
    try:
        cw, ch = map(int, args.canvas.split("x"))
        canvas_size = (cw, ch)
    except ValueError:
        canvas_size = (720, 480)

    # Load images
    images: dict[str, Image.Image] = {}
    for i, path in enumerate(args.images[:8]):
        try:
            images[f"duck_{i}"] = Image.open(path).convert("RGBA")
            print(f"  Loaded duck_{i} ← {path}")
        except Exception as exc:
            print(f"  WARNING: could not load {path}: {exc}")

    # Single-duck mode
    if args.single:
        if not images:
            print("No images provided for --single mode; using placeholder.")
            img = _make_placeholder("duck_0")
        else:
            img = next(iter(images.values()))
        run_realtime_single(
            image           = img,
            emotion         = args.emotion,
            fps             = args.fps,
            output          = args.output,
            buffer_seconds  = args.buffer,
            max_seconds     = args.max,
        )
        return

    # Multi-duck world mode
    num_ducks = max(2, len(images)) if images else 3
    run_realtime(
        images         = images,
        num_ducks      = num_ducks,
        fps            = args.fps,
        output         = args.output,
        buffer_seconds = args.buffer,
        max_seconds    = args.max,
        canvas_size    = canvas_size,
        draw_debug     = args.debug,
    )


if __name__ == "__main__":
    main()
