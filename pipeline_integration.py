#!/usr/bin/env python3
"""
pipeline_integration.py — Integration hook for the Duckling GIF Animation Engine
==================================================================================

Drop-in integration for existing duck image pipelines.

Python API
──────────
  from pipeline_integration import animate_image, DuckAnimationPipeline

  # One-liner integration into existing generator
  image = generate_duck()                          # your existing code
  animate_image(image, "duck.gif", emotion="happy")

  # Class-based pipeline for repeated calls with shared config
  pipe = DuckAnimationPipeline(emotion="happy", n_frames=60)
  pipe.run("duck1.png", "duck1.gif")
  pipe.run("duck2.png", "duck2.gif")

CLI usage
─────────
  python pipeline_integration.py input.png output.gif [emotion] [options]
  python pipeline_integration.py ./input_dir/ ./output_dir/ --batch --emotion happy

  Options:
    --frames N              Frames (default: 80)
    --duration MS           ms per frame (default: 80)
    --transition EMOTION    Emotion to blend toward at the end
    --transition-frames N   How many end-frames to use for the blend (default: 20)
    --emotions-json PATH    Custom emotions JSON file
    --loop N                GIF loop count; 0 = infinite (default: 0)
    --batch                 Treat input as directory; output as directory
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from emotion_engine import EMOTIONS, list_emotions, load_custom_emotions
from duck_animator import generate_frames, save_gif


# ─────────────────────────────────────────────────────────────────────────────
#  PYTHON API — animate_image()
# ─────────────────────────────────────────────────────────────────────────────

def animate_image(
    source:                  Union[str, Image.Image],
    output_path:             str,
    emotion:                 str            = "neutral",
    n_frames:                int            = 80,
    duration:                int            = 80,
    transition_to:           Optional[str]  = None,
    transition_frames:       int            = 20,
    custom_emotions_path:    Optional[str]  = None,
    loop:                    int            = 0,
) -> str:
    """
    Animate a static duck image and export it as an animated GIF.

    Parameters
    ──────────
    source               Path string OR PIL.Image
    output_path          Destination .gif file path
    emotion              Emotion name ('neutral', 'happy', …) or 'random'
    n_frames             Total animation frames (default: 80 ≈ 6.4 s at 80 ms)
    duration             Milliseconds per frame (60–100 ms recommended)
    transition_to        Optional emotion to blend toward at end of animation
    transition_frames    Number of end-frames used for the blend
    custom_emotions_path Path to a JSON file with custom/override emotions
    loop                 GIF loop count; 0 = infinite

    Returns
    ───────
    str — the output_path that was written
    """
    # ── Load image ─────────────────────────────────────────────────────────
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input image not found: {source}")
        img = Image.open(source)
        print(f"📂 Loaded: {source}  ({img.width}×{img.height} px)")
    elif isinstance(source, Image.Image):
        img = source
    else:
        raise TypeError(
            f"source must be a file path (str) or PIL.Image, got {type(source).__name__}"
        )

    # ── Custom emotions ────────────────────────────────────────────────────
    custom_emotions = None
    if custom_emotions_path:
        custom_emotions = load_custom_emotions(custom_emotions_path)
        print(f"📖 Loaded custom emotions from: {custom_emotions_path}")

    emotion_dict = custom_emotions or EMOTIONS

    # ── Resolve 'random' emotion ───────────────────────────────────────────
    if emotion == "random":
        emotion = random.choice(list(emotion_dict.keys()))
        print(f"🎲 Random emotion selected: {emotion}")
    elif emotion not in emotion_dict:
        print(f"⚠️  Emotion '{emotion}' not found — defaulting to 'neutral'.")
        emotion = "neutral"

    # ── Validate transition target ─────────────────────────────────────────
    if transition_to and transition_to not in emotion_dict:
        print(f"⚠️  Transition target '{transition_to}' not found — disabling transition.")
        transition_to = None

    # ── Ensure output directory exists ────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── Animate ───────────────────────────────────────────────────────────
    label = f"{emotion}" + (f" → {transition_to}" if transition_to else "")
    print(f"🎬 Generating {n_frames} frames  [emotion: {label}]")

    frames = generate_frames(
        base_image=img,
        emotion_name=emotion,
        n_frames=n_frames,
        transition_to=transition_to,
        transition_frames=transition_frames,
        custom_emotions=custom_emotions,
    )

    # ── Export ────────────────────────────────────────────────────────────
    save_gif(frames, output_path, duration=duration, loop=loop)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS-BASED PIPELINE  (shared config across multiple images)
# ─────────────────────────────────────────────────────────────────────────────

class DuckAnimationPipeline:
    """
    Reusable pipeline object that holds a fixed configuration and can process
    one or many images with a single method call.

    Example
    ───────
        pipe = DuckAnimationPipeline(emotion="happy", n_frames=60, duration=70)

        # Single image
        pipe.run("duck.png", "duck.gif")

        # Entire directory
        pipe.run_batch("./input_images/", "./output_gifs/")
    """

    def __init__(
        self,
        emotion:                 str           = "neutral",
        n_frames:                int           = 80,
        duration:                int           = 80,
        transition_to:           Optional[str] = None,
        transition_frames:       int           = 20,
        custom_emotions_path:    Optional[str] = None,
        loop:                    int           = 0,
    ):
        self.emotion               = emotion
        self.n_frames              = n_frames
        self.duration              = duration
        self.transition_to         = transition_to
        self.transition_frames     = transition_frames
        self.custom_emotions_path  = custom_emotions_path
        self.loop                  = loop

    def run(self, source: Union[str, Image.Image], output_path: str) -> str:
        """Animate a single image.  Returns output_path."""
        return animate_image(
            source=source,
            output_path=output_path,
            emotion=self.emotion,
            n_frames=self.n_frames,
            duration=self.duration,
            transition_to=self.transition_to,
            transition_frames=self.transition_frames,
            custom_emotions_path=self.custom_emotions_path,
            loop=self.loop,
        )

    def run_batch(
        self,
        input_dir:  str,
        output_dir: str,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> list[str]:
        """
        Process all images in input_dir and write GIFs to output_dir.
        Returns a list of successfully written output paths.
        """
        in_path  = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            f for f in in_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

        if not image_files:
            print(f"⚠️  No images found in {input_dir}")
            return []

        results: list[str] = []
        for idx, img_file in enumerate(image_files, 1):
            out_file = out_path / (img_file.stem + ".gif")
            # If emotion is 'random', pick independently for each image
            emo = (
                random.choice(list(EMOTIONS.keys()))
                if self.emotion == "random"
                else self.emotion
            )
            print(f"\n[{idx}/{len(image_files)}] {img_file.name}  →  {out_file.name}")
            try:
                self._run_single(img_file, out_file, emo)
                results.append(str(out_file))
            except Exception as exc:
                print(f"  ❌ Failed: {exc}")

        print(f"\n✅ Batch done: {len(results)}/{len(image_files)} images.")
        return results

    def _run_single(self, img_file: Path, out_file: Path, emo: str) -> None:
        animate_image(
            source=str(img_file),
            output_path=str(out_file),
            emotion=emo,
            n_frames=self.n_frames,
            duration=self.duration,
            transition_to=self.transition_to,
            transition_frames=self.transition_frames,
            custom_emotions_path=self.custom_emotions_path,
            loop=self.loop,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    emotions_str = ", ".join(list_emotions()) + ", random"
    parser = argparse.ArgumentParser(
        prog="pipeline_integration",
        description="🦆 Duckling GIF Animation Engine — Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available emotions: {emotions_str}",
    )
    parser.add_argument("input",  help="Input image path  (or directory if --batch)")
    parser.add_argument("output", help="Output GIF path   (or directory if --batch)")
    parser.add_argument(
        "emotion", nargs="?", default="random",
        help="Emotion name or 'random' (default: random)",
    )
    parser.add_argument("--frames",     type=int,  default=80,
                        help="Number of frames (default: 80)")
    parser.add_argument("--duration",   type=int,  default=80,
                        help="Milliseconds per frame (default: 80)")
    parser.add_argument("--transition", type=str,  default=None,
                        help="Emotion to transition toward at end")
    parser.add_argument("--transition-frames", type=int, default=20,
                        help="Frames for transition (default: 20)")
    parser.add_argument("--emotions-json", type=str, default=None,
                        help="Path to custom emotions JSON override file")
    parser.add_argument("--loop",   type=int,  default=0,
                        help="GIF loop count (0 = infinite, default: 0)")
    parser.add_argument("--batch", action="store_true",
                        help="Treat input/output as directories for batch mode")
    return parser


def _cli() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.batch:
        pipe = DuckAnimationPipeline(
            emotion=args.emotion,
            n_frames=args.frames,
            duration=args.duration,
            transition_to=args.transition,
            transition_frames=args.transition_frames,
            custom_emotions_path=args.emotions_json,
            loop=args.loop,
        )
        pipe.run_batch(args.input, args.output)
    else:
        animate_image(
            source=args.input,
            output_path=args.output,
            emotion=args.emotion,
            n_frames=args.frames,
            duration=args.duration,
            transition_to=args.transition,
            transition_frames=args.transition_frames,
            custom_emotions_path=args.emotions_json,
            loop=args.loop,
        )


if __name__ == "__main__":
    _cli()
