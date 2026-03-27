#!/usr/bin/env python3
"""
persistence_layer.py — Save / load the duck world to / from disk (JSON only)
=============================================================================

Provides three top-level functions:

    save_world(world, path)           → write world.json
    load_world(path, images)          → WorldSimulation
    create_world(num_ducks, images)   → fresh WorldSimulation (then save it)

No database, no binary formats — just human-readable JSON.

The persisted document looks like:

    {
      "version": "1.0",
      "saved_at": "<ISO timestamp>",
      "canvas_size": [800, 500],
      "clock": { … },
      "relationships": { … },
      "ducks": {
        "duck_0": { … },
        "duck_1": { … }
      },
      "frame_count": 1234
    }
"""

import json
import os
import shutil
import time
from datetime import datetime, timezone
from typing import Optional

from PIL import Image

from world_simulation import WorldSimulation, save_gif, _make_placeholder


SCHEMA_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_world(
    world: WorldSimulation,
    path:  str = "world.json",
) -> None:
    """
    Serialise the entire world state to a JSON file.

    Writes to a temp file first, then atomically renames — avoids
    corruption if the process is interrupted mid-write.
    """
    doc = world.to_dict()
    doc["version"]  = SCHEMA_VERSION
    doc["saved_at"] = datetime.now(timezone.utc).isoformat()

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, default=_json_default)
        # Atomic replace
        shutil.move(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"[persistence] World saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_world(
    path:         str,
    images:       dict[str, Image.Image] = None,
    *,
    draw_shadows: bool = True,
    draw_names:   bool = True,
    draw_debug:   bool = False,
    draw_beak:    bool = True,
    background:   Image.Image | None = None,
) -> WorldSimulation:
    """
    Reconstruct a WorldSimulation from a previously saved JSON file.

    images  dict mapping duck_id → portrait Image.
            Any duck_id not present in images will receive a coloured
            placeholder instead of raising an error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"World file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    _check_version(doc)

    images = images or {}
    world  = WorldSimulation.from_dict(
        doc,
        images       = images,
        draw_shadows = draw_shadows,
        draw_names   = draw_names,
        draw_debug   = draw_debug,
        draw_beak    = draw_beak,
        background   = background,
    )

    saved_at = doc.get("saved_at", "unknown")
    print(f"[persistence] World loaded from {path}  (saved {saved_at})")
    return world


# ─────────────────────────────────────────────────────────────────────────────
#  CREATE (fresh world)
# ─────────────────────────────────────────────────────────────────────────────

def create_world(
    num_ducks:   int                        = 3,
    images:      dict[str, Image.Image]     = None,
    canvas_size: tuple[int, int]            = (800, 500),
    save_path:   str | None                 = None,
    **renderer_kwargs,
) -> WorldSimulation:
    """
    Create a brand-new WorldSimulation and optionally save it immediately.

    save_path  If given, the world is saved to that path after creation.
    """
    world = WorldSimulation.create_new(
        num_ducks   = num_ducks,
        images      = images or {},
        canvas_size = canvas_size,
        **renderer_kwargs,
    )

    if save_path:
        save_world(world, save_path)

    return world


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD-OR-CREATE  (convenience wrapper for "resume or start fresh")
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_world(
    path:        str,
    num_ducks:   int                        = 3,
    images:      dict[str, Image.Image]     = None,
    canvas_size: tuple[int, int]            = (800, 500),
    **kwargs,
) -> WorldSimulation:
    """
    Try to load an existing world from `path`.
    If the file doesn't exist, create a fresh one and save it.

    This is the recommended entry point for persistent-world scenarios.
    """
    if os.path.exists(path):
        return load_world(path, images=images, **kwargs)
    else:
        print(f"[persistence] No world file at {path} — creating new world.")
        return create_world(
            num_ducks   = num_ducks,
            images      = images,
            canvas_size = canvas_size,
            save_path   = path,
            **kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SNAPSHOT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def snapshot(
    world:    WorldSimulation,
    gif_path: str,
    duration: float = 4.0,
    fps:      int   = 12,
    verbose:  bool  = False,
) -> list[Image.Image]:
    """
    Simulate `duration` seconds of the world and export as a GIF.

    The world state is mutated in-place (time advances, memories form).
    Call save_world() afterwards to persist the updated state.

    Returns the list of rendered frames.
    """
    frames = world.simulate(duration=duration, fps=fps, verbose=verbose)
    save_gif(frames, gif_path, fps=fps)
    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _check_version(doc: dict) -> None:
    """Warn (but don't crash) if the saved schema version differs."""
    saved = doc.get("version", "unknown")
    if saved != SCHEMA_VERSION:
        import warnings
        warnings.warn(
            f"World file version mismatch: expected {SCHEMA_VERSION!r}, "
            f"got {saved!r}.  Loading anyway.",
            UserWarning,
            stacklevel=3,
        )


def _json_default(obj):
    """Fallback JSON serialiser for non-standard types."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    world_file = sys.argv[1] if len(sys.argv) > 1 else "world.json"
    gif_file   = sys.argv[2] if len(sys.argv) > 2 else "snapshot.gif"

    # Load portrait images from current directory
    import os
    images = {}
    for i, f in enumerate(sorted(p for p in os.listdir(".") if p.endswith(".png"))[:4]):
        try:
            images[f"duck_{i}"] = Image.open(f).convert("RGBA")
            print(f"  duck_{i} ← {f}")
        except Exception as e:
            print(f"  WARNING: {e}")

    world = load_or_create_world(world_file, num_ducks=3, images=images)

    print(f"Simulating 4 seconds …")
    snapshot(world, gif_file, duration=4.0, fps=12, verbose=True)

    # Save updated world state (memory / relationships evolved during sim)
    save_world(world, world_file)
    print(f"World state saved back to {world_file}")
