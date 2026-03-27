#!/usr/bin/env python3
"""
world_simulation.py — Central multi-duck world orchestrator
=============================================================

Manages:
  • A collection of DuckEntity objects
  • Per-pair RelationshipEngine (shared across all ducks)
  • InteractionEngine (gaze / proximity scanning)
  • WorldClock for persistent simulation time
  • WorldRenderer for compositing frames

API
───
    sim = WorldSimulation(canvas_size=(800, 500))
    sim.add_duck("duck_0", duck_entity, image)

    # Snapshot GIF (offline)
    frames = sim.simulate(duration=5.0, fps=12)
    frames[0].save("world.gif", save_all=True, append_images=frames[1:],
                   loop=0, duration=83)

    # Continuous loop (for debug / real-time)
    sim.run_realtime()

    # Serialisation
    state = sim.to_dict()
    sim2  = WorldSimulation.from_dict(state, images={"duck_0": img0, ...})
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from duck_entity     import DuckEntity, PersonalityProfile
from interaction_engine import InteractionEngine
from relationship_engine import RelationshipEngine
from world_clock     import WorldClock
from renderer        import WorldRenderer


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FPS      = 12          # frames per second for GIF export
REALTIME_FPS     = 12          # frames per second in real-time mode
MAX_DUCKS        = 8           # hard cap for sanity


# ─────────────────────────────────────────────────────────────────────────────
#  WORLD SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

class WorldSimulation:
    """
    Central brain for the multi-duck world.

    Holds all ducks, runs per-tick updates, drives the interaction engine,
    and hands frames to the renderer.
    """

    def __init__(
        self,
        canvas_size:   tuple[int, int] = (800, 500),
        draw_shadows:  bool = True,
        draw_names:    bool = True,
        draw_debug:    bool = False,
        draw_beak:     bool = True,
        background:    Image.Image | None = None,
    ) -> None:
        self.canvas_size = canvas_size

        # Entity registry
        self.ducks:  dict[str, DuckEntity] = {}
        self.images: dict[str, Image.Image] = {}   # base portrait per duck_id

        # Shared relationship engine (all ducks share one)
        self.relationships = RelationshipEngine()

        # Interaction engine (proximity + gaze event broker)
        self.interaction = InteractionEngine()

        # World clock (tracks sim time — can be persisted)
        self.clock = WorldClock()

        # Renderer
        self.renderer = WorldRenderer(
            canvas_size  = canvas_size,
            background   = background,
            draw_shadows = draw_shadows,
            draw_names   = draw_names,
            draw_debug   = draw_debug,
            draw_beak    = draw_beak,
        )

        # Running totals for debug output
        self._frame_count = 0

    # ── Duck management ───────────────────────────────────────────────────────

    def add_duck(
        self,
        duck_id:   str,
        entity:    DuckEntity,
        image:     Image.Image,
    ) -> None:
        """Register a duck with its portrait image."""
        if len(self.ducks) >= MAX_DUCKS:
            raise RuntimeError(f"World already has {MAX_DUCKS} ducks (hard cap).")
        self.ducks[duck_id]  = entity
        self.images[duck_id] = image
        self.renderer.add_duck(duck_id, image)

    def remove_duck(self, duck_id: str) -> None:
        self.ducks.pop(duck_id, None)
        self.images.pop(duck_id, None)
        self.renderer.remove_duck(duck_id)

    # ── Simulation step ───────────────────────────────────────────────────────

    def step(self, dt: float) -> Image.Image:
        """
        Advance the world by dt seconds and return a rendered frame.

        Order of operations each tick:
          1. Advance world clock
          2. Update every duck (emotion blend, lipsync, gaze, wander)
          3. Run interaction engine (proximity / gaze events → reactions)
          4. Collect per-duck CharacterState
          5. Render frame
        """
        # 1. Clock
        self.clock.tick(dt)

        duck_list = list(self.ducks.values())

        # 2. Per-duck update — pass other ducks so gaze targeting works
        for duck in duck_list:
            duck.update(dt, duck_list)

        # 3. Interaction engine — detects proximity / gaze, fires events
        self.interaction.update(duck_list, self.relationships, dt,
                               sim_time=self.clock.sim_time)

        # 4. Apply relationship influence on each duck's gaze selection
        self._apply_relationship_gaze(duck_list)

        # 5. Render
        frame = self.renderer.render_frame(duck_list)
        self._frame_count += 1
        return frame

    def _apply_relationship_gaze(self, ducks: list[DuckEntity]) -> None:
        """
        Bias duck gaze toward friends; slightly repel from rivals.
        Called every tick; uses low-weight nudges to avoid being jittery.
        """
        for duck in ducks:
            if duck.gaze_target_id is not None:
                continue    # already locked on a target
            # Find closest friend
            best_id    = None
            best_score = 0.15  # minimum score to bother looking
            for other in ducks:
                if other.id == duck.id:
                    continue
                score = self.relationships.get_score(duck.id, other.id)
                if score > best_score:
                    best_score = score
                    best_id    = other.id
            if best_id is not None and random.random() < 0.05:
                duck.gaze_target_id = best_id

    # ── GIF simulation ────────────────────────────────────────────────────────

    def simulate(
        self,
        duration: float = 5.0,
        fps:      int   = DEFAULT_FPS,
        verbose:  bool  = False,
    ) -> list[Image.Image]:
        """
        Run the simulation for `duration` seconds at `fps` and return all frames.

        The returned list can be exported directly as an animated GIF.
        """
        dt          = 1.0 / fps
        total_frames = max(1, round(duration * fps))
        frames       = []

        for i in range(total_frames):
            frame = self.step(dt)
            frames.append(frame)
            if verbose and (i % fps == 0):
                print(f"[world] t={self.clock.sim_time:.2f}s  frame {i+1}/{total_frames}")

        return frames

    # ── Real-time loop ────────────────────────────────────────────────────────

    def run_realtime(
        self,
        fps:         int  = REALTIME_FPS,
        max_seconds: float = 0.0,   # 0 = infinite
        on_frame = None,            # optional callback(frame_image)
    ) -> None:
        """
        Blocking real-time simulation loop.

        on_frame is called with each rendered PIL Image so the caller can
        display it however they like (e.g. save to disk, push to a viewer).
        """
        dt         = 1.0 / fps
        elapsed    = 0.0
        wall_start = time.perf_counter()

        while True:
            t0    = time.perf_counter()
            frame = self.step(dt)
            elapsed += dt

            if on_frame is not None:
                on_frame(frame)

            if max_seconds > 0 and elapsed >= max_seconds:
                break

            # Throttle to target FPS
            spent = time.perf_counter() - t0
            sleep_t = max(0.0, dt - spent)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise the world state to a JSON-compatible dict."""
        return {
            "canvas_size":    list(self.canvas_size),
            "clock":          self.clock.to_dict(),
            "relationships":  self.relationships.to_dict(),
            "ducks": {
                duck_id: duck.to_dict()
                for duck_id, duck in self.ducks.items()
            },
            "frame_count": self._frame_count,
        }

    @classmethod
    def from_dict(
        cls,
        data:   dict,
        images: dict[str, Image.Image],
        *,
        draw_shadows: bool = True,
        draw_names:   bool = True,
        draw_debug:   bool = False,
        draw_beak:    bool = True,
        background:   Image.Image | None = None,
    ) -> "WorldSimulation":
        """Reconstruct a WorldSimulation from a serialised dict."""
        canvas_size = tuple(data.get("canvas_size", (800, 500)))
        world       = cls(
            canvas_size  = canvas_size,
            draw_shadows = draw_shadows,
            draw_names   = draw_names,
            draw_debug   = draw_debug,
            draw_beak    = draw_beak,
            background   = background,
        )

        # Restore clock
        if "clock" in data:
            world.clock = WorldClock.from_dict(data["clock"])

        # Restore relationships
        if "relationships" in data:
            world.relationships = RelationshipEngine.from_dict(data["relationships"])
            world.interaction   = InteractionEngine()

        # Restore ducks
        for duck_id, duck_data in data.get("ducks", {}).items():
            duck  = DuckEntity.from_dict(duck_data)
            image = images.get(duck_id, _make_placeholder(duck_id))
            world.add_duck(duck_id, duck, image)

        world._frame_count = data.get("frame_count", 0)
        return world

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def create_new(
        cls,
        num_ducks:   int                         = 3,
        images:      dict[str, Image.Image]      = None,
        canvas_size: tuple[int, int]             = (800, 500),
        emotions:    list[str]                   = None,
        **kwargs,
    ) -> "WorldSimulation":
        """
        Create a fresh WorldSimulation with `num_ducks` randomly placed ducks.

        images  dict mapping duck_id → portrait Image.  If fewer images than
                ducks are given, coloured placeholders fill the gaps.
        """
        images    = images   or {}
        emotions  = emotions or ["neutral", "happy", "curious", "calm"]
        world     = cls(canvas_size=canvas_size, **kwargs)
        margin    = 0.12

        for i in range(num_ducks):
            duck_id  = f"duck_{i}"
            x        = margin + random.random() * (1.0 - 2 * margin)
            y        = margin + random.random() * (1.0 - 2 * margin)
            emotion  = random.choice(emotions)
            duck     = DuckEntity(duck_id, name=duck_id, position=(x, y),
                                  emotion=emotion)
            image    = images.get(duck_id, _make_placeholder(duck_id))
            world.add_duck(duck_id, duck, image)

        return world


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PLACEHOLDER_COLORS = [
    (255, 220, 60),   # yellow
    (100, 200, 255),  # blue
    (255, 130, 90),   # orange
    (180, 255, 130),  # green
    (240, 160, 240),  # pink
    (200, 200, 200),  # grey
    (255, 100, 100),  # red
    (100, 100, 255),  # indigo
]

def _make_placeholder(duck_id: str, size: int = 128) -> Image.Image:
    """Make a coloured circle placeholder portrait for a duck."""
    idx   = int(duck_id.split("_")[-1]) % len(_PLACEHOLDER_COLORS) if "_" in duck_id else 0
    color = _PLACEHOLDER_COLORS[idx] + (255,)
    img   = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw  = ImageDraw.Draw(img)
    r     = size // 2 - 4
    cx    = cy = size // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return img


def save_gif(
    frames:   list[Image.Image],
    path:     str,
    fps:      int = DEFAULT_FPS,
    optimize: bool = True,
) -> None:
    """Save a list of RGBA frames as an animated GIF (shared palette)."""
    if not frames:
        raise ValueError("No frames to save.")

    # Convert to RGB for palette quantisation
    rgb_frames = [f.convert("RGB") for f in frames]

    master    = rgb_frames[0].quantize(colors=256)
    pal_frames = [master] + [
        f.quantize(palette=master, dither=0) for f in rgb_frames[1:]
    ]

    duration_ms = int(1000 / fps)
    pal_frames[0].save(
        path,
        format       = "GIF",
        save_all     = True,
        append_images = pal_frames[1:],
        loop         = 0,
        duration     = duration_ms,
        optimize     = optimize,
    )
    print(f"[world] Saved {len(frames)}-frame GIF → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os

    # Gather image paths from command line or fall back to any PNGs in the repo
    image_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not image_paths:
        image_paths = sorted(
            f for f in os.listdir(".") if f.lower().endswith(".png")
        )[:3]

    if not image_paths:
        print("Usage: python world_simulation.py [image1.png] [image2.png] ...")
        print("No images found — generating placeholder world...")

    images = {}
    for i, p in enumerate(image_paths[:MAX_DUCKS]):
        try:
            images[f"duck_{i}"] = Image.open(p).convert("RGBA")
            print(f"  Loaded duck_{i} from {p}")
        except Exception as exc:
            print(f"  WARNING: could not load {p}: {exc}")

    num_ducks = max(2, len(images)) if images else 3
    world     = WorldSimulation.create_new(
        num_ducks   = num_ducks,
        images      = images,
        canvas_size = (720, 480),
        draw_shadows = True,
        draw_names   = True,
    )

    print(f"\nSimulating {num_ducks} ducks for 4 seconds at 12 fps …")
    frames = world.simulate(duration=4.0, fps=12, verbose=True)

    out = "world_output.gif"
    save_gif(frames, out, fps=12)
    print(f"Done! → {out}")
