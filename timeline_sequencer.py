#!/usr/bin/env python3
"""
timeline_sequencer.py — Scripted single-duck animation timeline
================================================================

Provides a high-level API for creating scripted GIF animations where a duck
reacts to a sequence of timed events:

    script = [
        {"t": 0.0, "event": "idle"},
        {"t": 1.0, "event": "user_message"},
        {"t": 2.5, "event": "loud_noise"},
        {"t": 3.5, "event": "idle"},
    ]

    frames = animate(image, duration=5.0, script=script, emotion="neutral")
    frames[0].save("output.gif", save_all=True, append_images=frames[1:],
                   loop=0, duration=83)

Or via the CLI:

    python timeline_sequencer.py duck.png output.gif happy --duration 4.0

Architecture
────────────
• EmotionBlender — smoothly interpolates emotion parameters over time.
• LipsyncScheduler — manages mouth state from the lipsync engine.
• TimelineSequencer — drives everything from a script, one frame at a time.

All drawing is delegated to duck_animator.render_single_frame() — the
sequencer only computes *what* to draw, not *how* to draw it.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from emotion_engine  import EMOTIONS, interpolate_emotions, get_random_emotion
from duck_animator   import (
    FaceGeometry,
    CharacterState,
    MouthState,
    generate_blink_seq,
    generate_pupil_track,
    generate_head_bob,
    render_single_frame,
    _generate_reactive_blink_seq,
)

try:
    from reaction_engine import ReactionEngine, get_behavior_params, EVENT_CATALOG
    _HAS_REACTION = True
except ImportError:
    _HAS_REACTION = False
    ReactionEngine    = None     # type: ignore
    EVENT_CATALOG     = {}       # type: ignore
    def get_behavior_params(_: str) -> dict:   # type: ignore
        return {"head_bob_scale": 1.0, "eye_jitter_scale": 1.0, "blink_emphasis": 1.0}

try:
    from lipsync_engine import LipsyncEngine, MouthState as LipMouthState
    _HAS_LIPSYNC = True
except ImportError:
    _HAS_LIPSYNC = False
    LipsyncEngine = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
#  EMOTION BLENDER
# ─────────────────────────────────────────────────────────────────────────────

class EmotionBlender:
    """
    Smoothly interpolates between emotion parameter sets over time.

    Uses first-order IIR smoothing:
        value += (target - value) * alpha
    where alpha is derived from the configured blend_speed.
    """

    BLEND_SPEED = 3.0       # 1/seconds — higher = snappier transitions

    def __init__(self, emotion_name: str = "neutral") -> None:
        params           = EMOTIONS.get(emotion_name, EMOTIONS["neutral"])
        self.eye_open    = float(params["eye_open"])
        self.brow_angle  = float(params["brow_angle"])
        self.brow_raise  = float(params.get("brow_raise", 0.0))
        self.mouth_open  = float(params.get("mouth_open", 0.0))
        self._target     = dict(params)
        self._current_name = emotion_name

    def set_emotion(self, emotion_name: str) -> None:
        """Schedule a transition to a new emotion."""
        params = EMOTIONS.get(emotion_name, EMOTIONS["neutral"])
        self._target       = dict(params)
        self._current_name = emotion_name

    def update(self, dt: float) -> None:
        """Advance one timestep, blending toward the current target."""
        alpha = min(1.0, self.BLEND_SPEED * dt)
        self.eye_open   += (self._target["eye_open"]   - self.eye_open)   * alpha
        self.brow_angle += (self._target["brow_angle"] - self.brow_angle) * alpha
        self.brow_raise += (self._target.get("brow_raise", 0.0) - self.brow_raise) * alpha
        self.mouth_open += (self._target.get("mouth_open", 0.0) - self.mouth_open) * alpha

    @property
    def emotion_name(self) -> str:
        return self._current_name

    @property
    def look_bias(self) -> tuple[float, float]:
        return (
            float(self._target.get("look_bias", [0.0, 0.0])[0]),
            float(self._target.get("look_bias", [0.0, 0.0])[1]),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  LIPSYNC SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class LipsyncScheduler:
    """
    Maintains a queue of speech events and produces per-frame MouthState.

    If LipsyncEngine is not available, returns a simple closed-mouth state.
    """

    def __init__(self, fps: int = 12) -> None:
        self._fps      = fps
        self._engine   = LipsyncEngine(fps=fps) if _HAS_LIPSYNC else None
        self._track:   list = []
        self._idx:     int  = 0
        self._speaking = False

    def start_speech(self, duration_frames: int = 36) -> None:
        if self._engine is not None:
            # generate_track takes seconds, not frames
            duration_secs = duration_frames / self._fps
            self._track   = self._engine.generate_track(duration_secs)
        else:
            self._track   = []
        self._idx     = 0
        self._speaking = True

    def stop_speech(self) -> None:
        self._speaking = False
        self._track    = []
        self._idx      = 0

    def tick(self) -> MouthState:
        """Advance one frame and return current MouthState."""
        if not self._speaking or not self._track:
            return MouthState()   # closed

        if self._idx >= len(self._track):
            self._speaking = False
            return MouthState()

        state      = self._track[self._idx]
        self._idx += 1
        return state

    @property
    def is_speaking(self) -> bool:
        return self._speaking and self._idx < len(self._track)


# ─────────────────────────────────────────────────────────────────────────────
#  TIMELINE SEQUENCER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TimelineEntry:
    t:     float
    event: str
    kwargs: dict = field(default_factory=dict)


class TimelineSequencer:
    """
    Orchestrates emotion transitions, lipsync, and behavior switches from a
    timed script.

    Usage
    ─────
        seq = TimelineSequencer(image, fps=12)
        seq.load_script([
            {"t": 0.0, "event": "idle"},
            {"t": 1.5, "event": "user_message"},
        ])
        frames = seq.render(duration=4.0)
    """

    def __init__(
        self,
        image:         Image.Image,
        emotion:       str = "neutral",
        fps:           int = 12,
        head_bob_scale: float = 1.0,
    ) -> None:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        self.image    = image
        self.fps      = fps
        self.geo      = FaceGeometry(image.width, image.height)

        self.emotion_blender = EmotionBlender(emotion)
        self.lipsync         = LipsyncScheduler(fps=fps)
        self.reaction_engine = (
            ReactionEngine(emotion) if _HAS_REACTION else None
        )
        self.head_bob_scale  = head_bob_scale
        self._script: list[_TimelineEntry] = []
        self._next_entry_idx: int = 0

        # Pre-generated animation curves (refilled as needed)
        self._blink_l:  list[float] = []
        self._blink_r:  list[float] = []
        self._pupil_xy: list[tuple] = []
        self._bob_xy:   list[tuple] = []
        self._curve_i:  int = 0
        self._refill_curves()

    # ── Script loading ─────────────────────────────────────────────────────

    def load_script(self, script: list[dict]) -> None:
        """
        Load a timed event script.

        Each entry is a dict with at least:
          {"t": <seconds>, "event": "<event_name>"}
        Optional extra keys are forwarded to the reaction engine.
        """
        self._script = sorted(
            [
                _TimelineEntry(
                    t      = float(e["t"]),
                    event  = str(e["event"]),
                    kwargs = {k: v for k, v in e.items() if k not in ("t", "event")},
                )
                for e in script
            ],
            key=lambda e: e.t,
        )
        self._next_entry_idx = 0

    # ── Render ─────────────────────────────────────────────────────────────

    def render(
        self,
        duration: float = 4.0,
        verbose:  bool  = False,
    ) -> list[Image.Image]:
        """
        Render the animation for `duration` seconds and return frames.
        """
        dt           = 1.0 / self.fps
        total_frames = max(1, round(duration * self.fps))
        frames       = []

        for i in range(total_frames):
            t_now = i * dt
            self._dispatch_events(t_now)

            frame = self._render_frame(dt)
            frames.append(frame)

            if verbose and (i % self.fps == 0):
                print(
                    f"  t={t_now:.2f}s  emotion={self.emotion_blender.emotion_name}"
                    f"  speaking={self.lipsync.is_speaking}"
                )

        return frames

    # ── Internal: event dispatch ───────────────────────────────────────────

    def _dispatch_events(self, t_now: float) -> None:
        """Fire any script entries whose timestamp ≤ t_now."""
        while self._next_entry_idx < len(self._script):
            entry = self._script[self._next_entry_idx]
            if entry.t > t_now:
                break
            self._handle_event(entry.event, **entry.kwargs)
            self._next_entry_idx += 1

    def _handle_event(self, event: str, **kwargs) -> None:
        """
        Map an event name to emotion + behavior + optional lipsync.

        Simple built-in mappings are provided for the most common events.
        If a reaction engine is available, it overrides these.
        """
        if self.reaction_engine is not None:
            self.reaction_engine.trigger(event)
            # Sync our blender to whatever the reaction engine set
            em = self.reaction_engine.current_emotion
            self.emotion_blender.set_emotion(em)
            # Start speech if event implies talking
            if self.reaction_engine.is_speaking:
                self.lipsync.start_speech(duration_frames=int(self.fps * 2.5))
        else:
            # Fallback simple mapping
            _EVENT_EMOTION_MAP = {
                "idle":         "neutral",
                "user_message": "happy",
                "loud_noise":   "surprised",
                "attention":    "curious",
                "praise":       "happy",
                "warning":      "angry",
                "random_thought": random.choice(["curious", "neutral", "happy"]),
            }
            em = _EVENT_EMOTION_MAP.get(event, "neutral")
            self.emotion_blender.set_emotion(em)
            if event in ("user_message", "praise", "random_thought"):
                self.lipsync.start_speech(duration_frames=int(self.fps * 2.5))

    # ── Internal: per-frame render ─────────────────────────────────────────

    def _render_frame(self, dt: float) -> Image.Image:
        """Advance all subsystems by dt and produce one rendered frame."""
        # 1. Update emotion blender
        self.emotion_blender.update(dt)

        # 2. Update reaction engine
        if self.reaction_engine is not None:
            self.reaction_engine.update()
            self.emotion_blender.set_emotion(self.reaction_engine.current_emotion)
            blink_rate = 0.03 * (1.0 + self.emotion_blender.eye_open * 0.3)
            behavior   = self.reaction_engine.current_behavior
            b_params   = get_behavior_params(behavior)
        else:
            blink_rate = 0.03 * (1.0 + self.emotion_blender.eye_open * 0.3)
            behavior   = "idle"
            b_params   = {"head_bob_scale": 1.0, "eye_jitter_scale": 0.4, "blink_emphasis": 1.0}

        # 3. Lipsync
        mouth_state = self.lipsync.tick()

        # 4. Animation curves
        if self._curve_i >= len(self._blink_l) - 1:
            self._refill_curves(blink_rate=blink_rate)
        bl   = self._blink_l[self._curve_i]
        br   = self._blink_r[self._curve_i]
        px, py = self._pupil_xy[self._curve_i]
        bob_dy, _ = self._bob_xy[self._curve_i]
        self._curve_i += 1

        # 5. Apply look_bias from emotion
        lbx, lby = self.emotion_blender.look_bias
        pdx = px + lbx * 0.35 + random.gauss(0, 0.10 * b_params.get("eye_jitter_scale", 0.4))
        pdy = py + lby * 0.35 + random.gauss(0, 0.10 * b_params.get("eye_jitter_scale", 0.4))

        # 6. Build CharacterState
        cstate = CharacterState(
            eye_open          = self.emotion_blender.eye_open,
            brow_angle        = self.emotion_blender.brow_angle,
            brow_raise        = self.emotion_blender.brow_raise,
            blink_rate        = blink_rate,
            look_bias         = (lbx, lby),
            mouth             = mouth_state,
            head_bob_scale    = b_params.get("head_bob_scale", 1.0) * self.head_bob_scale,
            eye_jitter_scale  = b_params.get("eye_jitter_scale", 0.4),
            emotion_name      = self.emotion_blender.emotion_name,
            behavior          = behavior,
        )

        # 7. Render
        return render_single_frame(
            base_image = self.image,
            state      = cstate,
            geo        = self.geo,
            blink_l    = bl,
            blink_r    = br,
            pupil_dx   = pdx,
            pupil_dy   = pdy,
            bob_dy     = bob_dy * b_params.get("head_bob_scale", 1.0) * self.head_bob_scale,
        )

    def _refill_curves(self, blink_rate: float = 0.03) -> None:
        n = 120
        rates         = [blink_rate] * n
        self._blink_l = _generate_reactive_blink_seq(rates, frame_offset=0)
        self._blink_r = _generate_reactive_blink_seq(rates, frame_offset=random.randint(1, 4))
        self._pupil_xy = generate_pupil_track(n)
        self._bob_xy   = generate_head_bob(n)
        self._curve_i  = 0


# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL animate() FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def animate(
    image:    Image.Image,
    duration: float = 4.0,
    script:   list[dict] | None = None,
    emotion:  str = "neutral",
    fps:      int = 12,
    verbose:  bool = False,
) -> list[Image.Image]:
    """
    Render an animated sequence for a single duck.

    image     PIL Image (any mode; will be converted to RGBA)
    duration  Length of animation in seconds
    script    Optional timed-event script.  If None, the duck idles.
    emotion   Starting emotion (used when no script is provided)
    fps       Frames per second
    verbose   Print progress

    Returns a list of RGBA frames.
    """
    if script is None:
        # Simple idle animation with no events
        script = [{"t": 0.0, "event": "idle"}]

    seq = TimelineSequencer(image, emotion=emotion, fps=fps)
    seq.load_script(script)
    return seq.render(duration=duration, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
#  GIF EXPORT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_frames_as_gif(
    frames:   list[Image.Image],
    out_path: str,
    fps:      int = 12,
) -> None:
    """Save a list of RGBA frames as an animated GIF."""
    rgb_frames = [f.convert("RGB") for f in frames]
    master     = rgb_frames[0].quantize(colors=256)
    pal_frames = [master] + [f.quantize(palette=master, dither=0) for f in rgb_frames[1:]]
    ms_per_frame = int(1000 / fps)
    pal_frames[0].save(
        out_path,
        format        = "GIF",
        save_all      = True,
        append_images = pal_frames[1:],
        loop          = 0,
        duration      = ms_per_frame,
        optimize      = True,
    )
    print(f"[timeline] Saved {len(frames)}-frame GIF → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser(
        description="Animate a duck image with a scripted timeline"
    )
    parser.add_argument("input",              help="Input image path")
    parser.add_argument("output",             help="Output GIF path")
    parser.add_argument("emotion", nargs="?", default="neutral",
                        help="Starting emotion (default: neutral)")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Duration in seconds (default: 4.0)")
    parser.add_argument("--fps",      type=int,   default=12,
                        help="Frames per second (default: 12)")
    parser.add_argument("--script",   default=None,
                        help="Path to JSON script file")
    args = parser.parse_args()

    img = Image.open(args.input)

    script = None
    if args.script:
        import json
        with open(args.script) as f:
            script = json.load(f)
    else:
        # Default demo script
        script = [
            {"t": 0.0,  "event": "idle"},
            {"t": 1.2,  "event": "user_message"},
            {"t": 2.8,  "event": "loud_noise"},
            {"t": 3.5,  "event": "idle"},
        ]

    print(f"Animating {args.input!r} → {args.output!r}  "
          f"({args.duration}s, emotion={args.emotion!r}) …")
    frames = animate(img, duration=args.duration, script=script,
                     emotion=args.emotion, fps=args.fps, verbose=True)
    save_frames_as_gif(frames, args.output, fps=args.fps)
