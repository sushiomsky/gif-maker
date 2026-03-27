#!/usr/bin/env python3
"""
duck_animator.py — Core animation engine for the Duckling GIF Animation Engine
================================================================================

Architecture
────────────
  FaceGeometry          Proportional landmark positions derived from image size
  generate_blink_seq    Per-frame eye-openness curve with organic timing
  generate_pupil_track  Smooth sinusoidal + micro-saccade gaze positions
  generate_head_bob     Subtle frame-level vertical drift
  _draw_eye             Sclera → iris → pupil → catchlight overlay
  _draw_eyebrow         Emotion-driven brow line overlay
  generate_frames       Assembles all frames; supports emotion transitions
  save_gif              Exports frame list to an optimised animated GIF

All drawing is done on a COPY of the base image — the original is never modified.
No AI/ML models are used anywhere.  Dependencies: Pillow + Python stdlib only.

CLI usage
─────────
  python duck_animator.py input.png output.gif [emotion] [n_frames] [duration_ms]
"""

import math
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw

from emotion_engine import EMOTIONS, get_emotion, interpolate_emotions

# MouthState is defined in lipsync_engine.py (created in the reactive system
# module).  We import it here for type annotations in CharacterState.
# A minimal fallback is provided so duck_animator works standalone.
try:
    from lipsync_engine import MouthState
except ImportError:
    @dataclass
    class MouthState:   # type: ignore[no-redef]
        """Minimal fallback when lipsync_engine is not installed."""
        mouth_open:  float = 0.0
        width_scale: float = 1.0
        corner_pull: float = 0.0

        @classmethod
        def closed(cls) -> "MouthState":
            return cls()

        def to_dict(self) -> dict:
            return {"mouth_open": self.mouth_open,
                    "width_scale": self.width_scale,
                    "corner_pull": self.corner_pull}

        @classmethod
        def from_dict(cls, d: dict) -> "MouthState":
            return cls(**{k: v for k, v in d.items()
                          if k in ("mouth_open", "width_scale", "corner_pull")})


# ─────────────────────────────────────────────────────────────────────────────
#  FACE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

class FaceGeometry:
    """
    Estimates proportional facial landmark positions from image dimensions.

    Design assumption: the duck's head/face occupies roughly the upper-center
    60–80 % of a portrait-style image.  All values are computed from a single
    scale factor so the overlay self-adapts to any image resolution.

    Reference size: 256 px → scale = 1.0
    """

    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height

        # One global scale drives every measurement
        self.scale = min(width, height) / 256.0

        # ── Face centre ──────────────────────────────────────────────────────
        # Placed slightly above the geometric centre (typical duck portraits).
        self.cx = width  / 2.0
        self.cy = height * 0.44

        # ── Eye placement ────────────────────────────────────────────────────
        self.eye_sep   = 44 * self.scale   # half-distance between eye centres
        self.eye_y_off = -6 * self.scale   # eyes are above the face-centre

        self.left_eye_x  = self.cx - self.eye_sep
        self.right_eye_x = self.cx + self.eye_sep
        self.eye_y       = self.cy + self.eye_y_off

        # ── Eye ellipse base dimensions ──────────────────────────────────────
        # Modified per-frame by the emotion eye_open value and the blink curve
        self.eye_rx      = 17.0 * self.scale   # horizontal radius
        self.eye_ry_base = 12.0 * self.scale   # vertical radius at eye_open=1

        # ── Iris / pupil ─────────────────────────────────────────────────────
        self.pupil_r = 6.5 * self.scale

        # ── Eyebrow ──────────────────────────────────────────────────────────
        self.brow_half_len      = 18.0 * self.scale
        self.brow_default_raise = 20.0 * self.scale  # px above eye centre
        self.brow_thickness     = max(2, int(3.2 * self.scale))

        # ── Beak / mouth ─────────────────────────────────────────────────────
        # Duck's beak sits below the eye line; it animates for lipsync.
        self.beak_cx      = self.cx
        self.beak_cy      = self.cy + 38.0 * self.scale  # below face centre
        self.beak_w       = 22.0 * self.scale             # half-width
        self.beak_upper_h = 9.0  * self.scale             # upper bill height
        self.beak_lower_h = 7.0  * self.scale             # lower bill height
        self.beak_max_gap = 18.0 * self.scale             # max open gap


# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATION CURVES
# ─────────────────────────────────────────────────────────────────────────────

# Blink openness: 1.0 = fully open, 0.0 = fully shut
# Shape is deliberately asymmetric — eyes snap shut faster than they open,
# matching real eyelid physiology.
_BLINK_CURVE = [1.0, 0.72, 0.30, 0.05, 0.00, 0.05, 0.28, 0.62, 0.88, 1.0]


def _ease_in_out(t: float) -> float:
    """Smooth cubic ease for organic interpolation (t in [0, 1])."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def generate_blink_seq(n_frames: int, blink_rate: float,
                       frame_offset: int = 0) -> list[float]:
    """
    Pre-compute per-frame eye-openness values.

    blink_rate    Probability of beginning a blink on any given open frame.
    frame_offset  Shifts the returned window to stagger left/right eye timing,
                  creating the natural asymmetry of real blinking.

    Returns a list of floats (0.0–1.0) of length n_frames.
    """
    total = n_frames + frame_offset
    seq   = [1.0] * total

    i = 0
    # Leave room for a full blink + rest before the window ends
    while i < total - len(_BLINK_CURVE) - 5:
        if random.random() < blink_rate:
            # Only blink if all slots are still open (avoid overlapping blinks)
            if all(seq[i + j] >= 0.99 for j in range(len(_BLINK_CURVE))):
                for j, v in enumerate(_BLINK_CURVE):
                    seq[i + j] = v
                # Random rest period between blinks (prevents mechanical rhythm)
                i += len(_BLINK_CURVE) + random.randint(10, 30)
                continue
        i += 1

    # Trim the pre-computed window to exactly n_frames, shifted by frame_offset
    return seq[frame_offset: frame_offset + n_frames]


def generate_pupil_track(n_frames: int,
                         look_bias: tuple[float, float] = (0.0, 0.0)
                         ) -> list[tuple[float, float]]:
    """
    Generate smooth, organic gaze-offset positions (pixel offsets from centre).

    Technique: layered sinusoids at different frequencies + rare micro-saccades
    + Gaussian sub-pixel noise, then lightly smoothed.  look_bias shifts the
    whole track in a given direction (emotion-specific gaze direction).
    """
    phase_x  = random.uniform(0.0, 2 * math.pi)
    phase_y  = random.uniform(0.0, 2 * math.pi)
    phase_x2 = random.uniform(0.0, 2 * math.pi)

    raw: list[tuple[float, float]] = []
    for i in range(n_frames):
        # Primary slow horizontal sweep
        x = math.sin(i * 0.070 + phase_x)  * 3.8
        # Slower vertical drift
        y = math.sin(i * 0.045 + phase_y)  * 2.0
        # Secondary higher-frequency flutter (adds life)
        x += math.sin(i * 0.190 + phase_x2) * 1.1

        # Micro-saccade: a rare, brief snap to a new gaze point
        if random.random() < 0.018:
            x += random.uniform(-2.8, 2.8)
            y += random.uniform(-1.4, 1.4)

        # Sub-pixel Gaussian noise — makes no two frames identical
        x += random.gauss(0.0, 0.20)
        y += random.gauss(0.0, 0.20)

        # Apply the emotion's directional look-bias
        x += look_bias[0]
        y += look_bias[1]

        raw.append((x, y))

    # 3-frame moving-average smoothing to remove jerky artefacts
    smoothed: list[tuple[float, float]] = []
    for i in range(n_frames):
        window = raw[max(0, i - 1): i + 2]
        sx = sum(p[0] for p in window) / len(window)
        sy = sum(p[1] for p in window) / len(window)
        smoothed.append((sx, sy))

    return smoothed


def generate_head_bob(n_frames: int) -> list[tuple[float, float]]:
    """
    Generate per-frame subtle head-bob offsets.

    Returns list of (dy_px, tilt_deg):
      dy_px    Vertical shift of the entire overlay (fraction of a pixel)
      tilt_deg Very slight rotational drift (unused by default, reserved)
    """
    phase = random.uniform(0.0, 2 * math.pi)
    bobs: list[tuple[float, float]] = []
    for i in range(n_frames):
        dy    = math.sin(i * 0.058 + phase)           * 0.85
        dy   += math.sin(i * 0.023 + phase * 1.3)     * 0.30
        dy   += random.gauss(0.0, 0.08)
        tilt  = math.sin(i * 0.041 + phase * 0.9)     * 0.25
        tilt += random.gauss(0.0, 0.03)
        bobs.append((dy, tilt))
    return bobs


# ─────────────────────────────────────────────────────────────────────────────
#  OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def _draw_eye(
    draw:       ImageDraw.ImageDraw,
    geo:        FaceGeometry,
    side:       str,           # 'left' or 'right'
    eye_open:   float,         # emotion modifier
    blink_val:  float,         # current-frame blink curve value
    pupil_dx:   float,         # pixel offset of iris from eye centre (x)
    pupil_dy:   float,         # pixel offset of iris from eye centre (y)
) -> None:
    """
    Draw one eye composed of four concentric shapes:
      1. Sclera (eye white) — outer ellipse
      2. Iris (amber/brown duck colour) — inner circle
      3. Pupil (near-black) — smallest circle
      4. Catchlight (white specular dot) — realism detail

    When nearly closed only a thin arc is drawn.
    """
    cx = geo.left_eye_x  if side == "left" else geo.right_eye_x
    cy = geo.eye_y

    rx        = geo.eye_rx
    ry_full   = geo.eye_ry_base * max(0.0, eye_open)
    actual_ry = ry_full * max(0.0, blink_val)

    outline_w = max(1, int(geo.scale * 1.6))

    # ── Fully (or nearly) closed ─────────────────────────────────────────────
    if actual_ry < 0.8:
        if actual_ry < 0.25:
            # Fully shut: draw a gentle arc suggesting the lower eyelid
            arc_box = [cx - rx, cy - 2, cx + rx, cy + 2]
            draw.arc(arc_box, start=5, end=175,
                     fill=(22, 15, 6), width=max(1, outline_w))
        return

    # ── Sclera ───────────────────────────────────────────────────────────────
    sclera_box = (cx - rx, cy - actual_ry, cx + rx, cy + actual_ry)
    draw.ellipse(sclera_box, fill=(255, 249, 222), outline=(22, 15, 6),
                 width=outline_w)

    # ── Iris ─────────────────────────────────────────────────────────────────
    # Iris radius: smaller than sclera, constrained by current vertical opening
    iris_r = min(rx * 0.62, actual_ry * 0.82)

    # Clamp gaze offset so iris stays inside the sclera boundary
    max_dx = (rx        - iris_r) * 0.88
    max_dy = (actual_ry - iris_r) * 0.88
    sdx = max(-max_dx, min(max_dx, pupil_dx))
    sdy = max(-max_dy, min(max_dy, pupil_dy))

    icx, icy = cx + sdx, cy + sdy

    # Warm amber-brown iris colour (duck-appropriate)
    draw.ellipse(
        (icx - iris_r, icy - iris_r, icx + iris_r, icy + iris_r),
        fill=(112, 65, 18),
    )

    # ── Pupil ────────────────────────────────────────────────────────────────
    pr = iris_r * 0.56
    draw.ellipse(
        (icx - pr, icy - pr, icx + pr, icy + pr),
        fill=(7, 4, 1),
    )

    # ── Catchlight (specular highlight) ──────────────────────────────────────
    hl_r = max(1.0, pr * 0.31)
    hl_x  = icx - pr * 0.28
    hl_y  = icy - pr * 0.38
    draw.ellipse(
        (hl_x - hl_r, hl_y - hl_r, hl_x + hl_r, hl_y + hl_r),
        fill=(255, 255, 255),
    )


def _draw_eyebrow(
    draw:        ImageDraw.ImageDraw,
    geo:         FaceGeometry,
    side:        str,     # 'left' or 'right'
    brow_angle:  float,   # negative=angry \\/, positive=sad/surprised /\\
    brow_raise:  float,   # reference-pixel raise (+) or lower (-)
    jitter:      float,   # per-frame random y jitter for organic feel
) -> None:
    """
    Draw one eyebrow as a thick dark stroke above an eye.

    Eyebrows are the PRIMARY emotional indicator in this system.
    The angle convention (defined in emotion_engine.py):
      negative → inner ends pushed DOWN → \\ / = angry
      positive → inner ends pushed UP  → / \\ = sad / surprised

    For the LEFT brow  : outer point is on the left,  inner toward the nose.
    For the RIGHT brow : inner point is on the left, outer toward the nose.
    This mirrors the face symmetrically around the centre.
    """
    cx = geo.left_eye_x if side == "left" else geo.right_eye_x
    cy = geo.eye_y

    # Vertical centre of the brow stroke
    brow_y = cy - geo.brow_default_raise - brow_raise * geo.scale + jitter

    half     = geo.brow_half_len
    # How many pixels the inner end deviates vertically from the outer end
    angle_dy = math.tan(math.radians(brow_angle)) * half

    if side == "left":
        # outer (left side) → inner (right/nose side)
        x1, y1 = cx - half, brow_y + angle_dy   # outer: + angle_dy → lower when negative brow_angle
        x2, y2 = cx + half, brow_y - angle_dy   # inner: - angle_dy → higher when negative brow_angle
    else:
        # inner (left/nose side) → outer (right side)
        x1, y1 = cx - half, brow_y - angle_dy
        x2, y2 = cx + half, brow_y + angle_dy

    # Render a thick stroke by drawing parallel thin lines
    t     = geo.brow_thickness
    color = (32, 19, 5)
    for o in range(-(t // 2), t // 2 + 1):
        draw.line([(x1, y1 + o), (x2, y2 + o)], fill=color, width=1)

    # Soft rounded caps: tiny ellipses at each endpoint polish the stroke
    cap_r = max(1.0, t / 2.2)
    for ex, ey in [(x1, y1), (x2, y2)]:
        draw.ellipse(
            (ex - cap_r, ey - cap_r, ex + cap_r, ey + cap_r),
            fill=color,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_frame_geo(geo: FaceGeometry, bob_dy: float = 0.0) -> FaceGeometry:
    """Return a per-frame FaceGeometry copy with vertical head-bob applied."""
    fg = FaceGeometry.__new__(FaceGeometry)
    fg.__dict__ = geo.__dict__.copy()
    fg.eye_y   = geo.eye_y   + bob_dy
    fg.beak_cy = geo.beak_cy + bob_dy  # beak bobs with the head
    return fg


# ─────────────────────────────────────────────────────────────────────────────
#  BEAK / MOUTH DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def _draw_beak(
    draw:        ImageDraw.ImageDraw,
    geo:         FaceGeometry,
    mouth_state: MouthState,
) -> None:
    """
    Draw the duck's animated beak overlay.

    Structure
    ─────────
    Upper bill  fixed orange ellipse above beak_cy
    Lower bill  drops proportionally to mouth_state.mouth_open
    Interior    dark gap revealed when the beak is open

    The beak is drawn BEFORE the eyes so eyes appear on top.
    """
    cx = geo.beak_cx
    cy = geo.beak_cy
    w  = geo.beak_w * max(0.5, mouth_state.width_scale)

    gap       = mouth_state.mouth_open * geo.beak_max_gap
    upper_h   = geo.beak_upper_h
    lower_h   = geo.beak_lower_h
    outline_w = max(1, int(geo.scale * 1.4))

    beak_col    = (255, 168, 18)
    outline_col = (200, 120, 10)
    interior    = (25, 10, 5)

    # ── Dark interior visible when beak is open ───────────────────────────
    if gap > 1.5:
        draw.ellipse(
            (cx - w * 0.72, cy - gap * 0.52, cx + w * 0.72, cy + gap * 0.52),
            fill=interior,
        )

    # ── Upper bill (stays fixed) ──────────────────────────────────────────
    draw.ellipse(
        (cx - w,
         cy - upper_h - gap * 0.5,
         cx + w,
         cy + upper_h * 0.3 - gap * 0.5),
        fill=beak_col, outline=outline_col, width=outline_w,
    )

    # ── Lower bill (drops with mouth_open) ────────────────────────────────
    draw.ellipse(
        (cx - w * 0.82,
         cy - lower_h * 0.4 + gap * 0.5,
         cx + w * 0.82,
         cy + lower_h + gap * 0.5),
        fill=beak_col, outline=outline_col, width=outline_w,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CHARACTER STATE  — per-frame data carrier
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CharacterState:
    """
    Complete per-frame animation state for one duck character.

    Produced by the reactive/timeline system and consumed by
    render_single_frame() / generate_frames_from_track().
    """
    # ── Blended emotion parameters ────────────────────────────────────────
    eye_open:          float = 1.0
    brow_angle:        float = 0.0
    brow_raise:        float = 0.0
    blink_rate:        float = 0.030
    look_bias:         tuple = field(default_factory=lambda: (0.0, 0.0))

    # ── Mouth / lipsync ───────────────────────────────────────────────────
    mouth: MouthState = field(default_factory=MouthState.closed)

    # ── Behaviour modifiers ───────────────────────────────────────────────
    head_bob_scale:   float = 1.0   # multiplier on head-bob amplitude
    eye_jitter_scale: float = 1.0   # multiplier on pupil micro-jitter

    # ── Debug / metadata ─────────────────────────────────────────────────
    emotion_name: str = "neutral"
    behavior:     str = "idle"


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-FRAME RENDERER  (used by the world simulation's renderer)
# ─────────────────────────────────────────────────────────────────────────────

def render_single_frame(
    base_image:  Image.Image,
    state:       CharacterState,
    geo:         FaceGeometry,
    blink_l:     float,
    blink_r:     float,
    pupil_dx:    float,
    pupil_dy:    float,
    bob_dy:      float = 0.0,
    draw_beak:   bool  = True,
) -> Image.Image:
    """
    Render ONE animated frame by applying all overlays to a copy of base_image.

    This is the low-level compositing primitive used by the world renderer.
    Callers maintain their own blink / pupil / bob curves; this function just
    draws.

    Parameters
    ──────────
    base_image   Source duck image (must be RGBA)
    state        CharacterState for this frame
    geo          FaceGeometry for this image (pre-computed, shared across frames)
    blink_l/r    Current blink openness for each eye (0.0–1.0)
    pupil_dx/dy  Pixel offsets for gaze direction
    bob_dy       Vertical head-bob pixel offset
    draw_beak    Whether to draw the beak / mouth overlay

    Returns
    ───────
    New RGBA Image with overlays applied.
    """
    fg   = _make_frame_geo(geo, bob_dy)
    frame = base_image.copy()
    draw  = ImageDraw.Draw(frame, "RGBA")

    # Draw beak first so it appears visually behind the eyes
    if draw_beak:
        _draw_beak(draw, fg, state.mouth)

    # Per-frame brow jitter for asymmetric organic feel
    bjl = random.gauss(0.0, 0.30)
    bjr = random.gauss(0.0, 0.30) * 0.82

    _draw_eyebrow(draw, fg, "left",  state.brow_angle, state.brow_raise, bjl)
    _draw_eyebrow(draw, fg, "right", state.brow_angle, state.brow_raise, bjr)

    _draw_eye(draw, fg, "left",  state.eye_open, blink_l, pupil_dx, pupil_dy)
    _draw_eye(draw, fg, "right", state.eye_open, blink_r, pupil_dx, pupil_dy)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK-BASED FRAME GENERATION  (reactive / world simulation API)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_reactive_blink_seq(
    blink_rates: list[float], frame_offset: int = 0
) -> list[float]:
    """
    Per-frame blink sequence where the blink probability varies with emotion.
    Higher blink_rate frames cause more frequent blinks (e.g. nervous / sleepy).
    """
    total = len(blink_rates) + frame_offset
    seq   = [1.0] * total

    i = 0
    while i < total - len(_BLINK_CURVE) - 5:
        if random.random() < blink_rates[min(i, len(blink_rates) - 1)]:
            if all(seq[i + j] >= 0.99 for j in range(len(_BLINK_CURVE))):
                for j, v in enumerate(_BLINK_CURVE):
                    seq[i + j] = v
                i += len(_BLINK_CURVE) + random.randint(8, 28)
                continue
        i += 1

    return seq[frame_offset: frame_offset + len(blink_rates)]


def generate_frames_from_track(
    base_image: Image.Image,
    track:      list[CharacterState],
    draw_beak:  bool = True,
) -> list[Image.Image]:
    """
    Render a complete animation from a pre-computed CharacterState track.

    Each element of `track` drives one output frame.  The blink sequences and
    pupil tracks are auto-generated from the per-frame blink_rate and look_bias
    values in the track, so the animation feels natural even when emotions
    change mid-animation.

    Parameters
    ──────────
    base_image   Source image (any mode; auto-converted to RGBA)
    track        List of CharacterState, one per frame
    draw_beak    Whether to draw the animated beak overlay

    Returns
    ───────
    list[PIL.Image.Image] — one RGBA frame per CharacterState
    """
    if not track:
        return []

    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")

    width, height = base_image.size
    geo           = FaceGeometry(width, height)
    n             = len(track)

    # Per-frame blink rates from the track (emotion can change blink frequency)
    blink_rates   = [s.blink_rate for s in track]
    left_blinks   = _generate_reactive_blink_seq(blink_rates, frame_offset=0)
    right_blinks  = _generate_reactive_blink_seq(
        blink_rates, frame_offset=random.randint(1, 3)
    )

    # Average look_bias for the initial pupil track (live bias is added per-frame)
    avg_bx = sum(s.look_bias[0] for s in track) / n
    avg_by = sum(s.look_bias[1] for s in track) / n
    pupil_base = generate_pupil_track(n, (avg_bx, avg_by))

    head_bobs = generate_head_bob(n)

    frames: list[Image.Image] = []

    for i, state in enumerate(track):
        bob_dy, _ = head_bobs[i]
        bob_dy   *= state.head_bob_scale

        # Gaze bias: add per-frame live look_bias on top of base pupil track
        px = pupil_base[i][0] + state.look_bias[0] * 0.45
        py = pupil_base[i][1] + state.look_bias[1] * 0.45

        js  = max(0.1, state.eye_jitter_scale)
        ldx = px + random.gauss(0.0, 0.14 * js)
        ldy = py + random.gauss(0.0, 0.14 * js) + bob_dy * 0.35
        rdx = px + random.gauss(0.0, 0.14 * js)
        rdy = py + random.gauss(0.0, 0.14 * js) + bob_dy * 0.35

        frames.append(render_single_frame(
            base_image, state, geo,
            blink_l=left_blinks[i], blink_r=right_blinks[i],
            pupil_dx=ldx, pupil_dy=ldy,
            bob_dy=bob_dy, draw_beak=draw_beak,
        ))

    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  FRAME GENERATION  (original emotion-name API — backwards compatible)
# ─────────────────────────────────────────────────────────────────────────────

def generate_frames(
    base_image:        Image.Image,
    emotion_name:      str                  = "neutral",
    n_frames:          int                  = 80,
    transition_to:     Optional[str]        = None,
    transition_frames: int                  = 20,
    custom_emotions:   Optional[dict]       = None,
) -> list[Image.Image]:
    """
    Generate the complete list of animation frames.

    Each frame is an independent RGBA PIL Image.  The base image is never
    mutated.  Drawing happens on a per-frame copy.

    Parameters
    ──────────
    base_image        Source duck image (any PIL mode; auto-converted to RGBA)
    emotion_name      Key into EMOTIONS (or custom_emotions)
    n_frames          Total frames to produce
    transition_to     Optional emotion to blend toward across the last
                      `transition_frames` frames
    transition_frames Number of end-frames used for the emotion blend
    custom_emotions   Optional dict returned by load_custom_emotions()

    Returns
    ───────
    list[PIL.Image.Image]  — all frames in RGBA mode
    """
    emotion_dict = custom_emotions or EMOTIONS

    # Graceful fallback for unknown emotion names
    if emotion_name not in emotion_dict:
        print(f"⚠️  Unknown emotion '{emotion_name}', falling back to 'neutral'.")
        emotion_name = "neutral"

    emo_a = emotion_dict[emotion_name]
    emo_b = emotion_dict.get(transition_to, emo_a) if transition_to else emo_a

    # ── Base image setup ─────────────────────────────────────────────────────
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")
    width, height = base_image.size
    geo = FaceGeometry(width, height)

    # ── Pre-compute animation curves (once, before the frame loop) ───────────
    blink_rate = emo_a.get("blink_rate", 0.030)

    # Right eye lags 1–3 frames behind left for natural asymmetry
    left_blinks  = generate_blink_seq(n_frames, blink_rate, frame_offset=0)
    right_blinks = generate_blink_seq(n_frames, blink_rate,
                                      frame_offset=random.randint(1, 3))

    pupil_track = generate_pupil_track(
        n_frames, emo_a.get("look_bias", (0.0, 0.0))
    )
    head_bobs = generate_head_bob(n_frames)

    # ── Transition boundary ──────────────────────────────────────────────────
    trans_start = n_frames - transition_frames if transition_to else n_frames

    frames: list[Image.Image] = []

    for i in range(n_frames):

        # ── Emotion interpolation ────────────────────────────────────────────
        if transition_to and i >= trans_start:
            t_raw = (i - trans_start) / max(1, transition_frames - 1)
            emo   = interpolate_emotions(emo_a, emo_b, _ease_in_out(t_raw))
        else:
            emo = emo_a

        eye_open   = emo["eye_open"]
        brow_angle = emo["brow_angle"]
        brow_raise = emo["brow_raise"]

        # ── Head bob: shift face-centre Y each frame ──────────────────────
        bob_dy, _ = head_bobs[i]

        # Build a lightweight per-frame geometry (only eye_y is shifted)
        fg           = FaceGeometry.__new__(FaceGeometry)
        fg.width     = geo.width;   fg.height    = geo.height
        fg.scale     = geo.scale;   fg.cx        = geo.cx
        fg.eye_sep   = geo.eye_sep; fg.eye_y_off = geo.eye_y_off
        fg.left_eye_x  = geo.left_eye_x
        fg.right_eye_x = geo.right_eye_x
        fg.eye_y       = geo.eye_y + bob_dy          # <── bob applied here
        fg.eye_rx      = geo.eye_rx
        fg.eye_ry_base = geo.eye_ry_base
        fg.pupil_r     = geo.pupil_r
        fg.brow_half_len      = geo.brow_half_len
        fg.brow_default_raise = geo.brow_default_raise
        fg.brow_thickness     = geo.brow_thickness

        # ── Per-frame micro-randomness ────────────────────────────────────
        # Eyebrow jitter: independent for each brow → slight asymmetry
        brow_jitter_l = random.gauss(0.0, 0.32)
        brow_jitter_r = random.gauss(0.0, 0.32) * 0.85   # subtly less

        # Pupil offset: independent per-eye noise → pupils never perfectly aligned
        px, py = pupil_track[i]
        l_dx = px + random.gauss(0.0, 0.14)
        l_dy = py + random.gauss(0.0, 0.14) + bob_dy * 0.35
        r_dx = px + random.gauss(0.0, 0.14)
        r_dy = py + random.gauss(0.0, 0.14) + bob_dy * 0.35

        # ── Draw overlays on a fresh copy of the base ────────────────────
        frame = base_image.copy()
        draw  = ImageDraw.Draw(frame, "RGBA")

        # Eyebrows first (visually behind the eyeball)
        _draw_eyebrow(draw, fg, "left",  brow_angle, brow_raise, brow_jitter_l)
        _draw_eyebrow(draw, fg, "right", brow_angle, brow_raise, brow_jitter_r)

        # Eyes on top
        _draw_eye(draw, fg, "left",  eye_open, left_blinks[i],  l_dx, l_dy)
        _draw_eye(draw, fg, "right", eye_open, right_blinks[i], r_dx, r_dy)

        frames.append(frame)

    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  GIF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_gif(
    frames:      list[Image.Image],
    output_path: str,
    duration:    int = 80,
    loop:        int = 0,
) -> None:
    """
    Save animation frames to an optimised animated GIF.

    Frames are composited onto a white background (GIF has no full alpha),
    then quantised to a 256-colour adaptive palette.  All frames share the
    first frame's palette to prevent colour-shifting between frames.

    Parameters
    ──────────
    frames      List of PIL.Image (any mode)
    output_path Destination .gif file path
    duration    Milliseconds per frame  (60–100 ms recommended)
    loop        0 = loop forever;  N = loop N times
    """
    if not frames:
        raise ValueError("save_gif() received an empty frame list.")

    # ── Composite RGBA onto white background ─────────────────────────────────
    rgb_frames: list[Image.Image] = []
    for frame in frames:
        if frame.mode == "RGBA":
            bg = Image.new("RGB", frame.size, (255, 255, 255))
            bg.paste(frame, mask=frame.split()[3])
            rgb_frames.append(bg)
        else:
            rgb_frames.append(frame.convert("RGB"))

    # ── Build a single shared palette from the first frame ───────────────────
    # Using a shared palette prevents ugly colour flickering between frames.
    master = rgb_frames[0].quantize(
        colors=256,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.FLOYDSTEINBERG,
    )
    pal_frames = [master] + [
        f.quantize(palette=master, dither=0) for f in rgb_frames[1:]
    ]

    # ── Write GIF ────────────────────────────────────────────────────────────
    pal_frames[0].save(
        output_path,
        save_all=True,
        append_images=pal_frames[1:],
        loop=loop,
        duration=duration,
        optimize=True,
        disposal=2,   # restore to background between frames (clean compositing)
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(
        f"✅ GIF saved → {output_path}  "
        f"[{len(frames)} frames × {duration} ms  |  {size_kb:.0f} KB]"
    )


# keep os import near usage
import os   # noqa: E402 (used in save_gif above)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    """
    Minimal CLI:
      python duck_animator.py <input> <output> [emotion] [n_frames] [duration_ms]
    """
    if len(sys.argv) < 3:
        print(
            "Usage: python duck_animator.py <input_image> <output_gif> "
            "[emotion] [n_frames] [duration_ms]\n"
            f"Emotions: {', '.join(sorted(EMOTIONS.keys()))}"
        )
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    if len(sys.argv) > 3:
        emotion = sys.argv[3]
    else:
        emotion = random.choice(list(EMOTIONS.keys()))
        print(f"🎲 No emotion specified — using random: {emotion}")

    n_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    duration = int(sys.argv[5]) if len(sys.argv) > 5 else 80

    print(f"🦆 Loading: {input_path}")
    img = Image.open(input_path)

    print(f"🎭 Emotion: {emotion}  |  Frames: {n_frames}  |  {duration} ms/frame")
    frames = generate_frames(img, emotion_name=emotion, n_frames=n_frames)
    save_gif(frames, output_path, duration=duration)


if __name__ == "__main__":
    _cli()
