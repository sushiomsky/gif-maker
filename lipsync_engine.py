#!/usr/bin/env python3
"""lipsync_engine.py — Procedural fake lipsync for the Duckling character system.

Generates per-frame MouthState sequences using a first-order Markov chain over
discrete phoneme shapes.  No audio analysis is required; the output looks like
plausible talking animation.

Pipeline inside generate_track():
  1. Markov phoneme sequence  — shapes chosen stochastically at *rate* shapes/sec
  2. Frame expansion          — each shape block is mapped to N frames
  3. Crossfade blend          — 55 ms linear blend window at every transition
  4. Gaussian noise           — small per-frame jitter for liveliness
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Phoneme shape enumeration
# ---------------------------------------------------------------------------

class PhonemeShape(IntEnum):
    """Discrete mouth shapes used for procedural lipsync.

    Each value corresponds to a family of sounds (phonemes) that produce a
    visually similar mouth configuration on a 2-D character.
    """

    CLOSED = 0  # silence, bilabials (m, b, p)
    SMALL  = 1  # short vowels, fricatives (e, i, s, f)
    MID    = 2  # mid-open vowels (eh, uh)
    WIDE   = 3  # fully open vowels (ah, aa)
    ROUND  = 4  # rounded vowels (oh, oo, w)


# ---------------------------------------------------------------------------
# Per-shape mouth parameters
# ---------------------------------------------------------------------------

PHONEME_PARAMS: dict[int, dict[str, float]] = {
    PhonemeShape.CLOSED: {"mouth_open": 0.00, "width_scale": 1.00, "corner_pull": 0.00},
    PhonemeShape.SMALL:  {"mouth_open": 0.16, "width_scale": 0.72, "corner_pull": 0.05},
    PhonemeShape.MID:    {"mouth_open": 0.42, "width_scale": 0.88, "corner_pull": 0.12},
    PhonemeShape.WIDE:   {"mouth_open": 0.80, "width_scale": 1.12, "corner_pull": 0.22},
    PhonemeShape.ROUND:  {"mouth_open": 0.54, "width_scale": 0.62, "corner_pull": 0.00},
}


# ---------------------------------------------------------------------------
# Markov transition weights  (each row sums to 1.0)
# ---------------------------------------------------------------------------

_TRANSITIONS: dict[int, dict[int, float]] = {
    PhonemeShape.CLOSED: {
        PhonemeShape.SMALL: 0.40,
        PhonemeShape.MID:   0.30,
        PhonemeShape.WIDE:  0.15,
        PhonemeShape.ROUND: 0.15,
    },
    PhonemeShape.SMALL: {
        PhonemeShape.CLOSED: 0.35,
        PhonemeShape.MID:    0.35,
        PhonemeShape.WIDE:   0.15,
        PhonemeShape.ROUND:  0.15,
    },
    PhonemeShape.MID: {
        PhonemeShape.CLOSED: 0.25,
        PhonemeShape.SMALL:  0.25,
        PhonemeShape.WIDE:   0.30,
        PhonemeShape.ROUND:  0.20,
    },
    PhonemeShape.WIDE: {
        PhonemeShape.CLOSED: 0.20,
        PhonemeShape.SMALL:  0.20,
        PhonemeShape.MID:    0.35,
        PhonemeShape.ROUND:  0.25,
    },
    PhonemeShape.ROUND: {
        PhonemeShape.CLOSED: 0.30,
        PhonemeShape.SMALL:  0.25,
        PhonemeShape.MID:    0.30,
        PhonemeShape.WIDE:   0.15,
    },
}


# ---------------------------------------------------------------------------
# MouthState dataclass
# ---------------------------------------------------------------------------

@dataclass
class MouthState:
    """Snapshot of the duck's mouth configuration for a single animation frame.

    All float fields are normalised to meaningful ranges:
      - ``mouth_open``  : 0.0 (fully closed) → 1.0 (fully open)
      - ``width_scale`` : multiplier applied to the base mouth width
      - ``corner_pull`` : 0.0 (neutral) → 1.0 (maximum smile / grimace)
      - ``phoneme``     : the :class:`PhonemeShape` that drove this frame

    Attributes:
        mouth_open:  Vertical opening, 0.0–1.0.
        width_scale: Horizontal width multiplier.
        corner_pull: Smile/grimace intensity, 0.0–1.0.
        phoneme:     Source phoneme shape for this frame.
    """

    mouth_open:  float        = 0.0
    width_scale: float        = 1.0
    corner_pull: float        = 0.0
    phoneme:     PhonemeShape = PhonemeShape.CLOSED

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_phoneme(cls, shape: PhonemeShape) -> "MouthState":
        """Build a MouthState directly from a PhonemeShape constant.

        Args:
            shape: The phoneme shape to convert.

        Returns:
            A MouthState populated from :data:`PHONEME_PARAMS`.
        """
        params = PHONEME_PARAMS[shape]
        return cls(
            mouth_open=params["mouth_open"],
            width_scale=params["width_scale"],
            corner_pull=params["corner_pull"],
            phoneme=shape,
        )

    @classmethod
    def closed(cls) -> "MouthState":
        """Return a fully-closed, neutral MouthState."""
        return cls.from_phoneme(PhonemeShape.CLOSED)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def blend_toward(self, other: "MouthState", t: float) -> "MouthState":
        """Linearly interpolate from *self* toward *other*.

        Args:
            other: Target MouthState to blend toward.
            t:     Blend factor, automatically clamped to [0, 1].
                   ``t=0`` returns a copy of *self*; ``t=1`` returns *other*.

        Returns:
            A new MouthState with linearly interpolated field values.
        """
        t = max(0.0, min(1.0, t))
        return MouthState(
            mouth_open=self.mouth_open  + (other.mouth_open  - self.mouth_open)  * t,
            width_scale=self.width_scale + (other.width_scale - self.width_scale) * t,
            corner_pull=self.corner_pull + (other.corner_pull - self.corner_pull) * t,
            phoneme=(other.phoneme if t >= 0.5 else self.phoneme),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a plain dict suitable for JSON export.

        Returns:
            Dict with keys ``mouth_open``, ``width_scale``, ``corner_pull``,
            and ``phoneme`` (stored as its integer value).
        """
        return {
            "mouth_open":  self.mouth_open,
            "width_scale": self.width_scale,
            "corner_pull": self.corner_pull,
            "phoneme":     int(self.phoneme),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MouthState":
        """Deserialise a MouthState previously created by :meth:`to_dict`.

        Missing keys fall back to sensible defaults so the method is
        forward-compatible with partial dicts.

        Args:
            d: Dict as produced by :meth:`to_dict`.

        Returns:
            A reconstructed MouthState.
        """
        return cls(
            mouth_open=float(d.get("mouth_open",  0.0)),
            width_scale=float(d.get("width_scale", 1.0)),
            corner_pull=float(d.get("corner_pull", 0.0)),
            phoneme=PhonemeShape(int(d.get("phoneme", 0))),
        )


# ---------------------------------------------------------------------------
# Helper: weighted random choice
# ---------------------------------------------------------------------------

def _weighted_choice(weights: dict) -> PhonemeShape:
    """Pick a PhonemeShape at random according to *weights*.

    Args:
        weights: Mapping of ``{PhonemeShape: float}``.  Values are treated as
                 relative weights and need not sum to 1.0.

    Returns:
        The randomly selected :class:`PhonemeShape`.
    """
    items = list(weights.items())
    total = sum(w for _, w in items)
    threshold = random.random() * total
    cumulative = 0.0
    for shape, weight in items:
        cumulative += weight
        if threshold <= cumulative:
            return shape
    return items[-1][0]  # numerical-precision fallback


# ---------------------------------------------------------------------------
# LipsyncEngine
# ---------------------------------------------------------------------------

class LipsyncEngine:
    """Procedural fake-lipsync generator for the Duckling character.

    Produces sequences of :class:`MouthState` objects driven by a first-order
    Markov chain over :class:`PhonemeShape` values.  No audio analysis is
    performed; the output provides a plausible-looking talking animation.

    Args:
        fps: Target animation frame rate used to convert seconds to frame counts.
    """

    _STYLE_RATES: dict[str, float] = {
        "slow":   5.0,
        "normal": 7.5,
        "fast":  11.0,
        "mumble": 4.0,
    }

    def __init__(self, fps: int = 12) -> None:
        self.fps: int = fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_track(
        self,
        duration: float,
        rate: float | None = None,
        style: str = "normal",
        end_closed: bool = True,
    ) -> list[MouthState]:
        """Generate a lipsync track as a list of per-frame MouthState objects.

        Four-step pipeline
        ------------------
        1. **Markov sequence**   — random phoneme shapes chosen at *rate* / sec.
        2. **Frame expansion**   — each shape is assigned to a contiguous block
                                   of frames proportional to ``fps / rate``.
        3. **Crossfade blend**   — at every phoneme transition a 55 ms linear
                                   blend window smooths the discontinuity.
        4. **Gaussian noise**    — small per-frame jitter is added to
                                   ``mouth_open``, ``width_scale``, and
                                   ``corner_pull`` for liveliness.

        Args:
            duration:   Track length in seconds.
            rate:       Phoneme rate (shapes/second).  When ``None`` the rate
                        is taken from *style*.
            style:      ``"slow"`` (5.0/s) | ``"normal"`` (7.5/s) |
                        ``"fast"`` (11.0/s) | ``"mumble"`` (4.0/s).
                        The ``"mumble"`` style additionally caps any WIDE
                        shape to MID to keep movement subtle.
            end_closed: When ``True`` the final phoneme is forced to CLOSED
                        so the mouth visually returns to rest.

        Returns:
            Exactly ``int(duration * fps)`` :class:`MouthState` objects.
        """
        if rate is None:
            rate = self._STYLE_RATES.get(style, 7.5)

        n_frames = int(duration * self.fps)
        if n_frames == 0:
            return []

        # ── Step 1: Markov phoneme sequence ────────────────────────────────
        n_phonemes = max(2, int(duration * rate))
        n_phonemes = min(n_phonemes, n_frames)  # never more phonemes than frames

        shapes: list[PhonemeShape] = [PhonemeShape.CLOSED]
        for _ in range(n_phonemes - 1):
            next_shape = _weighted_choice(_TRANSITIONS[shapes[-1]])
            if style == "mumble" and next_shape == PhonemeShape.WIDE:
                next_shape = PhonemeShape.MID
            shapes.append(next_shape)

        if end_closed:
            shapes.append(PhonemeShape.CLOSED)

        # ── Step 2: Expand phoneme sequence to per-frame shapes ─────────────
        n_shapes = len(shapes)
        frame_shapes: list[PhonemeShape] = []
        for idx, shape in enumerate(shapes):
            start = int(idx       * n_frames / n_shapes)
            end   = int((idx + 1) * n_frames / n_shapes)
            frame_shapes.extend([shape] * (end - start))

        # Guarantee exact length (int() rounding can produce ±1 discrepancy)
        while len(frame_shapes) < n_frames:
            frame_shapes.append(shapes[-1])
        frame_shapes = frame_shapes[:n_frames]

        # ── Step 3: Crossfade blend at transitions (55 ms window) ───────────
        blend_half = max(1, int(round(0.0275 * self.fps)))
        blended: list[MouthState] = [MouthState.from_phoneme(s) for s in frame_shapes]

        for i in range(1, n_frames):
            if frame_shapes[i] != frame_shapes[i - 1]:
                lo   = max(0, i - blend_half)
                hi   = min(n_frames, i + blend_half)
                span = hi - lo
                prev = MouthState.from_phoneme(frame_shapes[i - 1])
                nxt  = MouthState.from_phoneme(frame_shapes[i])
                for j in range(lo, hi):
                    t = (j - lo) / span if span > 0 else 0.5
                    blended[j] = prev.blend_toward(nxt, t)

        # ── Step 4: Gaussian noise ───────────────────────────────────────────
        result: list[MouthState] = []
        for state in blended:
            result.append(MouthState(
                mouth_open=max(0.0, min(1.0, state.mouth_open  + random.gauss(0.0, 0.015))),
                width_scale=max(0.3, min(1.5, state.width_scale + random.gauss(0.0, 0.010))),
                corner_pull=max(0.0, min(1.0, state.corner_pull + random.gauss(0.0, 0.005))),
                phoneme=state.phoneme,
            ))

        return result

    def generate_idle_mouth(self, n_frames: int) -> list[MouthState]:
        """Generate a passive idle mouth track for when the duck is not speaking.

        Each frame is predominantly CLOSED with:

        * A **2 % chance** of a tiny random opening (0.0 – 0.07), simulating
          micro-expressions or micro-breaths.
        * Small Gaussian noise on all channels to simulate subtle muscle
          movement even when the mouth is fully at rest.

        Args:
            n_frames: Number of frames to generate.

        Returns:
            List of *n_frames* :class:`MouthState` objects.
        """
        result: list[MouthState] = []
        for _ in range(n_frames):
            base_open = random.uniform(0.0, 0.07) if random.random() < 0.02 else 0.0
            result.append(MouthState(
                mouth_open=max(0.0,       base_open + random.gauss(0.0, 0.004)),
                width_scale=max(0.3, min(1.5, 1.0   + random.gauss(0.0, 0.007))),
                corner_pull=max(0.0, min(1.0, 0.0   + random.gauss(0.0, 0.003))),
                phoneme=PhonemeShape.CLOSED,
            ))
        return result

    def idle_chatter_track(self, n_frames: int) -> list[MouthState]:
        """Generate a low-energy chatter track, as if the duck is muttering.

        Internally calls :meth:`generate_track` with ``style="mumble"`` and
        ``rate=9.0`` phonemes/sec (fast but capped), then dampens the result so
        mouth movement is barely perceptible — appropriate for background chatter.

        Dampening rules applied to each frame:

        * ``mouth_open``  is multiplied by **0.42** (amplitude reduction).
        * ``width_scale`` is blended 30 % toward **0.85** (toward neutral width).

        Args:
            n_frames: Number of frames to generate.

        Returns:
            List of exactly *n_frames* dampened :class:`MouthState` objects.
        """
        duration = n_frames / self.fps
        track = self.generate_track(duration, rate=9.0, style="mumble")

        result: list[MouthState] = []
        for state in track:
            result.append(MouthState(
                mouth_open=max(0.0, state.mouth_open * 0.42),
                width_scale=state.width_scale + (0.85 - state.width_scale) * 0.30,
                corner_pull=state.corner_pull,
                phoneme=state.phoneme,
            ))

        # Ensure exact length (generate_track already guarantees it, but be safe)
        result = result[:n_frames]
        while len(result) < n_frames:
            result.append(MouthState.closed())
        return result


# ---------------------------------------------------------------------------
# Demo / quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = LipsyncEngine(fps=12)
    track  = engine.generate_track(2.0)  # 2.0 s × 12 fps = 24 frames

    BAR_WIDTH = 26

    print("LipsyncEngine — 24-frame mouth_open preview")
    print(f"{'Fr':>3}  {'Phoneme':>8}  {'Open':>5}  {'Wid':>5}  Chart")
    print("─" * 62)
    for i, state in enumerate(track):
        filled = int(state.mouth_open * BAR_WIDTH)
        bar    = "█" * filled + "░" * (BAR_WIDTH - filled)
        print(
            f"  {i:02d}  {state.phoneme.name:>8}  "
            f"{state.mouth_open:.3f}  {state.width_scale:.3f}  {bar}"
        )

    print()
    print(f"Total frames returned : {len(track)}  (expected 24)")

    # Quick round-trip serialisation check
    sample     = track[6]
    rehydrated = MouthState.from_dict(sample.to_dict())
    assert abs(rehydrated.mouth_open - sample.mouth_open) < 1e-9, "to_dict/from_dict round-trip failed"
    print("Serialisation round-trip : OK")
