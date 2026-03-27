#!/usr/bin/env python3
"""
emotion_engine.py — Emotion parameter system for the Duckling GIF Animation Engine
====================================================================================

Each emotion is a dict with these keys
────────────────────────────────────────
  eye_open   float  0.0–1.5   Vertical eye-opening multiplier (1.0 = natural resting)
  brow_angle float  -30–+30   Brow tilt in degrees.
                              Negative → \\ / (angry inward-downward slant)
                              Positive → / \\ (sad/surprised inward-upward slant)
  brow_raise float            Reference-pixel raise (+) or lower (-) relative to default
  mouth_open float  0.0–1.0   Jaw opening (reserved; drawn as a hint line below beak)
  look_bias  tuple  (x, y)    Pixel offset bias for gaze direction
  blink_rate float  0–1       Per-frame probability of starting a blink cycle
  description str             Human-readable label

The EMOTIONS dict is the single source of truth.
Custom JSON files can override or extend it via load_custom_emotions().
"""

import json
import os
import random
from copy import deepcopy
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN EMOTION TABLE
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS: dict[str, dict] = {
    "neutral": {
        "eye_open":    1.00,
        "brow_angle":  0,
        "brow_raise":  0,
        "mouth_open":  0.00,
        "look_bias":   (0, 0),
        "blink_rate":  0.030,
        "description": "Calm, balanced resting expression",
    },
    "happy": {
        "eye_open":    0.82,    # slight squint from a smile
        "brow_angle":  4,       # very gentle / \\ arch
        "brow_raise":  4,       # raised with delight
        "mouth_open":  0.45,
        "look_bias":   (0, -1), # slight upward gaze
        "blink_rate":  0.025,
        "description": "Cheerful, squinted eyes, raised brows",
    },
    "sad": {
        "eye_open":    0.72,    # droopy
        "brow_angle":  18,      # / \\ pattern — inner ends up
        "brow_raise":  -3,      # overall lower
        "mouth_open":  0.00,
        "look_bias":   (0, 3),  # downward gaze
        "blink_rate":  0.048,   # blinks more (teary feel)
        "description": "Droopy eyes, inner brows raised, downward gaze",
    },
    "angry": {
        "eye_open":    1.18,    # wide, intense stare
        "brow_angle":  -22,     # \\ / pattern — inner ends DOWN
        "brow_raise":  -6,      # pushed hard down toward eyes
        "mouth_open":  0.10,
        "look_bias":   (0, 1),  # slight downward glare
        "blink_rate":  0.012,   # barely blinks — sustained glare
        "description": "Wide intense stare, furrowed lowered brows",
    },
    "surprised": {
        "eye_open":    1.45,    # very wide
        "brow_angle":  8,       # slight arch
        "brow_raise":  12,      # shot up high
        "mouth_open":  0.75,
        "look_bias":   (0, -3), # upward wide-eyed
        "blink_rate":  0.018,
        "description": "Very wide eyes, high raised brows, open mouth",
    },
    "sleepy": {
        "eye_open":    0.40,    # heavy half-closed lids
        "brow_angle":  3,
        "brow_raise":  -4,      # drooping
        "mouth_open":  0.05,
        "look_bias":   (0, 5),  # glazed downward
        "blink_rate":  0.095,   # slow heavy blinks
        "description": "Half-closed heavy eyelids, drooping brows",
    },
    "excited": {
        "eye_open":    1.32,
        "brow_angle":  -6,
        "brow_raise":  8,
        "mouth_open":  0.62,
        "look_bias":   (0, -2),
        "blink_rate":  0.020,
        "description": "Very open eyes, raised brows, high energy",
    },
    "confused": {
        "eye_open":    1.05,
        "brow_angle":  -5,      # slight uneven tilt
        "brow_raise":  2,
        "mouth_open":  0.15,
        "look_bias":   (3, 0),  # sideways glance
        "blink_rate":  0.035,
        "description": "Slight sideways glance, one brow raised",
    },
    "smug": {
        "eye_open":    0.78,    # half-lidded confidence
        "brow_angle":  -8,      # one side down feel
        "brow_raise":  1,
        "mouth_open":  0.20,
        "look_bias":   (2, -1), # side-eye
        "blink_rate":  0.022,
        "description": "Half-lidded eyes, side-eye, confident smirk",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_emotion(name: str, custom: Optional[dict] = None) -> dict:
    """Return emotion config by name.  Falls back to 'neutral' for unknowns."""
    source = custom if custom is not None else EMOTIONS
    return deepcopy(source.get(name.lower(), source.get("neutral", EMOTIONS["neutral"])))


def get_random_emotion(custom: Optional[dict] = None) -> tuple[str, dict]:
    """Return a random (name, config) pair."""
    source = custom if custom is not None else EMOTIONS
    name = random.choice(list(source.keys()))
    return name, deepcopy(source[name])


def list_emotions(custom: Optional[dict] = None) -> list[str]:
    """Return sorted list of available emotion names."""
    source = custom if custom is not None else EMOTIONS
    return sorted(source.keys())


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATION  (emotion transitions)
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_emotions(emo_a: dict, emo_b: dict, t: float) -> dict:
    """
    Linearly interpolate between two emotion configs.
    t=0.0 → pure emo_a,  t=1.0 → pure emo_b.
    """
    t = max(0.0, min(1.0, t))

    def lerp(a, b):
        return a + (b - a) * t

    ba = emo_a.get("look_bias", (0, 0))
    bb = emo_b.get("look_bias", (0, 0))

    return {
        "eye_open":    lerp(emo_a.get("eye_open",    1.0),  emo_b.get("eye_open",    1.0)),
        "brow_angle":  lerp(emo_a.get("brow_angle",  0),    emo_b.get("brow_angle",  0)),
        "brow_raise":  lerp(emo_a.get("brow_raise",  0),    emo_b.get("brow_raise",  0)),
        "mouth_open":  lerp(emo_a.get("mouth_open",  0.0),  emo_b.get("mouth_open",  0.0)),
        "look_bias":   (lerp(ba[0], bb[0]), lerp(ba[1], bb[1])),
        "blink_rate":  lerp(emo_a.get("blink_rate",  0.03), emo_b.get("blink_rate",  0.03)),
        "description": f"Transition {t:.2f}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EMOTION LOADER  (JSON override/extend)
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_emotions(json_path: str) -> dict:
    """
    Load a JSON file of emotion overrides/additions and merge with EMOTIONS.

    JSON format — a dict of emotion name → partial or full param dict.
    Missing keys fall back to built-in values (or 'neutral' defaults).

    Example JSON:
        {
          "party": { "eye_open": 1.3, "brow_raise": 9, "blink_rate": 0.015 },
          "angry": { "brow_angle": -28 }
        }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Custom emotions file not found: {json_path}")

    with open(json_path, "r") as fh:
        raw: dict = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {json_path}, got {type(raw).__name__}")

    merged = deepcopy(EMOTIONS)
    for name, params in raw.items():
        # Start from the existing entry (or neutral) so partial configs work
        base = deepcopy(merged.get(name, EMOTIONS["neutral"]))
        if "look_bias" in params and isinstance(params["look_bias"], list):
            params["look_bias"] = tuple(params["look_bias"])
        base.update(params)
        merged[name.lower()] = base

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-INSPECT CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'Emotion':<14} {'eye_open':>8} {'brow_angle':>10} {'brow_raise':>10} "
          f"{'blink_rate':>10}  description")
    print("─" * 72)
    for name, cfg in EMOTIONS.items():
        print(
            f"  {name:<12} {cfg['eye_open']:>8.2f} {cfg['brow_angle']:>10.0f} "
            f"{cfg['brow_raise']:>10.0f} {cfg['blink_rate']:>10.3f}  "
            f"{cfg['description']}"
        )
