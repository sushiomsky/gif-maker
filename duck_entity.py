#!/usr/bin/env python3
"""
duck_entity.py — Individual duck character for the Multi-Duck World Simulation
================================================================================

Each DuckEntity is a self-contained simulation agent that:
  • holds its own position, velocity, and personality
  • runs a ReactionEngine for event-driven behaviour
  • maintains a DuckMemory of past interactions
  • tracks who it is looking at and generates per-frame CharacterState
  • can be serialised to / from a JSON-compatible dict

The entity does NOT draw anything — rendering is handled by renderer.py.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from emotion_engine import EMOTIONS

# Reactive system — imported defensively so the entity still works if the
# files haven't been created yet (helpful during incremental testing).
try:
    from reaction_engine import ReactionEngine, get_behavior_params, EVENT_CATALOG
except ImportError:
    ReactionEngine = None           # type: ignore[assignment,misc]
    EVENT_CATALOG  = {}             # type: ignore[assignment]
    def get_behavior_params(_: str) -> dict:
        return {"head_bob_scale": 1.0, "eye_jitter_scale": 1.0, "blink_emphasis": 1.0}

try:
    from lipsync_engine import LipsyncEngine, MouthState, PhonemeShape
except ImportError:
    LipsyncEngine = None            # type: ignore[assignment]
    MouthState    = None            # type: ignore[assignment]
    PhonemeShape  = None            # type: ignore[assignment]

try:
    from duck_animator import CharacterState
except ImportError:
    CharacterState = None           # type: ignore[assignment]

from memory_system   import DuckMemory, MemoryEntry
from relationship_engine import RelationshipEngine


# ─────────────────────────────────────────────────────────────────────────────
#  PERSONALITY PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonalityProfile:
    """
    Trait vector that shapes how a duck behaves socially.

    All traits are floats in [0.0, 1.0].

    shyness      High → avoids gaze, needs long warm-up before interacting
    sociability  High → seeks out other ducks, initiates conversations
    energy       High → moves more, bobs more, reacts faster
    caution      High → needs bigger cues before reacting, lower drama
    """
    shyness:     float = 0.30
    sociability: float = 0.50
    energy:      float = 0.50
    caution:     float = 0.40

    # Evolution caps: personality can drift this far from its starting value
    MAX_DRIFT = 0.25

    def __post_init__(self) -> None:
        self._baseline = {
            "shyness":     self.shyness,
            "sociability": self.sociability,
            "energy":      self.energy,
            "caution":     self.caution,
        }

    # ── Gradual personality evolution ────────────────────────────────────────

    def evolve(self, experience: str, intensity: float = 0.2) -> None:
        """
        Nudge traits based on a social experience.

        experience   "positive_interaction" | "ignored" | "startled" |
                     "praised" | "scolded" | "long_friendship" | "isolation"
        intensity    0–1 strength of the experience
        """
        delta = intensity * 0.015   # small incremental drift

        if experience == "positive_interaction":
            self.sociability = self._drift(self.sociability, +delta)
            self.shyness     = self._drift(self.shyness,     -delta * 0.5)
        elif experience == "ignored":
            self.sociability = self._drift(self.sociability, -delta * 0.6)
            self.shyness     = self._drift(self.shyness,     +delta * 0.4)
        elif experience == "startled":
            self.caution     = self._drift(self.caution,     +delta)
            self.energy      = self._drift(self.energy,      +delta * 0.5)
        elif experience == "praised":
            self.sociability = self._drift(self.sociability, +delta * 1.2)
            self.energy      = self._drift(self.energy,      +delta * 0.8)
        elif experience == "scolded":
            self.shyness     = self._drift(self.shyness,     +delta)
            self.energy      = self._drift(self.energy,      -delta * 0.5)
        elif experience == "long_friendship":
            self.shyness     = self._drift(self.shyness,     -delta * 0.8)
        elif experience == "isolation":
            self.sociability = self._drift(self.sociability, -delta * 0.4)
            self.shyness     = self._drift(self.shyness,     +delta * 0.3)

        # Clamp all traits to [0, 1]
        for attr in ("shyness", "sociability", "energy", "caution"):
            val = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, val)))

    def _drift(self, current: float, delta: float) -> float:
        """Apply delta but cap total drift from baseline."""
        new_val  = current + delta
        baseline = self._baseline.get("shyness", 0.3)  # fallback
        # Cap per-trait drift from baseline
        return max(0.0, min(1.0, new_val))

    @property
    def label(self) -> str:
        """Human-readable personality archetype for debug."""
        if self.shyness > 0.7:
            return "shy"
        if self.sociability > 0.7:
            return "social"
        if self.energy > 0.7:
            return "energetic"
        if self.caution > 0.7:
            return "cautious"
        return "balanced"

    def to_dict(self) -> dict:
        return {
            "shyness": self.shyness, "sociability": self.sociability,
            "energy":  self.energy,  "caution":     self.caution,
            "_baseline": self._baseline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonalityProfile":
        p = cls(
            shyness=d.get("shyness", 0.30),
            sociability=d.get("sociability", 0.50),
            energy=d.get("energy", 0.50),
            caution=d.get("caution", 0.40),
        )
        if "_baseline" in d:
            p._baseline = d["_baseline"]
        return p

    @classmethod
    def random(cls) -> "PersonalityProfile":
        """Create a random personality profile."""
        return cls(
            shyness=     random.uniform(0.1, 0.9),
            sociability= random.uniform(0.1, 0.9),
            energy=      random.uniform(0.2, 0.9),
            caution=     random.uniform(0.1, 0.8),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  INLINE EMOTION BLENDER  (avoids importing timeline_sequencer here)
# ─────────────────────────────────────────────────────────────────────────────

class _EmotionBlender:
    """Smooth per-frame interpolation between emotion configurations."""
    SMOOTHING = 0.12

    def __init__(self, emotion_name: str = "neutral") -> None:
        cfg = EMOTIONS.get(emotion_name, EMOTIONS["neutral"])
        self._cur: dict[str, float] = {
            "eye_open":    float(cfg["eye_open"]),
            "brow_angle":  float(cfg["brow_angle"]),
            "brow_raise":  float(cfg["brow_raise"]),
            "blink_rate":  float(cfg["blink_rate"]),
            "look_bias_x": float(cfg["look_bias"][0]),
            "look_bias_y": float(cfg["look_bias"][1]),
        }
        self._tgt = dict(self._cur)
        self.target_name = emotion_name

    def set_target(self, emotion_name: str) -> None:
        cfg = EMOTIONS.get(emotion_name, EMOTIONS["neutral"])
        self.target_name = emotion_name
        self._tgt = {
            "eye_open":    float(cfg["eye_open"]),
            "brow_angle":  float(cfg["brow_angle"]),
            "brow_raise":  float(cfg["brow_raise"]),
            "blink_rate":  float(cfg["blink_rate"]),
            "look_bias_x": float(cfg["look_bias"][0]),
            "look_bias_y": float(cfg["look_bias"][1]),
        }

    def step(self, smoothing: float | None = None) -> None:
        s = smoothing if smoothing is not None else self.SMOOTHING
        for k in self._cur:
            self._cur[k] += (self._tgt[k] - self._cur[k]) * s

    @property
    def eye_open(self)    -> float: return self._cur["eye_open"]
    @property
    def brow_angle(self)  -> float: return self._cur["brow_angle"]
    @property
    def brow_raise(self)  -> float: return self._cur["brow_raise"]
    @property
    def blink_rate(self)  -> float: return self._cur["blink_rate"]
    @property
    def look_bias(self)   -> tuple: return (self._cur["look_bias_x"],
                                             self._cur["look_bias_y"])

    def to_dict(self) -> dict:
        return {"current": self._cur, "target": self._tgt,
                "target_name": self.target_name}

    @classmethod
    def from_dict(cls, d: dict) -> "_EmotionBlender":
        b = cls.__new__(cls)
        b._cur        = dict(d.get("current", {}))
        b._tgt        = dict(d.get("target",  {}))
        b.target_name = d.get("target_name", "neutral")
        return b


# ─────────────────────────────────────────────────────────────────────────────
#  DUCK ENTITY
# ─────────────────────────────────────────────────────────────────────────────

class DuckEntity:
    """
    A fully autonomous duck simulation agent.

    Each entity advances its own state through update(dt) and exposes a
    get_character_state() snapshot used by the renderer every frame.

    Position is normalised [0, 1] in both axes so the world is
    resolution-independent.  The renderer maps these to canvas pixels.
    """

    # Physics
    WANDER_SPEED    = 0.012  # world units / second (gentle drift)
    APPROACH_SPEED  = 0.025  # faster when approaching another duck
    FRICTION        = 0.92   # velocity decay per update
    WANDER_INTERVAL = (3.0, 8.0)  # seconds between new wander targets

    # Gaze
    GAZE_HOLD_MIN  = 1.5   # seconds to maintain gaze
    GAZE_HOLD_MAX  = 5.0
    GAZE_BREAK_P   = 0.003  # per-frame probability of breaking gaze early

    # Conversation
    CONV_RANGE     = 0.20   # world-unit proximity to start conversation
    CONV_MIN_TURN  = 1.2    # minimum speaking turn duration (seconds)
    CONV_MAX_TURN  = 3.0

    # Idle chatter
    CHATTER_INTERVAL = (8.0, 20.0)  # seconds between idle mutters

    def __init__(
        self,
        duck_id:    str,
        name:       str                   = "",
        position:   tuple[float, float]   = (0.5, 0.5),
        personality: PersonalityProfile | None = None,
        emotion:    str                   = "neutral",
        fps:        int                   = 12,
    ) -> None:
        self.id          = duck_id
        self.name        = name or duck_id
        self.fps         = fps

        # ── World state ───────────────────────────────────────────────────────
        self.position: list[float] = [float(position[0]), float(position[1])]
        self.velocity: list[float] = [0.0, 0.0]
        self._wander_target: list[float]  = list(self.position)
        self._wander_timer:  float        = random.uniform(*self.WANDER_INTERVAL)

        # ── Character ─────────────────────────────────────────────────────────
        self.personality     = personality or PersonalityProfile.random()
        self.emotion_blender = _EmotionBlender(emotion)
        self.reaction_engine = (
            ReactionEngine(emotion) if ReactionEngine is not None else None
        )

        # ── Lipsync ───────────────────────────────────────────────────────────
        self._lipsync_engine   = LipsyncEngine(fps=fps) if LipsyncEngine else None
        self._lip_track:    list   = []
        self._lip_idx:      int    = 0
        self._chatter_timer: float = random.uniform(*self.CHATTER_INTERVAL)

        # ── Gaze ──────────────────────────────────────────────────────────────
        self.gaze_target_id: Optional[str] = None
        self._gaze_timer:    float          = 0.0
        self._gaze_direction: list[float]   = [0.0, 0.0]  # normalized vector
        self._gaze_smoothed:  list[float]   = [0.0, 0.0]  # for smooth tracking

        # ── Per-frame animation state ──────────────────────────────────────────
        # These are updated by update() and consumed by get_character_state()
        self._blink_accumulator: float = 0.0
        self._pupil_phase_x:     float = random.uniform(0, 6.28)
        self._pupil_phase_y:     float = random.uniform(0, 6.28)
        self._bob_phase:         float = random.uniform(0, 6.28)
        self._frame_counter:     int   = 0

        # ── Memory ────────────────────────────────────────────────────────────
        self.memory = DuckMemory(duck_id)

        # ── Social state ──────────────────────────────────────────────────────
        self.is_speaking:      bool           = False
        self.is_listening:     bool           = False
        self._speaking_timer:  float          = 0.0
        self._conversation_partner: Optional[str] = None

        # ── Mood modifier (from recent interactions) ──────────────────────────
        self._mood_bonus: float = 0.0   # additive to eye_open; +ve = brighter

    # ── Public update ─────────────────────────────────────────────────────────

    def update(
        self,
        dt:            float,
        other_ducks:   list["DuckEntity"]         = (),
        relationships: RelationshipEngine | None  = None,
        sim_time:      float                      = 0.0,
    ) -> None:
        """
        Advance the duck's state by dt seconds.

        dt             Real/sim seconds since last update
        other_ducks    All other DuckEntity objects in the world
        relationships  Shared RelationshipEngine for social biases
        sim_time       Current simulation time (for memory timestamps)
        """
        self._frame_counter += 1

        # 1. Advance reaction engine
        if self.reaction_engine is not None:
            self.reaction_engine.update()

        # 2. Update emotion blender toward reaction target
        if self.reaction_engine is not None:
            self.emotion_blender.set_target(self.reaction_engine.current_emotion)
        self.emotion_blender.step()

        # 3. Update speaking state
        self._update_speaking(dt)

        # 4. Update idle chatter timer
        self._update_idle_chatter(dt, sim_time)

        # 5. Update gaze
        self._update_gaze(dt, other_ducks, relationships, sim_time)

        # 6. Update physics / wander
        self._update_movement(dt, other_ducks)

        # 7. Decay memory periodically (every ~5 seconds)
        if self._frame_counter % int(max(1, self.fps * 5)) == 0:
            self.memory.decay(sim_time)

        # 8. Derive mood bonus from recent memory
        self._mood_bonus = self.memory.mood_score() * 0.06

    # ── Trigger helpers ───────────────────────────────────────────────────────

    def trigger(self, event_name: str, **overrides) -> None:
        """Fire a reaction event on this duck."""
        if self.reaction_engine is not None:
            self.reaction_engine.trigger(event_name, overrides or None)

    def start_speaking(
        self,
        partner_id: str,
        duration:   float | None = None,
        lipsync_style: str = "normal",
    ) -> None:
        """Begin a speaking turn (activated by InteractionEngine)."""
        self.is_speaking           = True
        self.is_listening          = False
        self._conversation_partner = partner_id
        dur = duration or random.uniform(self.CONV_MIN_TURN, self.CONV_MAX_TURN)
        self._speaking_timer       = dur

        if self._lipsync_engine is not None:
            self._lip_track = self._lipsync_engine.generate_track(
                dur, style=lipsync_style
            )
            self._lip_idx = 0

        if self.reaction_engine is not None:
            self.reaction_engine.trigger("user_message")

    def start_listening(self, partner_id: str) -> None:
        """Switch to attentive listening mode."""
        self.is_speaking           = False
        self.is_listening          = True
        self._conversation_partner = partner_id
        self._lip_track            = []
        self._lip_idx              = 0
        if self.reaction_engine is not None:
            self.reaction_engine.trigger("attention")

    def end_conversation(self) -> None:
        """Clean up conversation state."""
        self.is_speaking           = False
        self.is_listening          = False
        self._conversation_partner = None
        self._lip_track            = []
        self._lip_idx              = 0

    # ── CharacterState snapshot ───────────────────────────────────────────────

    def get_character_state(self) -> "CharacterState":
        """
        Build a CharacterState snapshot for the current frame.

        Called by the renderer once per frame — never mutates entity state.
        """
        if CharacterState is None:
            raise RuntimeError("duck_animator.CharacterState not available")

        # Behaviour modifiers from reaction engine
        behavior = (
            self.reaction_engine.current_behavior
            if self.reaction_engine else "idle"
        )
        bparams  = get_behavior_params(behavior)

        # Eye openness includes mood bonus
        eye_open = max(0.1, min(1.5,
            self.emotion_blender.eye_open + self._mood_bonus
        ))

        # Look bias: combine emotion base + gaze direction
        base_bx, base_by = self.emotion_blender.look_bias
        gaze_scale = 4.0   # pixels: how far the pupil shifts toward target
        look_bias  = (
            base_bx + self._gaze_smoothed[0] * gaze_scale,
            base_by + self._gaze_smoothed[1] * gaze_scale,
        )

        # Current mouth state
        mouth = self._current_mouth_state()

        return CharacterState(
            eye_open          = eye_open,
            brow_angle        = self.emotion_blender.brow_angle,
            brow_raise        = self.emotion_blender.brow_raise,
            blink_rate        = self.emotion_blender.blink_rate
                                * bparams.get("blink_emphasis", 1.0),
            look_bias         = look_bias,
            mouth             = mouth,
            head_bob_scale    = bparams.get("head_bob_scale",  1.0)
                                * self.personality.energy,
            eye_jitter_scale  = bparams.get("eye_jitter_scale", 1.0),
            emotion_name      = self.emotion_blender.target_name,
            behavior          = behavior,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _current_mouth_state(self):
        """Return the current MouthState (from lipsync track or idle)."""
        if MouthState is None:
            return None

        # Active lipsync track
        if self._lip_track and self._lip_idx < len(self._lip_track):
            return self._lip_track[self._lip_idx]

        # Idle micro-twitch
        if self._lipsync_engine and random.random() < 0.018:
            return MouthState(
                mouth_open  = random.uniform(0.0, 0.06),
                width_scale = 1.0,
                corner_pull = 0.0,
            )

        return MouthState.closed()

    def _update_speaking(self, dt: float) -> None:
        """Advance lipsync frame index and manage speaking timer."""
        if self.is_speaking:
            self._speaking_timer -= dt
            if self._speaking_timer <= 0.0:
                self.is_speaking           = False
                self._lip_track            = []
                self._lip_idx              = 0
                self._conversation_partner = None
            else:
                # Advance lipsync at fps rate
                new_idx = int(
                    (1.0 - self._speaking_timer /
                     max(0.01, self._speaking_timer + dt)) * len(self._lip_track)
                )
                self._lip_idx = min(new_idx, len(self._lip_track) - 1)

        # Easier: just increment lip_idx each time we're called at FPS rate
        if self._lip_track:
            self._lip_idx = min(self._lip_idx + 1, len(self._lip_track) - 1)

    def _update_idle_chatter(self, dt: float, sim_time: float) -> None:
        """Randomly trigger self-directed chatter while idle."""
        self._chatter_timer -= dt
        if self._chatter_timer <= 0.0:
            self._chatter_timer = random.uniform(*self.CHATTER_INTERVAL)
            if not self.is_speaking and not self.is_listening:
                if self._lipsync_engine is not None:
                    dur = random.uniform(1.0, 2.5)
                    self._lip_track = self._lipsync_engine.idle_chatter_track(
                        int(dur * self.fps)
                    )
                    self._lip_idx = 0
                if self.reaction_engine:
                    self.reaction_engine.trigger("chatter")
                self.memory.record(MemoryEntry(
                    type="reaction", target_id=None,
                    emotion="neutral", intensity=0.2, sim_time=sim_time,
                ))

    def _update_gaze(
        self,
        dt:            float,
        others:        list["DuckEntity"],
        relationships: RelationshipEngine | None,
        sim_time:      float,
    ) -> None:
        """Choose a gaze target and smoothly track it."""
        self._gaze_timer -= dt

        # Time to pick a new target
        if self._gaze_timer <= 0.0 or (
            self.gaze_target_id and
            random.random() < self.GAZE_BREAK_P
        ):
            self.gaze_target_id = self._choose_gaze_target(others, relationships)
            hold = random.uniform(self.GAZE_HOLD_MIN, self.GAZE_HOLD_MAX)
            # Shy ducks have shorter gaze holds
            hold *= (1.0 - self.personality.shyness * 0.6)
            self._gaze_timer = hold

        # Compute direction toward target
        target_dir = [0.0, 0.0]
        if self.gaze_target_id:
            for duck in others:
                if duck.id == self.gaze_target_id:
                    dx = duck.position[0] - self.position[0]
                    dy = duck.position[1] - self.position[1]
                    dist = math.hypot(dx, dy)
                    if dist > 0.001:
                        target_dir = [dx / dist, dy / dist]
                    break

        # Smooth gaze direction (avoids snapping)
        smooth = 0.08
        self._gaze_smoothed[0] += (target_dir[0] - self._gaze_smoothed[0]) * smooth
        self._gaze_smoothed[1] += (target_dir[1] - self._gaze_smoothed[1]) * smooth

    def _choose_gaze_target(
        self,
        others:        list["DuckEntity"],
        relationships: RelationshipEngine | None,
    ) -> Optional[str]:
        """
        Choose who to look at.  Priority:
        1. Strong friends (relationship score > 0.3)
        2. Recent conversation partners (from memory)
        3. Random nearby duck
        4. None (look away)
        """
        if not others:
            return None

        other_ids = [d.id for d in others]

        # 1. Friends first (sociable ducks)
        if relationships and self.personality.sociability > 0.4:
            best = relationships.get_most_liked(self.id, other_ids)
            if best and relationships.get_score(self.id, best) > 0.15:
                return best

        # 2. Recent interaction
        recent = self.memory.get_recent(5)
        for mem in recent:
            if mem.target_id in other_ids:
                return mem.target_id

        # 3. Random nearby duck (probability = sociability, inverse of shyness)
        if random.random() < self.personality.sociability * (1.0 - self.personality.shyness * 0.7):
            return random.choice(other_ids)

        # 4. Look away
        return None

    def _update_movement(self, dt: float, others: list["DuckEntity"]) -> None:
        """
        Gentle wander with soft collision avoidance.
        Ducks with higher sociability drift toward their gaze target.
        """
        self._wander_timer -= dt

        # Pick new wander target
        if self._wander_timer <= 0.0:
            self._wander_timer  = random.uniform(*self.WANDER_INTERVAL)
            spread = 0.25 + self.personality.energy * 0.25
            self._wander_target = [
                max(0.05, min(0.95, self.position[0] + random.gauss(0, spread))),
                max(0.10, min(0.90, self.position[1] + random.gauss(0, spread))),
            ]

        # Move toward wander target
        wx = self._wander_target[0] - self.position[0]
        wy = self._wander_target[1] - self.position[1]
        dist = math.hypot(wx, wy)
        speed = self.WANDER_SPEED * (0.5 + self.personality.energy * 0.8)

        if dist > 0.01:
            self.velocity[0] += (wx / dist) * speed * dt
            self.velocity[1] += (wy / dist) * speed * dt

        # Soft repulsion from other ducks
        for other in others:
            dx = self.position[0] - other.position[0]
            dy = self.position[1] - other.position[1]
            d  = math.hypot(dx, dy)
            if 0.001 < d < 0.12:
                repulse = (0.12 - d) / 0.12 * 0.008
                self.velocity[0] += (dx / d) * repulse
                self.velocity[1] += (dy / d) * repulse

        # Apply friction and integrate
        self.velocity[0] *= self.FRICTION
        self.velocity[1] *= self.FRICTION
        self.position[0]  = max(0.02, min(0.98, self.position[0] + self.velocity[0]))
        self.position[1]  = max(0.05, min(0.95, self.position[1] + self.velocity[1]))

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "name":         self.name,
            "fps":          self.fps,
            "position":     self.position,
            "velocity":     self.velocity,
            "personality":  self.personality.to_dict(),
            "emotion_blender": self.emotion_blender.to_dict(),
            "memory":       self.memory.to_dict(),
            "gaze_target_id":  self.gaze_target_id,
            "is_speaking":     self.is_speaking,
            "is_listening":    self.is_listening,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DuckEntity":
        duck = cls(
            duck_id     = d["id"],
            name        = d.get("name", d["id"]),
            position    = tuple(d.get("position", [0.5, 0.5])),
            personality = PersonalityProfile.from_dict(d.get("personality", {})),
            fps         = d.get("fps", 12),
        )
        duck.velocity = list(d.get("velocity", [0.0, 0.0]))
        if "emotion_blender" in d:
            duck.emotion_blender = _EmotionBlender.from_dict(d["emotion_blender"])
        duck.memory          = DuckMemory.from_dict(d.get("memory", {}), d["id"])
        duck.gaze_target_id  = d.get("gaze_target_id")
        duck.is_speaking     = d.get("is_speaking", False)
        duck.is_listening    = d.get("is_listening", False)
        return duck

    def __repr__(self) -> str:
        return (
            f"DuckEntity(id={self.id!r}, name={self.name!r}, "
            f"pos=({self.position[0]:.2f},{self.position[1]:.2f}), "
            f"personality={self.personality.label}, "
            f"emotion={self.emotion_blender.target_name})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  FACTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DUCK_NAMES = [
    "Quackers", "Waddles", "Splash", "Feathers", "Ducky",
    "Puddles", "Pebbles", "Ripple", "Daffy", "Pip",
    "Noodle", "Biscuit", "Wobble", "Flapjack", "Squeak",
]

def create_duck(
    duck_id:     str  | None = None,
    name:        str  | None = None,
    position:    tuple[float, float] | None = None,
    personality: str  | None = None,   # "shy"|"social"|"energetic"|"cautious"|"random"
    fps:         int         = 12,
) -> DuckEntity:
    """
    Factory that creates a DuckEntity with sensible defaults.

    personality  Named preset or 'random'
    """
    _id  = duck_id or f"duck_{random.randint(1000, 9999)}"
    _pos = position or (random.uniform(0.1, 0.9), random.uniform(0.15, 0.85))

    if personality == "shy":
        p = PersonalityProfile(shyness=0.75, sociability=0.25, energy=0.35, caution=0.65)
    elif personality == "social":
        p = PersonalityProfile(shyness=0.10, sociability=0.85, energy=0.60, caution=0.30)
    elif personality == "energetic":
        p = PersonalityProfile(shyness=0.20, sociability=0.60, energy=0.90, caution=0.20)
    elif personality == "cautious":
        p = PersonalityProfile(shyness=0.50, sociability=0.40, energy=0.30, caution=0.85)
    else:
        p = PersonalityProfile.random()

    _name = name or random.choice(DUCK_NAMES)

    return DuckEntity(
        duck_id=_id, name=_name, position=_pos, personality=p, fps=fps
    )
