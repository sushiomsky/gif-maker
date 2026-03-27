#!/usr/bin/env python3
"""reaction_engine.py — Event-driven reaction system for the Duckling character.

Accepts named events from EVENT_CATALOG, manages an *active* Reaction via a
bounded priority queue, and maintains a decaying emotion memory that reflects
the character's recent emotional history.

Priority interrupt rules
------------------------
- **Higher priority**          → interrupt active; re-queue active if < 60 % done.
- **Equal priority + > 50 % done** → override the finishing active reaction.
- **Lower / same but early**   → enqueue for later.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Event catalogue
# ---------------------------------------------------------------------------

EVENT_CATALOG: dict[str, dict] = {
    "idle": {
        "priority":       0,
        "target_emotion": "neutral",
        "behavior":       "idle",
        "lipsync":        False,
        "duration":       5.0,
        "description":    "Passive idle state — the duck is at rest.",
    },
    "idle_timeout": {
        "priority":       1,
        "target_emotion": "drowsy",
        "behavior":       "drowsy",
        "lipsync":        False,
        "duration":       3.0,
        "description":    "Duck grows sleepy from prolonged inactivity.",
    },
    "random_thought": {
        "priority":       2,
        "target_emotion": "curious",
        "behavior":       "thinking",
        "lipsync":        False,
        "duration":       2.5,
        "description":    "An internal thought bubble surfaces spontaneously.",
    },
    "chatter": {
        "priority":       2,
        "target_emotion": "happy",
        "behavior":       "chattering",
        "lipsync":        True,
        "duration":       3.0,
        "description":    "Duck chatters quietly to itself.",
    },
    "smug_moment": {
        "priority":       4,
        "target_emotion": "smug",
        "behavior":       "smug",
        "lipsync":        False,
        "duration":       2.0,
        "description":    "Duck looks insufferably pleased with itself.",
    },
    "user_message": {
        "priority":       5,
        "target_emotion": "attentive",
        "behavior":       "talking",
        "lipsync":        True,
        "duration":       4.0,
        "description":    "Duck responds to an incoming user message.",
    },
    "greeting": {
        "priority":       5,
        "target_emotion": "happy",
        "behavior":       "bouncy",
        "lipsync":        True,
        "duration":       2.5,
        "description":    "Duck greets a user or a new session.",
    },
    "attention": {
        "priority":       6,
        "target_emotion": "curious",
        "behavior":       "alert",
        "lipsync":        False,
        "duration":       2.0,
        "description":    "Something off-screen catches the duck's eye.",
    },
    "praise": {
        "priority":       6,
        "target_emotion": "happy",
        "behavior":       "excited",
        "lipsync":        True,
        "duration":       3.0,
        "description":    "Duck receives praise or a compliment.",
    },
    "excitement": {
        "priority":       6,
        "target_emotion": "excited",
        "behavior":       "bouncy",
        "lipsync":        True,
        "duration":       2.5,
        "description":    "Duck becomes spontaneously excited.",
    },
    "scolding": {
        "priority":       7,
        "target_emotion": "guilty",
        "behavior":       "nervous",
        "lipsync":        True,
        "duration":       4.0,
        "description":    "Duck is being scolded and looks remorseful.",
    },
    "anger": {
        "priority":       7,
        "target_emotion": "angry",
        "behavior":       "agitated",
        "lipsync":        True,
        "duration":       3.5,
        "description":    "Duck is visibly angry.",
    },
    "scared": {
        "priority":       8,
        "target_emotion": "scared",
        "behavior":       "nervous",
        "lipsync":        False,
        "duration":       2.0,
        "description":    "Duck is frightened by something.",
    },
    "loud_noise": {
        "priority":       8,
        "target_emotion": "startled",
        "behavior":       "alert",
        "lipsync":        False,
        "duration":       1.5,
        "description":    "A sudden loud noise startles the duck.",
    },
}


# ---------------------------------------------------------------------------
# Behaviour parameters
# ---------------------------------------------------------------------------

BEHAVIOR_PARAMS: dict[str, dict[str, float]] = {
    "idle":       {"head_bob_scale": 0.5,  "eye_jitter_scale": 0.4,  "blink_emphasis": 1.0},
    "talking":    {"head_bob_scale": 1.2,  "eye_jitter_scale": 1.0,  "blink_emphasis": 0.8},
    "nervous":    {"head_bob_scale": 1.5,  "eye_jitter_scale": 2.2,  "blink_emphasis": 1.4},
    "curious":    {"head_bob_scale": 1.0,  "eye_jitter_scale": 1.6,  "blink_emphasis": 0.9},
    "drowsy":     {"head_bob_scale": 0.3,  "eye_jitter_scale": 0.3,  "blink_emphasis": 2.0},
    "alert":      {"head_bob_scale": 1.8,  "eye_jitter_scale": 2.0,  "blink_emphasis": 0.5},
    "thinking":   {"head_bob_scale": 0.7,  "eye_jitter_scale": 1.2,  "blink_emphasis": 1.1},
    "excited":    {"head_bob_scale": 2.2,  "eye_jitter_scale": 1.8,  "blink_emphasis": 0.7},
    "happy":      {"head_bob_scale": 1.6,  "eye_jitter_scale": 1.0,  "blink_emphasis": 0.8},
    "sulking":    {"head_bob_scale": 0.4,  "eye_jitter_scale": 0.6,  "blink_emphasis": 1.6},
    "agitated":   {"head_bob_scale": 2.0,  "eye_jitter_scale": 2.8,  "blink_emphasis": 0.4},
    "bouncy":     {"head_bob_scale": 2.5,  "eye_jitter_scale": 1.4,  "blink_emphasis": 0.6},
    "chattering": {"head_bob_scale": 1.1,  "eye_jitter_scale": 1.0,  "blink_emphasis": 0.9},
    "smug":       {"head_bob_scale": 0.8,  "eye_jitter_scale": 0.7,  "blink_emphasis": 1.2},
}

_DEFAULT_BEHAVIOR_PARAMS: dict[str, float] = {
    "head_bob_scale":   1.0,
    "eye_jitter_scale": 1.0,
    "blink_emphasis":   1.0,
}


def get_behavior_params(behavior: str) -> dict[str, float]:
    """Return the parameter dict for *behavior*, falling back to neutral defaults.

    Args:
        behavior: Behavior name, e.g. ``"excited"`` or ``"drowsy"``.

    Returns:
        A shallow copy of the matching entry in :data:`BEHAVIOR_PARAMS`, or the
        default ``{1.0, 1.0, 1.0}`` dict when the name is not found.
    """
    return dict(BEHAVIOR_PARAMS.get(behavior, _DEFAULT_BEHAVIOR_PARAMS))


# ---------------------------------------------------------------------------
# Reaction dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reaction:
    """Represents a single in-progress or queued character reaction.

    A Reaction is created when an event fires and is either activated
    immediately (assigned to :attr:`ReactionEngine.active`) or placed into the
    engine's priority queue.

    Attributes:
        event_name:     Key from :data:`EVENT_CATALOG`.
        target_emotion: Emotion string the character should express.
        behavior:       Behavior string for :func:`get_behavior_params` lookup.
        priority:       Integer priority (0 = lowest, 10 = highest).
        lipsync:        Whether lipsync animation should accompany this reaction.
        duration:       How long the reaction lasts in seconds.
        start_time:     ``time.monotonic()`` value at the moment the reaction
                        was *activated*.  Reset by
                        :meth:`ReactionEngine._do_activate`.
    """

    event_name:     str
    target_emotion: str
    behavior:       str
    priority:       int
    lipsync:        bool
    duration:       float
    start_time:     float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------
    # Live progress tracking
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds that have passed since this reaction was activated."""
        return time.monotonic() - self.start_time

    @property
    def remaining(self) -> float:
        """Seconds until this reaction expires (clamped to ≥ 0)."""
        return max(0.0, self.duration - self.elapsed)

    @property
    def is_expired(self) -> bool:
        """``True`` when the reaction's full duration has elapsed."""
        return self.elapsed >= self.duration

    @property
    def progress(self) -> float:
        """Normalised completion fraction: 0.0 = just started, 1.0 = done."""
        if self.duration <= 0.0:
            return 1.0
        return min(1.0, self.elapsed / self.duration)


# ---------------------------------------------------------------------------
# ReactionEngine
# ---------------------------------------------------------------------------

class ReactionEngine:
    """Priority-queue-based event reactor for the Duckling character.

    Accepts named events from :data:`EVENT_CATALOG`, manages a single *active*
    reaction, a bounded queue of pending reactions, and a decaying emotion
    memory that biases toward recently experienced emotions.

    Priority interrupt rules
    ------------------------
    - **Higher priority** → interrupt active; re-queue active if < 60 % done.
    - **Equal priority + active > 50 % done** → override the finishing reaction.
    - **Lower or same (early)** → enqueue for later.

    Attributes:
        MAX_QUEUE: Maximum number of reactions held in the pending queue (8).
    """

    MAX_QUEUE: int = 8

    def __init__(self, default_emotion: str = "neutral") -> None:
        self.default_emotion:  str                  = default_emotion
        self.active:           Optional[Reaction]   = None
        self.queue:            deque[Reaction]       = deque(maxlen=self.MAX_QUEUE)
        self.history:          list[tuple]           = []      # (timestamp, event_name)
        self.emotion_memory:   dict[str, float]      = {}      # emotion → decaying weight
        self._memory_decay:    float                 = 0.93
        self._force_idle()

    # ------------------------------------------------------------------
    # Public API — triggering
    # ------------------------------------------------------------------

    def trigger(
        self,
        event_name: str,
        overrides: dict | None = None,
    ) -> bool:
        """Trigger a named event, optionally overriding catalogue fields.

        Creates a :class:`Reaction` from :data:`EVENT_CATALOG` and routes it
        through the priority interrupt logic.

        Args:
            event_name: A key from :data:`EVENT_CATALOG`.
            overrides:  Optional dict whose keys override catalogue fields,
                        e.g. ``{"duration": 6.0, "priority": 9}``.

        Returns:
            ``True`` if the reaction was *immediately activated*;
            ``False`` if it was placed in the queue (or silently dropped
            because the queue was full and the event had low priority).
        """
        catalog_entry = EVENT_CATALOG.get(event_name)
        if catalog_entry is None:
            return False

        entry = dict(catalog_entry)
        if overrides:
            entry.update(overrides)

        new_reaction = Reaction(
            event_name=event_name,
            target_emotion=entry["target_emotion"],
            behavior=entry["behavior"],
            priority=int(entry["priority"]),
            lipsync=bool(entry["lipsync"]),
            duration=float(entry["duration"]),
        )
        return self._try_activate(new_reaction)

    # ------------------------------------------------------------------
    # Public API — update tick
    # ------------------------------------------------------------------

    def update(self) -> "ReactionEngine":
        """Advance the engine by one tick (one logical frame).

        Actions performed each tick:

        1. Decay all emotion memory weights by ``_memory_decay``.
        2. Expire the active reaction if its duration has elapsed.
        3. Promote the highest-priority queued reaction, or fall back to idle.

        Returns:
            *self*, enabling optional method chaining (``engine.update().update()``).
        """
        # 1. Decay emotion memory
        for em in list(self.emotion_memory.keys()):
            self.emotion_memory[em] *= self._memory_decay
            if self.emotion_memory[em] < 0.001:
                del self.emotion_memory[em]

        # 2. Expire finished active reaction
        if self.active is not None and self.active.is_expired:
            self.active = None

        # 3. Fill vacancy from queue or fall back to idle
        if self.active is None:
            if self.queue:
                queue_list = list(self.queue)
                best_idx   = max(range(len(queue_list)),
                                 key=lambda i: queue_list[i].priority)
                best       = queue_list[best_idx]
                self.queue.clear()
                self.queue.extend(r for j, r in enumerate(queue_list) if j != best_idx)
                self._do_activate(best)
            else:
                self._force_idle()

        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_emotion(self) -> str:
        """The emotion of the currently active reaction, or the default."""
        return self.active.target_emotion if self.active else self.default_emotion

    @property
    def current_behavior(self) -> str:
        """The behavior of the currently active reaction, or ``"idle"``."""
        return self.active.behavior if self.active else "idle"

    @property
    def is_speaking(self) -> bool:
        """``True`` when the active reaction has ``lipsync`` enabled."""
        return bool(self.active and self.active.lipsync)

    @property
    def reaction_progress(self) -> float:
        """Progress of the active reaction (0.0 → 1.0), or 0.0 when none."""
        return self.active.progress if self.active else 0.0

    # ------------------------------------------------------------------
    # Emotion memory
    # ------------------------------------------------------------------

    def dominant_memory_emotion(self) -> Optional[str]:
        """Return the emotion with the highest decayed memory weight, or None.

        Returns ``None`` when the strongest weight falls below **0.05**
        (i.e. all emotions have effectively faded from memory).
        """
        if not self.emotion_memory:
            return None
        dominant = max(self.emotion_memory, key=lambda k: self.emotion_memory[k])
        if self.emotion_memory[dominant] < 0.05:
            return None
        return dominant

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def debug_state(self) -> str:
        """Return a formatted multi-line string summarising full engine state.

        Returns:
            Human-readable snapshot of emotion, behavior, active reaction,
            queue depth, emotion memory, and dominant emotion.
        """
        if self.active:
            active_str = (
                f"'{self.active.event_name}' "
                f"(pri={self.active.priority}, "
                f"{self.active.progress:.0%} done, "
                f"{self.active.remaining:.2f}s left)"
            )
        else:
            active_str = "None"

        mem_str = ", ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(self.emotion_memory.items(), key=lambda x: -x[1])
            if v >= 0.01
        ) or "(empty)"

        return "\n".join([
            f"Emotion  : {self.current_emotion}",
            f"Behavior : {self.current_behavior}",
            f"Speaking : {self.is_speaking}",
            f"Active   : {active_str}",
            f"Queue    : {len(self.queue)} / {self.MAX_QUEUE}",
            f"Memory   : {mem_str}",
            f"Dominant : {self.dominant_memory_emotion()}",
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_activate(self, reaction: Reaction) -> bool:
        """Route *reaction* through priority interrupt logic.

        Returns:
            ``True`` if *reaction* was immediately activated.
        """
        # No active reaction, or new one has strictly higher priority → interrupt
        if self.active is None or reaction.priority > self.active.priority:
            # Re-queue the active reaction if it's not idle and not too far along
            if (
                self.active is not None
                and self.active.event_name != "idle"
                and self.active.progress < 0.6
            ):
                self.queue.appendleft(self.active)
            self._do_activate(reaction)
            return True

        # Same priority and active is past its halfway point → clean override
        if (
            reaction.priority == self.active.priority
            and self.active.progress > 0.50
        ):
            self._do_activate(reaction)
            return True

        # Lower priority (or same priority but active < 50 % done) → enqueue
        self.queue.append(reaction)
        return False

    def _do_activate(self, reaction: Reaction) -> None:
        """Set *reaction* as active, stamping its start time and updating state.

        Records the event to :attr:`history` and boosts the corresponding
        emotion weight in :attr:`emotion_memory`.

        Args:
            reaction: The reaction to activate.
        """
        reaction.start_time = time.monotonic()
        self.active = reaction

        # Record to history (capped to last 200 entries to avoid unbounded growth)
        self.history.append((time.monotonic(), reaction.event_name))
        if len(self.history) > 200:
            self.history = self.history[-200:]

        # Boost emotion memory for this emotion
        emotion = reaction.target_emotion
        self.emotion_memory[emotion] = min(
            1.0,
            self.emotion_memory.get(emotion, 0.0) + 0.30,
        )

    def _force_idle(self) -> None:
        """Silently install an idle reaction without polluting history or memory.

        Called at init and whenever the queue is empty after a reaction expires.
        Does *not* call :meth:`_do_activate` so that continuous idle cycling
        does not inflate emotion memory or history.
        """
        idle = EVENT_CATALOG["idle"]
        self.active = Reaction(
            event_name="idle",
            target_emotion=idle["target_emotion"],
            behavior=idle["behavior"],
            priority=int(idle["priority"]),
            lipsync=bool(idle["lipsync"]),
            duration=float(idle["duration"]),
        )


# ---------------------------------------------------------------------------
# Demo / quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ReactionEngine(default_emotion="neutral")

    divider = "═" * 56

    print(divider)
    print("  ReactionEngine — event-driven reaction demo")
    print(divider)
    print("\n[t=+0.00s]  Initial state")
    print(engine.debug_state())

    # Events (scheduled_time_sec, event_name)
    events_to_fire: list[tuple[float, str]] = [
        (0.10, "greeting"),       # pri=5  → interrupts idle (pri=0)
        (0.40, "chatter"),        # pri=2  → queued behind greeting
        (0.65, "loud_noise"),     # pri=8  → interrupts greeting, re-queues it
        (1.15, "praise"),         # pri=6  → queued behind loud_noise
        (1.50, "user_message"),   # pri=5  → queued
        (2.20, "idle_timeout"),   # pri=1  → queued
        (2.55, "random_thought"), # pri=2  → queued
        (3.00, "smug_moment"),    # pri=4  → queued
    ]

    print("\n[Running 80 ticks at 50 ms/tick ≈ 4 s real time]\n")

    start          = time.monotonic()
    tick_interval  = 0.05   # seconds per tick
    fired: set[int] = set()

    for tick in range(80):
        now = time.monotonic() - start
        engine.update()

        for idx, (t, ev) in enumerate(events_to_fire):
            if idx not in fired and now >= t:
                activated = engine.trigger(ev)
                status    = "ACTIVATED" if activated else "queued   "
                print(f"  t={now:5.2f}s  TRIGGER '{ev:16s}'  →  {status}")
                # Indent debug state for readability
                for line in engine.debug_state().splitlines():
                    print(f"           {line}")
                print()
                fired.add(idx)

        time.sleep(tick_interval)

    print(divider)
    print("[Final state after ~4 s]")
    print(engine.debug_state())

    print("\nHistory (last 10 activations):")
    for ts, ev in engine.history[-10:]:
        print(f"  {ts:.3f}  {ev}")
