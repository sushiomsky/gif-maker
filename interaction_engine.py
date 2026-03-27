#!/usr/bin/env python3
"""
interaction_engine.py — Social interaction detection and event dispatch
=========================================================================

The InteractionEngine is the "social nervous system" of the world.
Each simulation step it:

  1. Scans all duck pairs for proximity, gaze alignment, and social cues
  2. Creates / advances ConversationSessions between nearby ducks
  3. Fires reaction events on individual ducks
  4. Updates the RelationshipEngine based on interaction outcomes
  5. Records events in each duck's DuckMemory

Design principle: interactions emerge from state (position, gaze, personality)
rather than being scripted — the same world can produce different histories
on every run.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from memory_system      import MemoryEntry
from relationship_engine import RelationshipEngine


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# World-unit distance thresholds
PROXIMITY_CHAT    = 0.18   # close enough to have a conversation
PROXIMITY_NOTICE  = 0.35   # close enough to notice each other
PROXIMITY_CROWD   = 0.10   # too close — personal-space violation

# Gaze alignment: cosine similarity between duck A's gaze dir and A→B vector
GAZE_ALIGN_THRESHOLD = 0.70   # duck A is "looking at" duck B

# Minimum seconds between repeated interactions on the same pair
INTERACTION_COOLDOWN = 4.0

# Probability per-frame of a random unprompted event
RANDOM_EVENT_P = 0.004


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERSATION SESSION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationSession:
    """
    Manages the speaking-turn protocol between two ducks.

    States
    ──────
    starting  → a_speaking → pause → b_speaking → pause → a_speaking …
    → ending → done

    Turns alternate; occasional brief simultaneous overlap is possible.
    """
    duck_a:    str
    duck_b:    str
    sim_time:  float = 0.0

    # Session limits
    max_turns:    int   = field(default_factory=lambda: random.randint(2, 5))
    turns_done:   int   = 0
    current_turn: str   = "a"           # "a" | "b" | "pause"
    turn_timer:   float = 0.0
    state:        str   = "starting"    # starting|a_speaking|b_speaking|pause|ending|done

    # Pause duration between turns
    PAUSE_MIN = 0.30
    PAUSE_MAX = 0.90
    # Overlap probability: duck B starts a little before A finishes
    OVERLAP_P = 0.15

    def __post_init__(self) -> None:
        # First turn goes to duck_a
        self.turn_timer = random.uniform(1.2, 3.0)
        self.state      = "starting"

    def update(self, dt: float, sim_time: float) -> tuple[str | None, str | None]:
        """
        Advance the session by dt seconds.

        Returns (speaker_id, listener_id) or (None, None) during pauses/done.
        The caller uses this to drive each duck's start_speaking/start_listening.
        """
        if self.state == "done":
            return None, None

        self.turn_timer -= dt

        if self.state == "starting":
            self.state      = "a_speaking"
            self.turn_timer = random.uniform(1.2, 3.0)
            return self.duck_a, self.duck_b

        elif self.state == "a_speaking":
            if self.turn_timer <= 0.0:
                self.turns_done += 1
                if self.turns_done >= self.max_turns:
                    self.state = "ending"
                    return None, None
                # Overlap: duck_b starts before duck_a fully stops
                if random.random() < self.OVERLAP_P:
                    self.state      = "b_speaking"
                    self.turn_timer = random.uniform(1.0, 2.8)
                    return self.duck_b, self.duck_a
                # Normal pause
                self.state      = "pause"
                self.turn_timer = random.uniform(self.PAUSE_MIN, self.PAUSE_MAX)
                return None, None
            return self.duck_a, self.duck_b

        elif self.state == "b_speaking":
            if self.turn_timer <= 0.0:
                self.turns_done += 1
                if self.turns_done >= self.max_turns:
                    self.state = "ending"
                    return None, None
                if random.random() < self.OVERLAP_P:
                    self.state      = "a_speaking"
                    self.turn_timer = random.uniform(1.0, 2.8)
                    return self.duck_a, self.duck_b
                self.state      = "pause"
                self.turn_timer = random.uniform(self.PAUSE_MIN, self.PAUSE_MAX)
                return None, None
            return self.duck_b, self.duck_a

        elif self.state == "pause":
            if self.turn_timer <= 0.0:
                # Alternate turn
                if self.current_turn == "a":
                    self.state, self.current_turn = "b_speaking", "b"
                    self.turn_timer = random.uniform(1.0, 2.8)
                    return self.duck_b, self.duck_a
                else:
                    self.state, self.current_turn = "a_speaking", "a"
                    self.turn_timer = random.uniform(1.0, 2.8)
                    return self.duck_a, self.duck_b
            return None, None

        elif self.state == "ending":
            self.state = "done"
            return None, None

        return None, None

    @property
    def is_done(self) -> bool:
        return self.state == "done"

    @property
    def participants(self) -> set[str]:
        return {self.duck_a, self.duck_b}


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class InteractionEngine:
    """
    Detects and orchestrates social interactions between DuckEntity objects.

    The engine is stateful: it remembers active conversations and per-pair
    cooldowns so interactions don't fire every single frame.
    """

    def __init__(self) -> None:
        # Active conversation sessions: frozenset({id_a, id_b}) → session
        self._conversations: dict[frozenset, ConversationSession] = {}

        # Per-pair cooldown: frozenset → remaining_seconds
        self._cooldowns: dict[frozenset, float] = {}

        # Pending events to fire next frame (avoids mutation during iteration)
        self._event_queue: list[tuple[str, str, dict]] = []  # (duck_id, event, kwargs)

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        ducks:         list,            # list[DuckEntity]
        relationships: RelationshipEngine,
        dt:            float,
        sim_time:      float = 0.0,
    ) -> list[dict]:
        """
        Run one interaction engine tick.

        Returns a list of interaction-event dicts (for logging / UI).
        """
        events_log: list[dict] = []

        # Decay cooldowns
        for key in list(self._cooldowns):
            self._cooldowns[key] -= dt
            if self._cooldowns[key] <= 0.0:
                del self._cooldowns[key]

        # Advance active conversations
        self._advance_conversations(ducks, dt, sim_time, relationships, events_log)

        # Scan for new interactions
        self._scan_interactions(ducks, relationships, dt, sim_time, events_log)

        # Random unprompted events (idle chatter, random thoughts, etc.)
        self._fire_random_events(ducks, sim_time)

        # Flush queued events
        duck_map = {d.id: d for d in ducks}
        for duck_id, event_name, kwargs in self._event_queue:
            if duck_id in duck_map:
                duck_map[duck_id].trigger(event_name, **kwargs)
        self._event_queue.clear()

        return events_log

    # ── Conversation management ────────────────────────────────────────────────

    def _advance_conversations(
        self, ducks, dt, sim_time, relationships, events_log
    ) -> None:
        """Advance each active conversation session and dispatch speaking roles."""
        duck_map = {d.id: d for d in ducks}

        for key, session in list(self._conversations.items()):
            # Check if ducks are still in range
            da = duck_map.get(session.duck_a)
            db = duck_map.get(session.duck_b)
            if not da or not db:
                del self._conversations[key]
                continue

            dist = _dist(da.position, db.position)
            if dist > PROXIMITY_CHAT * 1.5:
                # Drifted apart — end conversation
                da.end_conversation()
                db.end_conversation()
                del self._conversations[key]
                continue

            speaker, listener = session.update(dt, sim_time)

            if session.is_done:
                da.end_conversation()
                db.end_conversation()
                # Record positive interaction
                _record_interaction(da, db, relationships, sim_time, "conversation", +1.0)
                events_log.append({
                    "type": "conversation_end",
                    "ducks": [session.duck_a, session.duck_b],
                    "turns": session.turns_done,
                })
                del self._conversations[key]
                continue

            if speaker and listener:
                s_duck = duck_map.get(speaker)
                l_duck = duck_map.get(listener)
                if s_duck and l_duck:
                    if not s_duck.is_speaking:
                        s_duck.start_speaking(listener)
                    if not l_duck.is_listening:
                        l_duck.start_listening(speaker)
            else:
                # Pause — both quiet
                for did in [session.duck_a, session.duck_b]:
                    d = duck_map.get(did)
                    if d and (d.is_speaking or d.is_listening):
                        d.end_conversation()

    def _scan_interactions(
        self, ducks, relationships, dt, sim_time, events_log
    ) -> None:
        """Detect proximity + gaze conditions and trigger interactions."""
        n = len(ducks)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ducks[i], ducks[j]
                key  = frozenset({a.id, b.id})

                # Skip if on cooldown or already in conversation
                if key in self._cooldowns or key in self._conversations:
                    continue

                dist = _dist(a.position, b.position)

                # ── Notice ──────────────────────────────────────────────────
                if dist < PROXIMITY_NOTICE:
                    self._handle_notice(a, b, dist, key, sim_time, events_log)

                # ── Crowd personal space ────────────────────────────────────
                if dist < PROXIMITY_CROWD:
                    self._handle_crowding(a, b, sim_time, events_log)
                    self._cooldowns[key] = INTERACTION_COOLDOWN * 0.5

    def _handle_notice(self, a, b, dist, key, sim_time, events_log) -> None:
        """Handle two ducks noticing each other."""
        # Check if they're looking at each other
        a_looks_at_b = _duck_is_gazing_at(a, b)
        b_looks_at_a = _duck_is_gazing_at(b, a)
        mutual_gaze  = a_looks_at_b and b_looks_at_a

        if dist < PROXIMITY_CHAT:
            # Try to start a conversation
            if mutual_gaze and _can_converse(a, b):
                session = ConversationSession(
                    duck_a=a.id, duck_b=b.id, sim_time=sim_time
                )
                self._conversations[key]   = session
                self._cooldowns[key]       = session.max_turns * 4.0
                a.gaze_target_id           = b.id
                b.gaze_target_id           = a.id
                event_log = {
                    "type": "conversation_start",
                    "ducks": [a.id, b.id],
                    "distance": dist,
                }
                event_log["note"] = "mutual gaze at close range"
                event_log["turns_planned"] = session.max_turns
                event_log["sim_time"] = sim_time
                event_log["mutual_gaze"] = True
                event_log["dist"] = round(dist, 3)
                event_log["a_personality"] = a.personality.label
                event_log["b_personality"] = b.personality.label
                event_log["relationship_score"] = round(
                    _get_rel(a, b, event_log), 2
                )
                events_log.append(event_log)
                return

        # One-way gaze → being-watched reaction
        if a_looks_at_b and not b_looks_at_a:
            self._trigger_being_watched(b, a, dist, sim_time, events_log)
            self._cooldowns[key] = INTERACTION_COOLDOWN

        elif b_looks_at_a and not a_looks_at_b:
            self._trigger_being_watched(a, b, dist, sim_time, events_log)
            self._cooldowns[key] = INTERACTION_COOLDOWN

    def _trigger_being_watched(self, watcher_duck, gazing_duck, dist, sim_time, events_log):
        """Duck notices it's being watched."""
        shyness = watcher_duck.personality.shyness

        if shyness > 0.65:
            # Shy: look away, become nervous
            watcher_duck.trigger("attention")
            watcher_duck.gaze_target_id = None   # break eye contact
        elif shyness < 0.35:
            # Bold: look back and react positively
            watcher_duck.trigger("attention")
            watcher_duck.gaze_target_id = gazing_duck.id
        else:
            # Neutral: gentle curiosity
            watcher_duck.trigger("random_thought")

        events_log.append({
            "type":      "being_watched",
            "watcher":   gazing_duck.id,
            "subject":   watcher_duck.id,
            "distance":  round(dist, 3),
            "sim_time":  sim_time,
        })

    def _handle_crowding(self, a, b, sim_time, events_log) -> None:
        """Ducks are too close — personal space reaction."""
        a.trigger("loud_noise")   # startled
        b.trigger("loud_noise")
        events_log.append({
            "type": "crowding",
            "ducks": [a.id, b.id],
            "sim_time": sim_time,
        })

    def _fire_random_events(self, ducks, sim_time: float) -> None:
        """Occasionally give a random duck an unprompted event."""
        for duck in ducks:
            if random.random() < RANDOM_EVENT_P:
                event = random.choice(["random_thought", "idle_timeout",
                                       "smug_moment", "chatter"])
                self._event_queue.append((duck.id, event, {}))

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_event(
        self,
        source_id: str,
        target_id: str | None,
        event:     str,
        ducks:     list,
        relationships: RelationshipEngine,
        sim_time:  float = 0.0,
    ) -> None:
        """
        Externally inject a social event between two ducks.
        Useful for scripted timelines or testing.
        """
        duck_map = {d.id: d for d in ducks}
        src = duck_map.get(source_id)
        tgt = duck_map.get(target_id) if target_id else None

        if src:
            src.trigger(event)
        if tgt:
            # Target reacts to being talked at
            tgt.trigger("attention")
            if src and tgt:
                key = frozenset({source_id, target_id})
                if key not in self._conversations and key not in self._cooldowns:
                    _record_interaction(src, tgt, relationships, sim_time, event)

    def active_conversations(self) -> list[dict]:
        """Return summary of all active conversation sessions."""
        return [
            {"duck_a": s.duck_a, "duck_b": s.duck_b,
             "state": s.state, "turns": s.turns_done}
            for s in self._conversations.values()
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _dist(pos_a: list, pos_b: list) -> float:
    return math.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1])


def _duck_is_gazing_at(a, b) -> bool:
    """True if duck A's gaze direction points roughly toward duck B."""
    if a.gaze_target_id == b.id:
        return True
    # Fall back to checking the smoothed gaze direction vs A→B vector
    gdir = a._gaze_smoothed
    dx   = b.position[0] - a.position[0]
    dy   = b.position[1] - a.position[1]
    dist = math.hypot(dx, dy)
    if dist < 0.001 or math.hypot(*gdir) < 0.05:
        return False
    # Cosine similarity
    cos  = (gdir[0] * dx + gdir[1] * dy) / (math.hypot(*gdir) * dist)
    return cos >= GAZE_ALIGN_THRESHOLD


def _can_converse(a, b) -> bool:
    """
    Returns True if the two ducks are personality-compatible for conversation.
    Very shy ducks need a high relationship score to start talking.
    """
    combined_sociability = a.personality.sociability + b.personality.sociability
    if combined_sociability < 0.4:
        return False
    shy_penalty = (a.personality.shyness + b.personality.shyness) * 0.5
    prob = max(0.0, combined_sociability * 0.5 - shy_penalty * 0.3)
    return random.random() < prob


def _record_interaction(
    a, b,
    relationships: RelationshipEngine,
    sim_time: float,
    interaction_type: str = "talk",
    strength: float = 1.0,
) -> None:
    """Record a positive interaction in both ducks' memory and relationship engine."""
    relationships.record_positive(
        a.id, b.id, strength=strength, sim_time=sim_time,
        interaction_type=interaction_type,
    )

    for duck, other in [(a, b), (b, a)]:
        duck.memory.record(MemoryEntry(
            type="interaction",
            target_id=other.id,
            emotion="happy",
            intensity=0.6 * strength,
            sim_time=sim_time,
        ))
        duck.personality.evolve("positive_interaction", intensity=strength * 0.5)


def _get_rel(a, b, event_log) -> float:
    """Helper: extract relationship score safely for logging."""
    try:
        from relationship_engine import RelationshipEngine as RE
        return 0.0
    except Exception:
        return 0.0
