#!/usr/bin/env python3
"""memory_system.py — Event memory with temporal decay for duck characters."""

from dataclasses import dataclass, field

MEMORY_TYPES: frozenset[str] = frozenset(
    {"interaction", "gaze", "conversation", "reaction", "observation"}
)

# Valence maps each emotion to a signed weight in [-1, +1].
EMOTION_VALENCE: dict[str, float] = {
    "happy":     +0.8,
    "excited":   +0.9,
    "neutral":    0.0,
    "confused":  -0.1,
    "sad":       -0.6,
    "angry":     -0.8,
    "surprised": +0.3,
    "sleepy":    -0.2,
    "smug":      +0.1,
    "scared":    -0.7,
}


@dataclass
class MemoryEntry:
    """A single episodic memory recorded by a duck."""

    type: str
    target_id: str | None
    emotion: str
    intensity: float      # 0.0 – 1.0; decays over time
    sim_time: float       # sim-clock value when memory was formed (immutable after creation)
    metadata: dict = field(default_factory=dict)

    @property
    def valence(self) -> float:
        """Signed emotional weight of this memory: EMOTION_VALENCE × intensity."""
        return EMOTION_VALENCE.get(self.emotion, 0.0) * self.intensity

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target_id": self.target_id,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "sim_time": self.sim_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            type=d["type"],
            target_id=d.get("target_id"),
            emotion=d["emotion"],
            intensity=float(d["intensity"]),
            sim_time=float(d["sim_time"]),
            metadata=d.get("metadata", {}),
        )


class DuckMemory:
    """Per-duck episodic memory store with exponential intensity decay.

    Design notes
    ────────────
    * Memories are stored with their *creation* sim_time, which never changes.
    * `decay()` is incremental: it applies decay for the time elapsed *since
      the previous decay call*, tracked via `_last_decay_sim_time`.  This
      prevents double-counting when decay() is called on every clock tick.
    * After decay, entries with intensity < 0.01 are pruned (effectively
      forgotten).
    * The buffer is capped at MAX_ENTRIES; oldest entry is evicted when full.
    """

    MAX_ENTRIES: int = 200
    DECAY_HALF_LIFE: float = 300.0  # sim-seconds; intensity halves every 5 sim-minutes

    def __init__(self, duck_id: str) -> None:
        self.duck_id = duck_id
        self._entries: list[MemoryEntry] = []
        # Tracks the sim_time at which intensities were last updated by decay().
        # Initialized to 0 so the first decay() call covers all elapsed time.
        self._last_decay_sim_time: float = 0.0

    # ---------------------------------------------------------------- mutation

    def record(self, entry: MemoryEntry) -> None:
        """Append a new memory entry, evicting the oldest if the buffer is full."""
        self._entries.append(entry)
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries.pop(0)  # remove earliest (index 0)

    def decay(self, current_sim_time: float) -> None:
        """Decay all intensities by the amount of time since the last decay call.

        Formula (incremental):
            intensity *= 0.5 ** (elapsed / DECAY_HALF_LIFE)
        where elapsed = current_sim_time − _last_decay_sim_time.

        Entries with intensity < 0.01 after decay are pruned.
        """
        elapsed = current_sim_time - self._last_decay_sim_time
        if elapsed <= 0:
            return

        decay_factor = 0.5 ** (elapsed / self.DECAY_HALF_LIFE)
        surviving: list[MemoryEntry] = []
        for entry in self._entries:
            entry.intensity *= decay_factor
            if entry.intensity >= 0.01:
                surviving.append(entry)
        self._entries = surviving
        self._last_decay_sim_time = current_sim_time

    # ----------------------------------------------------------------- queries

    def get_memories_of(self, target_id: str, n: int = 10) -> list[MemoryEntry]:
        """Return the N most-recent memories involving *target_id* (newest first)."""
        relevant = [e for e in self._entries if e.target_id == target_id]
        relevant.sort(key=lambda e: e.sim_time, reverse=True)
        return relevant[:n]

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most-recent memories regardless of target (newest first)."""
        return sorted(self._entries, key=lambda e: e.sim_time, reverse=True)[:n]

    def get_emotional_valence_toward(self, target_id: str) -> float:
        """Weighted-average valence of all memories about *target_id*.

        Recency weighting: the most-recent memory receives weight N, the oldest
        receives weight 1.  Returns 0.0 when there are no relevant memories.
        """
        memories = self.get_memories_of(target_id, n=self.MAX_ENTRIES)
        if not memories:
            return 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        n = len(memories)
        for rank, entry in enumerate(memories):
            weight = n - rank  # linear recency: rank 0 (newest) → weight n
            weighted_sum += entry.valence * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def get_interaction_count(self, target_id: str) -> int:
        """Total number of stored memories involving *target_id*."""
        return sum(1 for e in self._entries if e.target_id == target_id)

    def mood_score(self) -> float:
        """Average valence of the 20 most-recent memories.

        Positive → good mood; negative → bad mood; 0.0 → neutral or empty.
        """
        recent = self.get_recent(n=20)
        if not recent:
            return 0.0
        return sum(e.valence for e in recent) / len(recent)

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "duck_id": self.duck_id,
            "last_decay_sim_time": self._last_decay_sim_time,
            "entries": [e.to_dict() for e in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict, duck_id: str) -> "DuckMemory":
        memory = cls(duck_id)
        memory._last_decay_sim_time = float(data.get("last_decay_sim_time", 0.0))
        memory._entries = [MemoryEntry.from_dict(d) for d in data.get("entries", [])]
        return memory

    def __repr__(self) -> str:
        return (
            f"DuckMemory(duck={self.duck_id!r}, entries={len(self._entries)}, "
            f"mood={self.mood_score():+.2f})"
        )


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mem = DuckMemory("Quackers")

    entries = [
        MemoryEntry("interaction", "Waddles",  "happy",    1.0, sim_time=0.0),
        MemoryEntry("gaze",        "Splash",   "confused", 0.8, sim_time=50.0),
        MemoryEntry("conversation","Waddles",  "excited",  0.9, sim_time=120.0),
        MemoryEntry("reaction",    "Feathers", "angry",    0.7, sim_time=200.0),
        MemoryEntry("observation", None,       "sleepy",   0.5, sim_time=250.0),
    ]
    for e in entries:
        mem.record(e)

    print("Before decay:", mem)
    print("Valence toward Waddles :", f"{mem.get_emotional_valence_toward('Waddles'):+.3f}")
    print("Mood score             :", f"{mem.mood_score():+.3f}")

    mem.decay(current_sim_time=300.0)
    print("\nAfter 300s decay:", mem)
    print("Valence toward Waddles :", f"{mem.get_emotional_valence_toward('Waddles'):+.3f}")

    # Round-trip serialization
    restored = DuckMemory.from_dict(mem.to_dict(), "Quackers")
    assert len(restored._entries) == len(mem._entries)
    print("\nSerialization round-trip OK.")
