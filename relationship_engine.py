#!/usr/bin/env python3
"""relationship_engine.py — Pairwise relationship engine for duck social dynamics."""

from dataclasses import dataclass


@dataclass
class RelationshipData:
    """Bidirectional relationship state for one duck pair.

    score: -1.0 (hostile) → 0.0 (neutral) → +1.0 (best friends).
    """

    score: float = 0.0
    interaction_count: int = 0
    last_interaction_time: float = 0.0
    last_interaction_type: str = "none"

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "interaction_count": self.interaction_count,
            "last_interaction_time": self.last_interaction_time,
            "last_interaction_type": self.last_interaction_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipData":
        return cls(
            score=float(d["score"]),
            interaction_count=int(d["interaction_count"]),
            last_interaction_time=float(d["last_interaction_time"]),
            last_interaction_type=d["last_interaction_type"],
        )


class RelationshipEngine:
    """Manages pairwise relationships between any number of ducks.

    Design notes
    ────────────
    * Pairs are stored under a *canonical* key: the alphabetically-sorted
      (a, b) tuple.  This means (A,B) and (B,A) always resolve to the same
      record, keeping the relationship symmetric by construction.
    * decay_all() pulls every score toward 0.0 (neutral) proportionally.
      The formula is: score += (0 - score) * DECAY_RATE * dt_sim.
      A guard prevents overshoot: if the computed change exceeds the remaining
      distance to 0, the score is set exactly to 0.
    """

    DECAY_RATE: float = 0.002     # fraction of remaining distance decayed per sim-second
    MAX_SCORE: float = 1.0
    MIN_SCORE: float = -1.0
    POSITIVE_DELTA: float = 0.08  # base score increase per positive interaction
    NEGATIVE_DELTA: float = 0.12  # base score decrease per negative interaction

    def __init__(self) -> None:
        self._pairs: dict[tuple[str, str], RelationshipData] = {}

    # --------------------------------------------------------- internal helpers

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        """Return a canonical sorted pair key so (A,B) ≡ (B,A)."""
        return (a, b) if a < b else (b, a)

    # -------------------------------------------------------------- accessors

    def get(self, a: str, b: str) -> RelationshipData:
        """Retrieve (or lazily create) the relationship record for duck pair (a, b)."""
        key = self._key(a, b)
        if key not in self._pairs:
            self._pairs[key] = RelationshipData()
        return self._pairs[key]

    def get_score(self, a: str, b: str) -> float:
        """Return the current relationship score between *a* and *b*."""
        return self.get(a, b).score

    # --------------------------------------------------------- event recording

    def record_positive(
        self,
        a: str,
        b: str,
        strength: float = 1.0,
        sim_time: float = 0.0,
        interaction_type: str = "talk",
    ) -> None:
        """Record a positive interaction; score rises by POSITIVE_DELTA × strength."""
        rel = self.get(a, b)
        rel.score = min(self.MAX_SCORE, rel.score + self.POSITIVE_DELTA * strength)
        rel.interaction_count += 1
        rel.last_interaction_time = sim_time
        rel.last_interaction_type = interaction_type

    def record_negative(
        self,
        a: str,
        b: str,
        strength: float = 1.0,
        sim_time: float = 0.0,
        interaction_type: str = "conflict",
    ) -> None:
        """Record a negative interaction; score drops by NEGATIVE_DELTA × strength."""
        rel = self.get(a, b)
        rel.score = max(self.MIN_SCORE, rel.score - self.NEGATIVE_DELTA * strength)
        rel.interaction_count += 1
        rel.last_interaction_time = sim_time
        rel.last_interaction_type = interaction_type

    # ------------------------------------------------------------------ decay

    def decay_all(self, dt_sim: float) -> None:
        """Nudge every relationship score toward 0 (neutral) over *dt_sim* seconds.

        Formula: score += (0 − score) × DECAY_RATE × dt_sim

        The change is clamped so it cannot overshoot 0 (prevents sign flips on
        large dt values when DECAY_RATE × dt_sim ≥ 1).
        """
        for rel in self._pairs.values():
            if rel.score == 0.0:
                continue
            change = (0.0 - rel.score) * self.DECAY_RATE * dt_sim
            # Prevent overshoot: the change must not be larger in magnitude than
            # the score itself (which would flip the sign past neutral).
            if abs(change) >= abs(rel.score):
                rel.score = 0.0
            else:
                rel.score += change

    # --------------------------------------------------------------- queries

    def get_most_liked(self, duck_id: str, all_duck_ids: list[str]) -> str | None:
        """Return the duck (from *all_duck_ids*) that *duck_id* likes most."""
        others = [d for d in all_duck_ids if d != duck_id]
        if not others:
            return None
        return max(others, key=lambda d: self.get_score(duck_id, d))

    def get_least_liked(self, duck_id: str, all_duck_ids: list[str]) -> str | None:
        """Return the duck (from *all_duck_ids*) that *duck_id* likes least."""
        others = [d for d in all_duck_ids if d != duck_id]
        if not others:
            return None
        return min(others, key=lambda d: self.get_score(duck_id, d))

    def get_all_scores(self, duck_id: str, all_duck_ids: list[str]) -> dict[str, float]:
        """Return {other_id: score} for every duck in *all_duck_ids* except self."""
        return {d: self.get_score(duck_id, d) for d in all_duck_ids if d != duck_id}

    def get_friends(
        self,
        duck_id: str,
        all_duck_ids: list[str],
        threshold: float = 0.3,
    ) -> list[str]:
        """Return ducks whose score with *duck_id* is ≥ *threshold*."""
        return [
            d for d in all_duck_ids
            if d != duck_id and self.get_score(duck_id, d) >= threshold
        ]

    def get_rivals(
        self,
        duck_id: str,
        all_duck_ids: list[str],
        threshold: float = -0.3,
    ) -> list[str]:
        """Return ducks whose score with *duck_id* is ≤ *threshold*."""
        return [
            d for d in all_duck_ids
            if d != duck_id and self.get_score(duck_id, d) <= threshold
        ]

    def total_pairs(self) -> int:
        """Number of unique duck pairs tracked."""
        return len(self._pairs)

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        """Serialize all relationship pairs.  Keys are 'a|b' strings."""
        return {f"{k[0]}|{k[1]}": v.to_dict() for k, v in self._pairs.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipEngine":
        """Restore a RelationshipEngine from a previously serialized dict."""
        engine = cls()
        for key_str, rel_data in d.items():
            a, b = key_str.split("|", 1)
            engine._pairs[(a, b)] = RelationshipData.from_dict(rel_data)
        return engine

    def __repr__(self) -> str:
        return f"RelationshipEngine(pairs={self.total_pairs()})"


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ducks = ["Quackers", "Waddles", "Splash", "Feathers"]
    engine = RelationshipEngine()

    print("═" * 50)
    print("  Duck Relationship Engine — Demo")
    print("═" * 50)

    # --- Friendly encounters ---
    engine.record_positive("Quackers", "Waddles",  strength=1.2, sim_time=100.0, interaction_type="play")
    engine.record_positive("Quackers", "Waddles",  strength=0.8, sim_time=200.0, interaction_type="share_food")
    engine.record_positive("Splash",   "Quackers", strength=0.5, sim_time=150.0, interaction_type="talk")

    # --- Conflicts ---
    engine.record_negative("Splash",   "Feathers", strength=1.5, sim_time=160.0, interaction_type="argument")
    engine.record_negative("Waddles",  "Feathers", strength=1.0, sim_time=210.0, interaction_type="territorial")

    print("\nScores after interactions:")
    for a in ducks:
        for b in ducks:
            if a < b:
                score = engine.get_score(a, b)
                bar = "█" * int(abs(score) * 10)
                sign = "+" if score >= 0 else "-"
                print(f"  {a:10s} ↔ {b:10s}  {sign}{abs(score):.3f}  {bar}")

    print(f"\nQuackers' best friend : {engine.get_most_liked('Quackers', ducks)}")
    print(f"Quackers' least liked : {engine.get_least_liked('Quackers', ducks)}")
    print(f"Quackers' friends     : {engine.get_friends('Quackers', ducks)}")
    print(f"Splash's rivals       : {engine.get_rivals('Splash', ducks)}")
    print(f"Total tracked pairs   : {engine.total_pairs()}")

    # --- Decay simulation ---
    print("\nSimulating 500 sim-seconds of drift toward neutral…")
    engine.decay_all(500.0)

    print("\nScores after decay:")
    for a in ducks:
        for b in ducks:
            if a < b:
                score = engine.get_score(a, b)
                print(f"  {a:10s} ↔ {b:10s}  {score:+.3f}")

    # --- Serialization round-trip ---
    serialized = engine.to_dict()
    engine2 = RelationshipEngine.from_dict(serialized)
    assert engine2.total_pairs() == engine.total_pairs()
    assert abs(engine2.get_score("Quackers", "Waddles") - engine.get_score("Quackers", "Waddles")) < 1e-9
    print(f"\nSerialization round-trip OK — {engine2.total_pairs()} pairs restored.")
