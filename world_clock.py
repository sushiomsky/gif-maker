#!/usr/bin/env python3
"""world_clock.py — Persistent simulation time for the Duck World."""

import math
import time


class WorldClock:
    """
    Tracks simulation time that persists across program runs.

    sim_time  = total accumulated simulation seconds (survives serialization).
    time_scale = ratio of sim-time to real-time (1.0 = real-time, 2.0 = 2× fast).

    Typical usage per game loop tick:
        dt_sim = clock.tick()   # measure real elapsed & advance
        save(clock.to_dict())   # persist to JSON between sessions
    """

    def __init__(self, sim_time: float = 0.0, time_scale: float = 1.0) -> None:
        self.sim_time = sim_time
        self.time_scale = time_scale
        self._session_start: float = time.monotonic()
        self._last_tick: float = time.monotonic()

    # ------------------------------------------------------------------ ticking

    def tick(self, dt_real: float | None = None) -> float:
        """Advance the clock by one step.

        If *dt_real* is supplied, use it directly (useful for deterministic
        replays or tests).  Otherwise, measure wall-clock elapsed since the
        previous tick.  Returns the sim-time delta that was applied.
        """
        now = time.monotonic()
        measured = now - self._last_tick
        dt = dt_real if dt_real is not None else measured
        dt_sim = dt * self.time_scale
        self.sim_time += dt_sim
        self._last_tick = now
        return dt_sim

    def reset_session(self) -> None:
        """Re-anchor _last_tick to now.

        Call this after a long pause (e.g. the process was suspended) to
        prevent a single huge dt from spiking the simulation forward.
        """
        self._last_tick = time.monotonic()

    # ---------------------------------------------------------------- properties

    @property
    def sim_day(self) -> int:
        """Whole sim-days elapsed (0-indexed)."""
        return int(self.sim_time // 86_400)

    @property
    def sim_hour(self) -> float:
        """Fractional hour within the current sim-day (0.0 – 24.0)."""
        return (self.sim_time % 86_400) / 3_600.0

    @property
    def formatted(self) -> str:
        """Human-readable timestamp: 'Day 3, 14:22:05'."""
        seconds_today = self.sim_time % 86_400
        h = int(seconds_today // 3_600)
        m = int((seconds_today % 3_600) // 60)
        s = int(seconds_today % 60)
        return f"Day {self.sim_day}, {h:02d}:{m:02d}:{s:02d}"

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        """Return a JSON-safe dict; call from_dict() to restore."""
        return {"sim_time": self.sim_time, "time_scale": self.time_scale}

    @classmethod
    def from_dict(cls, d: dict) -> "WorldClock":
        """Restore a WorldClock from a previously serialized dict."""
        return cls(sim_time=float(d["sim_time"]), time_scale=float(d["time_scale"]))

    # -------------------------------------------------------------------  misc

    def __repr__(self) -> str:
        return f"WorldClock({self.formatted}, scale={self.time_scale}×)"


# ---------------------------------------------------------------------------
# Stand-alone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clock = WorldClock(sim_time=0.0, time_scale=60.0)  # 1 real-second → 1 sim-minute
    print("Starting:", clock)

    for _ in range(5):
        time.sleep(0.05)
        clock.tick()

    print("After 5 ticks:", clock)

    # Round-trip serialization
    restored = WorldClock.from_dict(clock.to_dict())
    print("Restored:", restored)
    assert math.isclose(clock.sim_time, restored.sim_time)
    print("Serialization round-trip OK.")
