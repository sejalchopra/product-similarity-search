"""Progress lines and stage timers for scripts/evaluate.py."""

from __future__ import annotations

import time

_origin: float | None = None
_last_heartbeat: float = 0.0
HEARTBEAT_SEC = 30.0


def start() -> None:
    global _origin, _last_heartbeat
    _origin = time.perf_counter()
    _last_heartbeat = 0.0


def log(msg: str) -> None:
    global _origin
    if _origin is None:
        start()
    elapsed = time.perf_counter() - _origin
    print(f"[eval {elapsed:7.1f}s] {msg}", flush=True)


class StageTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.total = 0
        self.done = 0
        self.t0 = 0.0
        self.elapsed = 0.0

    def start(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.t0 = time.perf_counter()
        log(f"stage {self.name}: start (0/{total})")

    def tick(self, done: int) -> None:
        self.done = done
        now = time.perf_counter()
        global _last_heartbeat
        if now - _last_heartbeat >= HEARTBEAT_SEC or done >= self.total:
            _last_heartbeat = now
            elapsed = now - self.t0
            rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
            eta = (self.total - done) / rate if rate > 0 else 0.0
            log(f"stage {self.name}: {done}/{self.total} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    def finish(self) -> float:
        self.elapsed = time.perf_counter() - self.t0
        log(f"stage {self.name}: done in {self.elapsed:.1f}s")
        return self.elapsed


def print_runtime_table(stages: dict[str, float]) -> None:
    total = sum(stages.values())
    log("--- runtime per stage ---")
    for name, sec in stages.items():
        pct = 100.0 * sec / total if total > 0 else 0.0
        log(f"  {name}: {sec:.1f}s ({pct:.0f}%)")
    log(f"  total: {total:.1f}s")
