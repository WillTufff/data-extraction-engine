"""Progress reporting for long-running jobs.

A job writes a small JSON heartbeat to `data/jobs/<name>.json` as it goes;
`tools/jobs.py` reads them from any terminal.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
JOBS = ROOT / "data" / "jobs"

# How often to write the heartbeat file, in seconds.
HEARTBEAT = 0.5


@dataclass
class Snapshot:
    name: str
    label: str
    done: int
    total: int | None
    started: float
    updated: float
    pid: int
    finished: bool = False
    error: str | None = None
    note: str = ""
    history: list[tuple[float, int]] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return self.updated - self.started

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.done / self.total)

    @property
    def rate(self) -> float | None:
        """Units per second over the recent window, not the whole run."""
        if len(self.history) < 2:
            return None
        (t0, d0), (t1, d1) = self.history[0], self.history[-1]
        if t1 <= t0 or d1 <= d0:
            return None
        return (d1 - d0) / (t1 - t0)

    @property
    def eta(self) -> float | None:
        rate, total = self.rate, self.total
        if not rate or not total or self.done >= total:
            return None
        return (total - self.done) / rate


class Progress:
    """Report a job's progress to a heartbeat file and to stdout.

    Used as a context manager so the terminal state is written even when the
    job raises.
    """

    def __init__(
        self,
        name: str,
        total: int | None = None,
        label: str = "",
        every: int = 0,
        jobs_dir: Path | None = None,
    ) -> None:
        self.jobs_dir = jobs_dir or JOBS
        self.every = every
        now = time.time()
        self.snap = Snapshot(
            name=name,
            label=label or name,
            done=0,
            total=total,
            started=now,
            updated=now,
            pid=os.getpid(),
        )
        self._last_write = 0.0
        self._last_print = 0

    @property
    def path(self) -> Path:
        return self.jobs_dir / f"{self.snap.name}.json"

    def __enter__(self) -> Progress:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._write(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.snap.finished = True
        if exc is not None:
            self.snap.error = f"{exc_type.__name__}: {exc}"
        self._write(force=True)

    def step(self, n: int = 1, note: str = "") -> None:
        self.snap.done += n
        if note:
            self.snap.note = note
        self._tick()

    def set(self, done: int, note: str = "") -> None:
        """Set absolute progress, for jobs that report a position not a delta."""
        self.snap.done = done
        if note:
            self.snap.note = note
        self._tick()

    def _tick(self) -> None:
        now = time.time()
        self.snap.updated = now
        self.snap.history.append((now, self.snap.done))
        if len(self.snap.history) > 40:
            del self.snap.history[:-40]
        self._write()
        if self.every and self.snap.done - self._last_print >= self.every:
            self._last_print = self.snap.done
            print("  " + self.render(), flush=True)

    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < HEARTBEAT:
            return
        self._last_write = now
        payload = asdict(self.snap)
        payload["history"] = payload["history"][-2:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)  # atomic, so a reader never sees half a file

    def render(self) -> str:
        s = self.snap
        parts = [f"{s.done}" + (f"/{s.total}" if s.total else "")]
        if s.fraction is not None:
            parts.append(f"{s.fraction:.0%}")
        parts.append(f"{human(s.elapsed)} elapsed")
        if s.eta:
            parts.append(f"~{human(s.eta)} left")
        if s.note:
            parts.append(s.note)
        return "  ".join(parts)


def human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def read_all(jobs_dir: Path | None = None) -> list[Snapshot]:
    """Every job heartbeat, newest first."""
    jobs_dir = jobs_dir or JOBS
    out: list[Snapshot] = []
    for path in sorted(jobs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # mid-write or truncated; it will be there next tick
        data["history"] = [tuple(h) for h in data.get("history", [])]
        out.append(Snapshot(**data))
    return sorted(out, key=lambda s: s.updated, reverse=True)


def is_running(snap: Snapshot, stale_after: float = 30.0) -> bool:
    """Whether a job looks alive, based on how recently its heartbeat updated."""
    if snap.finished:
        return False
    return (time.time() - snap.updated) < stale_after
