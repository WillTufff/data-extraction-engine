"""Show what the long jobs are doing.

    uv run python tools/jobs.py           # one look
    uv run python tools/jobs.py --watch   # refresh until interrupted

Reads the heartbeats under `data/jobs/`, so it works from any terminal and
does not care whether the job was started in the foreground, in the
background, or by something else entirely.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.progress import human, is_running, read_all  # noqa: E402

BAR = 28


def bar(fraction: float | None) -> str:
    if fraction is None:
        return "?" * 0 + " " * BAR
    filled = int(round(fraction * BAR))
    return "█" * filled + "·" * (BAR - filled)


def status(snap) -> str:
    if snap.error:
        return "failed"
    if snap.finished:
        return "done"
    if not is_running(snap):
        return "stalled"
    return "running"


def render(width: int) -> list[str]:
    snaps = read_all()
    if not snaps:
        return ["no jobs recorded yet"]

    lines = []
    for s in snaps:
        state = status(s)
        head = f"{s.label[:28]:28s} [{bar(s.fraction)}] {state:>7s}"
        counts = f"{s.done}" + (f"/{s.total}" if s.total else "")
        tail = f"  {counts}  {human(s.elapsed)}"
        if state == "running" and s.eta:
            tail += f"  ~{human(s.eta)} left"
        if s.rate:
            tail += f"  {s.rate:.1f}/s"
        lines.append((head + tail)[:width])
        if s.error:
            lines.append(f"    {s.error}"[:width])
        elif s.note:
            lines.append(f"    {s.note}"[:width])
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    width = shutil.get_terminal_size((100, 24)).columns
    if not args.watch:
        print("\n".join(render(width)))
        return 0

    try:
        while True:
            lines = render(width)
            print("\033[H\033[J" + "\n".join(lines), flush=True)
            if all(status(s) in ("done", "failed") for s in read_all()):
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
