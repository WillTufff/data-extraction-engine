"""Read the live scoreboard over the whole VOD, once, into a tunable cache.

Makes one pass over the cached panel strips and stores a run-length timeline
(reading, first seen, last seen) per slot rather than a reading per frame.

Only LIVE seconds are read; the panels are also drawn during replays but show
the same state twice, which would double every run.

The pass is chunked across processes: a strip is cheap to decode, so the
reader is the bottleneck and splitting the frame range scales close to
linearly. Runs are closed at each chunk boundary and rejoined by `_stitch`.

    uv run python tools/scan_scoreboard.py --out data/scoreboard_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Two runs either side of a chunk boundary are the same run if they are this
# close. One strip frame at 6fps is 0.167s; anything larger is a real gap.
BOUNDARY_GAP = 0.2


class Composite:
    """Pastes the two panel strips back into one full-frame coordinate system.

    The reader addresses absolute source coordinates and the mirror check
    needs both panels present at once.
    """

    def __init__(self, crops, width: int = 1920, height: int = 1080) -> None:
        self.crops = list(crops)
        self._buf = np.zeros((height, width, 3), dtype=np.uint8)

    def place(self, frames) -> np.ndarray:
        for crop, frame in zip(self.crops, frames):
            self._buf[crop.y : crop.y + crop.height,
                      crop.x : crop.x + crop.width] = frame
        return self._buf


def live_spans(timeline: Path) -> list[tuple[float, float]]:
    return [
        (s["start"] + 1, s["end"] - 1)
        for s in json.loads(timeline.read_text())
        if s["state"] == "live" and s["end"] - s["start"] > 2
    ]


def _is_live(spans, ts: float) -> bool:
    return any(a <= ts <= b for a, b in spans)


_STATE: dict = {}


def _init(left_path: str, right_path: str, timeline: str) -> None:
    _STATE["left"] = video.load_strip(left_path)
    _STATE["right"] = video.load_strip(right_path)
    _STATE["spans"] = live_spans(Path(timeline))
    _STATE["reader"] = sb.Reader.load()
    _STATE["composite"] = Composite([_STATE["left"].crop, _STATE["right"].crop])


def _chunk(job: tuple[int, int]) -> dict:
    """Read one frame range, returning its runs and its counters.

    Runs are closed at the chunk boundary and rejoined by `_stitch`, so the
    split is invisible in the output.
    """
    first, count = job
    left, right = _STATE["left"], _STATE["right"]
    spans, reader = _STATE["spans"], _STATE["reader"]
    composite = _STATE["composite"]

    runs: list[dict] = []
    open_runs: dict[tuple[str, int], dict] = {}
    read = layout_errors = unread = 0

    streams = zip(
        video.stream_strip(left, first, count),
        video.stream_strip(right, first, count),
    )
    for (ts, lframe), (_, rframe) in streams:
        if not _is_live(spans, ts):
            runs += _close(open_runs)
            continue
        try:
            reading = reader.read(composite.place((lframe, rframe)))
        except sb.LayoutError:
            # Panels not where expected; close every run rather than attribute
            # this frame's pixels to the previous reading.
            layout_errors += 1
            runs += _close(open_runs)
            continue
        read += 1

        for row in reading.rows:
            key = (row.side, row.slot)
            value = _value(row, reading.stat_label)
            run = open_runs.get(key)
            if run is not None and run["value"] == value:
                run["end"] = ts
                run["samples"] += 1
                continue
            if run is not None:
                runs.append(run)
            open_runs[key] = {
                "side": row.side, "slot": row.slot,
                "value": value, "start": ts, "end": ts, "samples": 1,
            }
            if row.kills is None or row.name is None:
                unread += 1
    runs += _close(open_runs)
    return {"first": first, "runs": runs, "read": read,
            "layout_errors": layout_errors, "unread": unread}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="data/strip_scoreboard_left.mp4")
    ap.add_argument("--right", default="data/strip_scoreboard_right.mp4")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--out", default="data/scoreboard_runs.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="frames, for smoke tests")
    args = ap.parse_args()

    left = video.load_strip(ROOT / args.left)
    right = video.load_strip(ROOT / args.right)
    if left.fps != right.fps or left.frames != right.frames:
        raise SystemExit(
            f"strips disagree: {left.frames}@{left.fps}fps vs "
            f"{right.frames}@{right.fps}fps -- they must come from one pass"
        )

    total = args.limit or left.frames
    size = (total + args.workers - 1) // args.workers
    jobs = [(i, min(size, total - i)) for i in range(0, total, size)]
    print(f"{total} strip frames at {left.fps}fps, "
          f"{len(jobs)} chunks of {size} across {args.workers} workers")

    began = time.monotonic()
    results = []
    with Pool(
        args.workers, initializer=_init,
        initargs=(str(ROOT / args.left), str(ROOT / args.right),
                  str(ROOT / args.timeline)),
    ) as pool, Progress(
        "scan_scoreboard", total=len(jobs), label="scan scoreboard"
    ) as bar:
        for i, result in enumerate(pool.imap_unordered(_chunk, jobs), 1):
            results.append(result)
            bar.set(i)

    results.sort(key=lambda r: r["first"])
    runs = _stitch([run for r in results for run in r["runs"]])

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for run in runs:
            out.write(json.dumps(run) + "\n")

    elapsed = time.monotonic() - began
    read = sum(r["read"] for r in results)
    print(f"\n{total} strip frames, {read} LIVE frames read in {elapsed / 60:.1f} min")
    print(f"{sum(r['layout_errors'] for r in results)} frames failed the mirror check")
    print(f"{sum(r['unread'] for r in results)} runs began with an unreadable cell")
    print(f"{len(runs)} runs written to {out_path}")
    return 0


def _stitch(runs: list[dict]) -> list[dict]:
    """Rejoin runs that a chunk boundary split.

    Merges only same-slot, same-value runs that are adjacent in time, since a
    slot can legitimately return to an earlier value (e.g. a streak of 1, 0, 1).
    """
    by_slot: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        by_slot[(run["side"], run["slot"])].append(run)

    out: list[dict] = []
    for slot_runs in by_slot.values():
        slot_runs.sort(key=lambda r: r["start"])
        for run in slot_runs:
            if (out and out[-1]["side"] == run["side"]
                    and out[-1]["slot"] == run["slot"]
                    and out[-1]["value"] == run["value"]
                    and run["start"] - out[-1]["end"] <= BOUNDARY_GAP):
                out[-1]["end"] = run["end"]
                out[-1]["samples"] += run["samples"]
            else:
                out.append(run)
    return sorted(out, key=lambda r: (r["start"], r["side"], r["slot"]))


def _value(row: sb.PlayerRow, stat_label: str | None) -> dict:
    return {
        "number": row.number, "name": row.name,
        "kills": row.kills, "deaths": row.deaths,
        "streak": row.streak, "stat": row.stat, "stat_label": stat_label,
        "gutter": row.gutter, "spectated": row.spectated,
    }


def _close(open_runs: dict) -> list[dict]:
    closed = list(open_runs.values())
    open_runs.clear()
    return closed


if __name__ == "__main__":
    raise SystemExit(main())
