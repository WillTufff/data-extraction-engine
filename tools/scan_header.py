"""Read the centre header over the whole VOD, once, into a tunable cache.

Makes one pass over the cached header strip, chunked across processes, and
stores runs per field rather than per frame or per whole header -- fields
change at very different rates (clock every second, score a few times a
minute, pips once a map), so per-field run-length encoding avoids both
breaking a run on every clock tick and repeating the pips thousands of times.

    uv run python tools/scan_header.py --out data/header_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import header as hd  # noqa: E402
from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Two runs either side of a chunk boundary are the same run if they are this
# close. One strip frame at 6fps is 0.167s; anything larger is a real gap.
BOUNDARY_GAP = 0.2

FIELDS = (
    "best_of", "maps_left", "maps_right", "score_left", "score_right",
    "clock", "rotation", "mode", "substrip", "lives_left", "lives_right",
)


def live_spans(timeline: Path) -> list[tuple[float, float]]:
    return [
        (s["start"] + 1, s["end"] - 1)
        for s in json.loads(timeline.read_text())
        if s["state"] == "live" and s["end"] - s["start"] > 2
    ]


def _is_live(spans, ts: float) -> bool:
    return any(a <= ts <= b for a, b in spans)


_STATE: dict = {}


def _init(strip_path: str, timeline: str) -> None:
    strip = video.load_strip(strip_path)
    _STATE["strip"] = strip
    _STATE["spans"] = live_spans(Path(timeline))
    _STATE["reader"] = hd.Reader.load()
    _STATE["canvas"] = video.Canvas(strip.crop, 1920, 1080)


def _chunk(job: tuple[int, int]) -> dict:
    """Read one frame range, returning its per-field runs.

    Runs are closed at the chunk boundary and rejoined by `_stitch`, so the
    split leaves no trace in the output.
    """
    first, count = job
    strip, spans = _STATE["strip"], _STATE["spans"]
    reader, canvas = _STATE["reader"], _STATE["canvas"]

    runs: list[dict] = []
    open_runs: dict[str, dict] = {}
    read = layout_errors = 0
    unread: Counter = Counter()

    for ts, frame in video.stream_strip(strip, first, count):
        if not _is_live(spans, ts):
            runs += _close(open_runs)
            continue
        try:
            reading = reader.read(canvas.place(frame))
        except sb.LayoutError:
            # Unreadable header layout; close every run rather than attribute
            # this frame's pixels to the previous reading.
            layout_errors += 1
            runs += _close(open_runs)
            continue
        read += 1

        for field in FIELDS:
            value = getattr(reading, field)
            if value is None:
                unread[field] += 1
            run = open_runs.get(field)
            if run is not None and run["value"] == value:
                run["end"] = ts
                run["samples"] += 1
                continue
            if run is not None:
                runs.append(run)
            open_runs[field] = {
                "field": field, "value": value,
                "start": ts, "end": ts, "samples": 1,
            }
    runs += _close(open_runs)
    return {"first": first, "runs": runs, "read": read,
            "layout_errors": layout_errors, "unread": dict(unread)}


def _close(open_runs: dict) -> list[dict]:
    closed = list(open_runs.values())
    open_runs.clear()
    return closed


def _stitch(runs: list[dict]) -> list[dict]:
    """Rejoin runs that a chunk boundary split.

    Merges only same-field, same-value runs that are adjacent in time, since a
    field can return to an earlier value (e.g. the clock counting through the
    same time on the next map) without the runs being one continuous span.
    """
    by_field: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_field[run["field"]].append(run)

    out: list[dict] = []
    for field_runs in by_field.values():
        field_runs.sort(key=lambda r: r["start"])
        merged: list[dict] = []
        for run in field_runs:
            if (merged and merged[-1]["value"] == run["value"]
                    and run["start"] - merged[-1]["end"] <= BOUNDARY_GAP):
                merged[-1]["end"] = run["end"]
                merged[-1]["samples"] += run["samples"]
            else:
                merged.append(run)
        out += merged
    return sorted(out, key=lambda r: (r["start"], r["field"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", default="data/strip_scoreboard_header.mp4")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--out", default="data/header_runs.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="frames, for smoke tests")
    args = ap.parse_args()

    strip = video.load_strip(ROOT / args.strip)
    total = args.limit or strip.frames
    size = (total + args.workers - 1) // args.workers
    jobs = [(i, min(size, total - i)) for i in range(0, total, size)]
    print(f"{total} strip frames at {strip.fps}fps, "
          f"{len(jobs)} chunks of {size} across {args.workers} workers")

    began = time.monotonic()
    results = []
    with Pool(
        args.workers, initializer=_init,
        initargs=(str(ROOT / args.strip), str(ROOT / args.timeline)),
    ) as pool, Progress(
        "scan_header", total=len(jobs), label="scan header"
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
    print(f"{sum(r['layout_errors'] for r in results)} frames failed the pip check")
    unread: Counter = Counter()
    for r in results:
        unread.update(r["unread"])
    print("unread samples per field (None is often the correct reading):")
    for field in FIELDS:
        n = unread.get(field, 0)
        print(f"   {field:14s} {n:7d}  {100 * n / max(read, 1):5.1f}%")
    print(f"{len(runs)} runs written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
