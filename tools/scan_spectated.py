"""Read the spectated panel over the whole VOD, once, into a tunable cache.

One pass over a cached strip, chunked across processes, storing run-length
records per field rather than a reading per frame.

The panel can be legitimately absent for several reasons (free camera, a
washed-out weapon name, an empty sixth column), so the summary separates two
denominators: of LIVE frames sampled, how many drew a panel; and of the
frames that drew one, how many read each field. A field unread against the
second denominator is a bug; against the first it may just be a camera.

    uv run python tools/scan_spectated.py --out data/spectated_runs.jsonl
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

from cdlvision import spectated as sp  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BOUNDARY_GAP = 0.2

FIELDS = ("name", "team", "hp", "kills", "deaths", "streak", "stat",
          "stat_label", "weapon")

_STATE: dict = {}


def live_spans(timeline: Path) -> list[tuple[float, float]]:
    return [
        (s["start"] + 1, s["end"] - 1)
        for s in json.loads(timeline.read_text())
        if s["state"] == "live" and s["end"] - s["start"] > 2
    ]


def _is_live(spans, ts: float) -> bool:
    return any(a <= ts <= b for a, b in spans)


def _init(strip_path: str, timeline: str) -> None:
    strip = video.load_strip(strip_path)
    _STATE["strip"] = strip
    _STATE["spans"] = live_spans(Path(timeline))
    _STATE["reader"] = sp.Reader.load()
    _STATE["canvas"] = video.Canvas(strip.crop, 1920, 1080)


def _chunk(job: tuple[int, int]) -> dict:
    first, count = job
    strip, spans = _STATE["strip"], _STATE["spans"]
    reader, canvas = _STATE["reader"], _STATE["canvas"]

    runs: list[dict] = []
    open_runs: dict[str, dict] = {}
    live = drawn = layout_errors = 0
    unread: Counter[str] = Counter()

    for ts, strip_frame in video.stream_strip(strip, first, count):
        if not _is_live(spans, ts):
            runs += _close(open_runs)
            continue
        live += 1
        frame = canvas.place(strip_frame)
        if not sp.present(frame):
            runs += _close(open_runs)
            continue
        try:
            panel = reader.read(frame)
        except sp.LayoutError:
            # Panel drawn but not where expected; close every run rather than
            # attribute these pixels to the previous reading.
            layout_errors += 1
            runs += _close(open_runs)
            continue
        if panel is None:
            runs += _close(open_runs)
            continue
        drawn += 1

        for field, value in panel.fields.items():
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
    return {"first": first, "runs": runs, "live": live, "drawn": drawn,
            "layout_errors": layout_errors, "unread": dict(unread)}


def _close(open_runs: dict) -> list[dict]:
    out = list(open_runs.values())
    open_runs.clear()
    return out


def _stitch(runs: list[dict]) -> list[dict]:
    """Rejoin runs split by a chunk boundary."""
    by_field: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_field[run["field"]].append(run)
    out: list[dict] = []
    for field_runs in by_field.values():
        field_runs.sort(key=lambda r: r["start"])
        merged: list[dict] = []
        for run in field_runs:
            last = merged[-1] if merged else None
            if (last is not None and last["value"] == run["value"]
                    and run["start"] - last["end"] <= BOUNDARY_GAP):
                last["end"] = run["end"]
                last["samples"] += run["samples"]
                continue
            merged.append(dict(run))
        out += merged
    out.sort(key=lambda r: (r["field"], r["start"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", default="data/strip_spectated.mp4")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--out", default="data/spectated_runs.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="frames, for smoke tests")
    args = ap.parse_args()

    strip = video.load_strip(ROOT / args.strip)
    total = min(strip.frames, args.limit or strip.frames)
    size = max(1, total // (args.workers * 4))
    jobs = [(first, min(size, total - first)) for first in range(0, total, size)]
    print(f"{total} strip frames at {strip.fps}fps, {len(jobs)} chunks "
          f"across {args.workers} workers")

    began = time.monotonic()
    results = []
    with Progress("scan_spectated", total=total, label="scan spectated") as bar:
        done = 0
        with Pool(args.workers, initializer=_init,
                  initargs=(str(ROOT / args.strip), str(ROOT / args.timeline))) as pool:
            for result in pool.imap_unordered(_chunk, jobs):
                results.append(result)
                done += next(c for f, c in jobs if f == result["first"])
                bar.set(done)
    elapsed = time.monotonic() - began

    runs = _stitch([r for result in results for r in result["runs"]])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in runs))

    live = sum(r["live"] for r in results)
    drawn = sum(r["drawn"] for r in results)
    errors = sum(r["layout_errors"] for r in results)
    unread: Counter[str] = Counter()
    for result in results:
        unread.update(result["unread"])

    print(f"\n{len(runs)} runs -> {args.out} in {elapsed / 60:.1f} min")
    print(f"{live} LIVE frames, {drawn} drew a panel "
          f"({100 * drawn / max(live, 1):.1f}%), {errors} failed the layout check")
    print("\nper field, of the frames that drew a panel:")
    for field in FIELDS:
        missing = unread.get(field, 0)
        print(f"  {field:11} {drawn - missing:6d} read "
              f"({100 * (drawn - missing) / max(drawn, 1):5.1f}%), {missing} unread")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
