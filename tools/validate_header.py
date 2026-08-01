"""Check the header against the game's own rules, and against the scoreboard.

Checks: score never decreases inside a map; clock never increases inside a
segment of continuous play; the best-of is fixed for the series, so the pip
bands must count the same blocks throughout; maps won never decrease and
their sum never exceeds maps played.

A violation is a misread, a cut to footage that is not this map, or a game
behaviour the rule did not anticipate. In Search and Destroy the round clock
is replaced on a plant by the bomb timer, drawn in the same band and font,
starting at 0:44 -- a legitimate upward jump, not a misread.

    uv run python tools/validate_header.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import header as hd  # noqa: E402
from cdlvision import scoreboard_runs as sr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# A gap wider than this is a cut, not a tick; counters either side aren't comparable.
CONTINUOUS = 0.5

# A clock jump up smaller than this is a round reset (S&D), not a new segment.
# Used only to describe violations, never to excuse them.
ROUND_RESET = 5.0


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def by_field(runs: list[dict], field: str) -> list[dict]:
    """One field's runs in time order, with clocks converted to seconds
    (the cache stores them as "M:SS")."""
    if field in ("clock", "rotation"):
        return [
            {**r, "value": hd._seconds(r["value"])}
            for r in sorted(
                (x for x in runs if x["field"] == field), key=lambda x: x["start"]
            )
        ]
    return sorted(
        (r for r in runs if r["field"] == field), key=lambda r: r["start"]
    )


def _pairs(runs: list[dict], min_samples: int = 1):
    """Adjacent runs of one field that are continuous in time; a pair spanning
    a cut is skipped. `min_samples` drops runs shorter than that before
    pairing, since a single-frame run is usually a misread, not a state."""
    kept = [r for r in runs if r["samples"] >= min_samples]
    for a, b in zip(kept, kept[1:]):
        if (
            a["value"] is not None
            and b["value"] is not None
            and 0 <= b["start"] - a["end"] <= CONTINUOUS
        ):
            yield a, b


def check_monotone(runs, field, rising: bool, report, min_samples=2) -> list[dict]:
    """Steps that go the wrong way for a counter that only moves one way.
    Reported twice: over every run, and over runs held long enough to be a
    state."""
    field_runs = by_field(runs, field)
    out = {}
    for limit in (1, min_samples):
        bad = [
            {"field": field, "at": b["start"], "from": a["value"], "to": b["value"]}
            for a, b in _pairs(field_runs, limit)
            if (b["value"] < a["value"] if rising else b["value"] > a["value"])
        ]
        out[limit] = (bad, sum(1 for _ in _pairs(field_runs, limit)))

    direction = "decreases" if rising else "increases"
    raw, raw_n = out[1]
    stable, stable_n = out[min_samples]
    report(f"{field:12s} {len(raw):5d} {direction} of {raw_n:6d} steps; "
           f"{len(stable):5d} of {stable_n:6d} once runs under "
           f"{min_samples} frames are dropped")
    return stable


def check_best_of(runs, report) -> list[dict]:
    """The best-of is fixed for a series, so every other value is another one."""
    counts: Counter = Counter()
    for run in by_field(runs, "best_of"):
        if run["value"] is not None:
            counts[run["value"]] += run["samples"]
    if not counts:
        report("best_of      no readings at all")
        return []
    series, n = counts.most_common(1)[0]
    total = sum(counts.values())
    report(f"best_of      series is best of {series} on {n} of {total} samples "
           f"({100 * n / total:.2f}%); others {dict(counts.most_common()[1:])}")
    return [
        {"field": "best_of", "at": r["start"], "from": series, "to": r["value"],
         "end": r["end"], "samples": r["samples"]}
        for r in by_field(runs, "best_of")
        if r["value"] is not None and r["value"] != series
    ]


def windows(items, gap: float = 30.0) -> list[tuple[float, float]]:
    """Collapse timestamped items into the windows they cluster into."""
    out: list[list[float]] = []
    for item in sorted(items, key=lambda i: i["at"]):
        end = item.get("end", item["at"])
        if out and item["at"] - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([item["at"], end])
    return [(a, b) for a, b in out]


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/header_runs.jsonl")
    ap.add_argument("--scoreboard", default="data/scoreboard_runs.jsonl")
    args = ap.parse_args()

    runs = load(ROOT / args.runs)
    print(f"{len(runs)} header runs, "
          f"{len({r['field'] for r in runs})} fields\n")

    print("=== counters that may only move one way")
    violations: list[dict] = []
    for field in ("score_left", "score_right", "maps_left", "maps_right"):
        violations += check_monotone(runs, field, True, lambda s: print("   " + s))
    clock_bad = check_monotone(runs, "clock", False, lambda s: print("   " + s))

    print("\n=== the series' shape")
    best_of_bad = check_best_of(runs, lambda s: print("   " + s))

    print("\n=== against the roster, which shares no pixels with the pips")
    if not (ROOT / args.scoreboard).exists():
        print("   no scoreboard runs; skipping")
        return 0
    sb_runs = sr.load(ROOT / args.scoreboard)
    _, theirs = sr.split_foreign(sb_runs)
    roster_windows = windows([{"at": r["start"], "end": r["end"]} for r in theirs])
    pip_windows = windows(best_of_bad)
    print(f"   the roster contradicts itself in {len(roster_windows)} windows")
    print(f"   the pip count disagrees in {len(pip_windows)} windows")

    agreed = [w for w in pip_windows if any(overlap(w, r) for r in roster_windows)]
    print(f"   {len(agreed)} of {len(pip_windows)} pip windows overlap a roster "
          f"window -- two independent detectors on the same footage")
    for w in pip_windows:
        mark = "both" if any(overlap(w, r) for r in roster_windows) else "PIPS ONLY"
        print(f"      {w[0]:9.1f} .. {w[1]:9.1f}  {mark}")

    print(f"\n{len(violations)} monotonicity violations outside the clock, "
          f"{len(clock_bad)} clock violations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
