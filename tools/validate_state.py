"""Validate the HUD state classifier over a full VOD.

Reads the cached raw scan from `tools/scan_states.py`, applies temporal
smoothing, and reports: per-state precision/recall against hand-labelled
frames, reconciliation of the smoothed timeline against the known box-score
and series-overview timestamps, and structural probes for internally
contradictory frames or implausible segment durations.

Usage:
    uv run python tools/validate_state.py [--raw data/states_raw.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.state import (  # noqa: E402
    Probes,
    Segment,
    Smoothing,
    State,
    label_at,
    segment,
    state_from_probes,
)

ROOT = Path(__file__).resolve().parent.parent

# Post-map box scores and the series overview, from the box-score parser.
BOXSCORE_ANCHORS = (760.0, 2800.0, 3940.0, 4920.0, 7350.0, 8230.0)
SERIES_ANCHOR = 4980.0


def load_raw(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["t"])
    return rows


def rederive(row: dict) -> State:
    """Re-run the state decision against a cached scan's probe output."""
    return state_from_probes(
        Probes(
            name_rows=row["name_rows"],
            name_pitches=tuple(row["name_pitches"]),
            chrome=row["chrome"],
            tag=row["tag"],
            tag_distance=row["tag_distance"],
            series_orange=row["series_orange"],
            boxscore_columns=row["boxscore_columns"],
        )
    )


def prf(
    labels: list[tuple[float, State]],
    predict,
) -> dict[State, dict[str, float]]:
    """Per-state precision/recall over labelled timestamps."""
    tp: Counter[State] = Counter()
    fp: Counter[State] = Counter()
    fn: Counter[State] = Counter()
    support: Counter[State] = Counter()

    for t, truth in labels:
        support[truth] += 1
        pred = predict(t)
        if pred is truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            if pred is not None:
                fp[pred] += 1

    out: dict[State, dict[str, float]] = {}
    for s in State:
        p_den = tp[s] + fp[s]
        r_den = tp[s] + fn[s]
        out[s] = {
            "support": support[s],
            "tp": tp[s],
            "fp": fp[s],
            "fn": fn[s],
            "precision": tp[s] / p_den if p_den else float("nan"),
            "recall": tp[s] / r_den if r_den else float("nan"),
        }
    return out


def print_prf(title: str, table: dict[State, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(f"  {'state':9s} {'support':>7s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
          f"{'prec':>7s} {'recall':>7s}")
    for s in State:
        r = table[s]
        if not r["support"] and not r["fp"]:
            continue
        prec = "     --" if r["precision"] != r["precision"] else f"{r['precision']:7.3f}"
        rec = "     --" if r["recall"] != r["recall"] else f"{r['recall']:7.3f}"
        print(f"  {s.value:9s} {int(r['support']):7d} {int(r['tp']):4d} "
              f"{int(r['fp']):4d} {int(r['fn']):4d} {prec} {rec}")


def reconcile(segments: list[Segment]) -> bool:
    """Check the timeline against the independently known map structure."""
    print("\nbox-score anchor reconciliation")
    ok = True
    for t in BOXSCORE_ANCHORS:
        state = label_at(segments, t)
        hit = state is State.BOXSCORE
        ok &= hit
        seg = next((s for s in segments if s.contains(t)), None)
        span = f"[{seg.start:.0f}, {seg.end:.0f}) {seg.duration:.0f}s" if seg else "-"
        print(f"  t={t:7.0f}  {'PASS' if hit else 'FAIL'}  "
              f"{state.value if state else 'none':9s} {span}")

    state = label_at(segments, SERIES_ANCHOR)
    hit = state is State.SERIES
    ok &= hit
    print(f"  t={SERIES_ANCHOR:7.0f}  {'PASS' if hit else 'FAIL'}  "
          f"{state.value if state else 'none':9s} (series overview)")

    box = [s for s in segments if s.state is State.BOXSCORE]
    print(f"\n  {len(box)} BOXSCORE segments; {len(BOXSCORE_ANCHORS)} maps expected")
    for s in box:
        anchored = any(s.contains(t) for t in BOXSCORE_ANCHORS)
        print(f"    [{s.start:7.0f}, {s.end:7.0f})  {s.duration:6.0f}s  "
              f"{'anchored' if anchored else 'UNANCHORED'}")

    # Each map should be one contiguous stretch of play between box scores.
    print("\n  inter-anchor spans (map boundaries)")
    prev = 0.0
    for i, t in enumerate(BOXSCORE_ANCHORS, 1):
        print(f"    map {i}: {prev:7.0f} -> {t:7.0f}  ({t - prev:6.0f}s)")
        prev = t
    return ok


def run_lengths(raw_obs: list[tuple[float, State]], interval: float) -> None:
    """Distribution of unsmoothed run lengths, per state; what a dwell floor
    is chosen against."""
    runs: list[tuple[State, int]] = []
    for _, s in raw_obs:
        if runs and runs[-1][0] is s:
            runs[-1] = (s, runs[-1][1] + 1)
        else:
            runs.append((s, 1))

    print("\nraw run-length distribution (samples), per state")
    print(f"  {'state':9s} {'runs':>5s} {'1':>4s} {'2':>4s} {'3':>4s} "
          f"{'4-9':>5s} {'10+':>5s} {'max':>6s}   dwell floor")
    for st in State:
        lens = sorted(n for s, n in runs if s is st)
        if not lens:
            continue
        bucket = Counter(min(n, 10) for n in lens)
        four_nine = sum(v for k, v in bucket.items() if 4 <= k <= 9)
        print(f"  {st.value:9s} {len(lens):5d} {bucket[1]:4d} {bucket[2]:4d} "
              f"{bucket[3]:4d} {four_nine:5d} {bucket[10]:5d} "
              f"{max(lens) * interval:5.0f}s   "
              f"{Smoothing().dwell_for(st):.0f}s")


def structural_probes(rows: list[dict], segments: list[Segment]) -> None:
    """Look for the failure mode the labelled set is blind to."""
    print("\nstructural probes (these do not depend on hand labels)")

    tagged = [r for r in rows if r["tag"]]
    tag_no_chrome = [r for r in tagged if not r["chrome"]]
    tag_and_grid = [r for r in tagged if r["boxscore_columns"] is not None]
    print(f"  frames with a matched tag:              {len(tagged)}")
    print(f"    ... but no live chrome (-> OTHER):    {len(tag_no_chrome)}"
          f"  {[int(r['t']) for r in tag_no_chrome[:12]]}")
    print(f"    ... but also a box-score grid:        {len(tag_and_grid)}"
          f"  {[int(r['t']) for r in tag_and_grid[:12]]}")

    orange = [r for r in rows if r["series_orange"] > 0.15 and r["chrome"]]
    print(f"  series-orange over live chrome:         {len(orange)}")

    near = [r for r in rows if r["name_rows"] in (3, 5)]
    print(f"  name column with 3 or 5 rows:           {len(near)}"
          f"   (4 expected; near-misses hint at occlusion)")

    print("\n  state occupancy and longest segments")
    total = sum(s.duration for s in segments)
    for st in State:
        segs = [s for s in segments if s.state is st]
        dur = sum(s.duration for s in segs)
        share = 100 * dur / total if total else 0
        longest = sorted(segs, key=lambda s: -s.duration)[:3]
        tops = ", ".join(f"{s.start:.0f}+{s.duration:.0f}s" for s in longest)
        print(f"    {st.value:9s} {len(segs):4d} segs {dur:7.0f}s "
              f"{share:5.1f}%  longest: {tops or '-'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/states_raw.jsonl")
    ap.add_argument("--labels", default="tests/fixtures/state_labels.json")
    ap.add_argument("--out", default="data/state_timeline.json")
    ap.add_argument("--min-report", type=float, default=20.0,
                    help="print segments at least this long")
    args = ap.parse_args()

    rows = load_raw(ROOT / args.raw)
    if not rows:
        print("no observations; run tools/scan_states.py first")
        return 1

    span = rows[-1]["t"] - rows[0]["t"]
    print(f"{len(rows)} observations spanning {rows[0]['t']:.0f}s - "
          f"{rows[-1]['t']:.0f}s ({span / 60:.1f} min)")

    cached = [(r["t"], State(r["state"])) for r in rows]
    raw_obs = [(r["t"], rederive(r)) for r in rows]
    raw_counts = Counter(s for _, s in raw_obs)

    drift = [(t, a, b) for (t, a), (_, b) in zip(cached, raw_obs) if a is not b]
    if drift:
        moved = Counter((a.value, b.value) for _, a, b in drift)
        print(f"\ndecision changed since the scan ran: {len(drift)} samples "
              f"({100 * len(drift) / len(rows):.2f}%)")
        for (a, b), n in moved.most_common():
            print(f"  {a:9s} -> {b:9s} {n:5d}")

    params = Smoothing()
    segments = segment(raw_obs, params)

    smoothed_counts: Counter[State] = Counter()
    changed = 0
    for t, s in raw_obs:
        got = label_at(segments, t)
        if got is not None:
            smoothed_counts[got] += 1
            if got is not s:
                changed += 1

    print(f"\nper-sample counts (raw -> smoothed), {changed} samples relabelled "
          f"({100 * changed / len(raw_obs):.2f}%)")
    for s in State:
        print(f"  {s.value:9s} {raw_counts[s]:5d} -> {smoothed_counts[s]:5d}")

    raw_runs = sum(1 for a, b in zip(raw_obs, raw_obs[1:]) if a[1] is not b[1]) + 1
    print(f"\nruns: {raw_runs} raw -> {len(segments)} smoothed segments "
          f"({100 * (1 - len(segments) / raw_runs):.1f}% fragmentation removed)")

    label_path = ROOT / args.labels
    if label_path.exists():
        payload = json.loads(label_path.read_text())
        labels = [(float(x["t"]), State(x["state"])) for x in payload["labels"]]
        # A label outside the scanned range is unmeasured, not a miss.
        lo, hi = rows[0]["t"], rows[-1]["t"]
        skipped = [t for t, _ in labels if not lo <= t <= hi]
        labels = [(t, s) for t, s in labels if lo <= t <= hi]
        if skipped:
            print(f"\n{len(skipped)} labels outside the scanned range, not scored: "
                  f"{[int(t) for t in skipped]}")
        by_t = dict(raw_obs)
        print_prf("per-state accuracy, RAW (provisional recall -- see fixture)",
                  prf(labels, lambda t: by_t.get(t)))
        print_prf("per-state accuracy, SMOOTHED",
                  prf(labels, lambda t: label_at(segments, t)))

        for t, truth in labels:
            got = label_at(segments, t)
            if got is not truth:
                print(f"    MISS t={t:.0f}: labelled {truth.value}, got "
                      f"{got.value if got else 'none'}")

    ok = reconcile(segments)
    run_lengths(raw_obs, rows[1]["t"] - rows[0]["t"] if len(rows) > 1 else 1.0)
    structural_probes(rows, segments)

    print(f"\nsegments at least {args.min_report:.0f}s")
    for s in segments:
        if s.duration >= args.min_report:
            print(f"  [{s.start:7.0f}, {s.end:7.0f})  {s.duration:6.0f}s  "
                  f"{s.state.value}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            [
                {
                    "state": s.state.value,
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration,
                    "n_samples": s.n_samples,
                }
                for s in segments
            ],
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote timeline to {out}")

    print("\nRECONCILIATION: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
