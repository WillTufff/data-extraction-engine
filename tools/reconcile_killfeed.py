"""Reconcile deduplicated killfeed events against the box-score K/D totals.

Six box scores across three modes give six independent per-player checks:
kills counted from killfeed events should equal the kills the box score
reports.

Deaths can run ahead of kills (suicides, explosives, falls have no
attacker row) and kills can run behind (rows lost over bright gameplay).
An excess in either direction means an event was invented.

    uv run python tools/reconcile_killfeed.py <video>      # first run, parses
    uv run python tools/reconcile_killfeed.py              # uses the cache
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.boxscore import (  # noqa: E402
    MODE_SCHEMAS,
    GridDetectionError,
    detect_grid,
    detect_mode,
    parse,
)
from cdlvision.events import DENSE_12HZ, SPARSE_1HZ, build_events  # noqa: E402
from cdlvision.glyphs import GlyphAtlas, TemplateSet, binarize  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def boxscore_segments(timeline: Path) -> list[tuple[float, float]]:
    return [
        (s["start"], s["end"])
        for s in json.loads(timeline.read_text())
        if s["state"] == "boxscore"
    ]


# Max template distance for an accepted box-score name match.
MAX_NAME_DISTANCE = 40


def read_boxscores(source: str, segments, atlas, modes, names) -> list[dict]:
    """Per-map K/D, taken from the middle of each box-score segment."""
    info = video.probe(source)
    out = []
    for i, (start, end) in enumerate(segments):
        ts = (start + end) / 2
        gray = cv2.cvtColor(video.frame_at(source, ts, info), cv2.COLOR_BGR2GRAY)
        try:
            grid = detect_grid(gray)
        except GridDetectionError as exc:
            print(f"  map {i + 1}: no grid at t={ts:.0f} ({exc})")
            continue
        mode = detect_mode(gray, modes).char
        result = parse(gray, atlas, mode=mode, grid=grid)

        schema = MODE_SCHEMAS[mode]
        nx0, nx1 = grid.columns[schema.index("name")]

        kd = {}
        for player, y in zip(result.players, grid.rows):
            cell = binarize(gray[y : y + grid.row_height, nx0:nx1])
            m = names.match(cell)
            if m.distance > MAX_NAME_DISTANCE:
                print(f"  map {i + 1}: unrecognised name "
                      f"(best {m.char!r} at distance {m.distance})")
                continue
            name = m.char
            raw = player.text("kills_deaths")
            if "/" not in raw:
                print(f"  map {i + 1}: unparsable K/D {raw!r} for {name!r}")
                continue
            k, d = raw.split("/", 1)
            if not (k.isdigit() and d.isdigit()):
                print(f"  map {i + 1}: non-numeric K/D {raw!r} for {name!r}")
                continue
            kd[name] = {"kills": int(k), "deaths": int(d)}
        # Segment bounds travel with the K/D so callers can bucket events by map.
        out.append({
            "map": i + 1, "t": ts, "mode": mode,
            "start": start, "end": end, "kd": kd,
        })
        print(f"  map {i + 1}: {mode} at t={ts:.0f}, {len(kd)} players")
    return out


def map_of(t: float, segments) -> int | None:
    """Which map an event at `t` belongs to: the first map whose box score
    has not happened yet."""
    for i, (start, _end) in enumerate(segments):
        if t < start:
            return i + 1
    return None


def choose_params(requested: str | None, records: list[dict]):
    """Pick the chaining preset; inferred from the cache's own sample rate
    when not given explicitly."""
    if requested == "sparse":
        return SPARSE_1HZ
    if requested == "dense":
        return DENSE_12HZ

    stamps = sorted({float(r["t"]) for r in records})
    if len(stamps) < 3:
        return SPARSE_1HZ
    step = min(b - a for a, b in zip(stamps, stamps[1:]))
    params = SPARSE_1HZ if step > 0.5 else DENSE_12HZ
    name = "sparse" if params is SPARSE_1HZ else "dense"
    print(f"{len(stamps)} distinct timestamps, min step {step:.3f}s "
          f"-> {name} chaining preset")
    return params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("--rows", default="data/killfeed_rows.jsonl")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--cache", default="data/boxscore_kd.json")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--params", choices=["sparse", "dense"], default=None,
                    help="chaining preset; inferred from the cache's sample "
                         "rate when omitted")
    args = ap.parse_args()

    segments = boxscore_segments(ROOT / args.timeline)
    cache = ROOT / args.cache
    if args.source:
        print(f"parsing {len(segments)} box scores")
        atlas = GlyphAtlas.load(ROOT / "src/cdlvision/config/atlas_boxscore.json")
        modes = TemplateSet.load(ROOT / "src/cdlvision/config/modes.json")
        bs_names = TemplateSet.load(
            ROOT / "src/cdlvision/config/boxscore_names.json"
        )
        maps = read_boxscores(args.source, segments, atlas, modes, bs_names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(maps, indent=2) + "\n")
        print(f"wrote {cache}\n")
    elif cache.exists():
        maps = json.loads(cache.read_text())
    else:
        print(f"no {cache}; pass the video once to build it", file=sys.stderr)
        return 1

    matcher = NameMatcher.load(ROOT / args.names)
    records = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines()
        if line.strip()
    ]
    params = choose_params(args.params, records)
    events = build_events(matcher, records, decode, **params.kwargs())
    resolved = [e for e in events if e.resolved]
    print(f"{len(records)} rows -> {len(events)} events, "
          f"{len(resolved)} with both names read "
          f"({len(resolved) / max(len(events), 1):.1%})")

    kills: dict[int, Counter] = {}
    deaths: dict[int, Counter] = {}
    for e in resolved:
        m = map_of(e.t, segments)
        if m is None:
            continue
        kills.setdefault(m, Counter())[e.attacker] += 1
        deaths.setdefault(m, Counter())[e.victim] += 1

    total_k = total_bk = total_d = total_bd = 0
    over = 0
    for entry in maps:
        m = entry["map"]
        print(f"\nmap {m} ({entry['mode']}):")
        print(f"  {'player':10s} {'kills':>13s} {'deaths':>13s}")
        for name, box in sorted(entry["kd"].items()):
            k = kills.get(m, Counter())[name]
            d = deaths.get(m, Counter())[name]
            total_k += k
            total_d += d
            total_bk += box["kills"]
            total_bd += box["deaths"]
            over += max(0, k - box["kills"]) + max(0, d - box["deaths"])
            print(f"  {name:10s} {k:5d}/{box['kills']:<4d} "
                  f"{k - box['kills']:+3d}  "
                  f"{d:5d}/{box['deaths']:<4d} {d - box['deaths']:+3d}")

    print(f"\ntotals: kills {total_k}/{total_bk} "
          f"({total_k / max(total_bk, 1):.1%} recovered), "
          f"deaths {total_d}/{total_bd} ({total_d / max(total_bd, 1):.1%})")
    print(f"excess over box score (should be 0): {over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
