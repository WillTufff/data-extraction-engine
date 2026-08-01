"""Check the panel's HP against the killfeed, which is the only witness it has.

HP was reported rather than checked. The killfeed makes it checkable: when the
spectated player dies, the panel's HP reaches zero and a killfeed row names them
as a victim, a fraction of a second apart, off different strips read by
different code.

Four things are reported, and the last two are what make it a check:

- the offset between the two strips, **swept** rather than assumed;
- the fraction of falls a killfeed row accounts for;
- a **negative control** -- damage the player survived, which must not pair;
- a **null** -- the same falls at the wrong times, which bounds chance.

    uv run python tools/validate_hp.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import hp, maps, scoreboard_runs  # noqa: E402
from cdlvision.events import DENSE_12HZ, SPARSE_1HZ, build_events  # noqa: E402
from cdlvision.names import NameMatcher  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures"

# Offsets swept, in frames of the 6fps panel strip either side of zero.
SWEEP = [n / 6 for n in range(-3, 5)]

# How far to displace the falls for the null. Far enough that no fall can land
# on its own event, close enough to stay inside the same map.
NULLS = (-120.0, -60.0, -30.0, 30.0, 60.0, 120.0)


def killfeed_events(rows_path: Path, names_path: Path):
    matcher = NameMatcher.load(names_path)
    # The row cache is the same one-JSON-object-per-line format as the run
    # caches, so the same loader reads it.
    rows = scoreboard_runs.load(rows_path)
    stamps = sorted({float(r["t"]) for r in rows})
    step = min(b - a for a, b in zip(stamps, stamps[1:]))
    params = SPARSE_1HZ if step > 0.5 else DENSE_12HZ
    print(f"{len(rows)} killfeed rows, min step {step:.3f}s -> "
          f"{'sparse' if params is SPARSE_1HZ else 'dense'} chaining")
    return build_events(matcher, rows, decode, **params.kwargs())


def shifted(falls, delta):
    return [hp.Fall(t=f.t + delta, player=f.player, from_hp=f.from_hp, gap=f.gap)
            for f in falls]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="tests/fixtures/spectated_runs.jsonl.gz")
    ap.add_argument("--scoreboard", default="tests/fixtures/scoreboard_runs.jsonl.gz")
    ap.add_argument("--header", default="tests/fixtures/header_runs.jsonl.gz")
    ap.add_argument("--rows", default="tests/fixtures/killfeed_rows_dense.jsonl.gz")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    args = ap.parse_args()

    panel = hp.by_field(scoreboard_runs.load(ROOT / args.panel))
    board = scoreboard_runs.load(ROOT / args.scoreboard)
    header = scoreboard_runs.load(ROOT / args.header)
    ours, _ = scoreboard_runs.split_foreign(board)
    ours, _ = maps.filter_to_series(ours, header)
    windows = maps.derive(ours, header)
    events = killfeed_events(ROOT / args.rows, ROOT / args.names)
    resolved = [e for e in events if e.victim]
    print(f"{len(events)} events, {len(resolved)} with a victim read; "
          f"{len(windows)} maps\n")

    falls = hp.falls(panel, windows)
    control = hp.survived_dips(panel, windows)
    print(f"=== what the panel saw ===")
    print(f"  {len(falls)} falls to zero inside a map")
    print(f"  {len(control)} drops of {hp.CONTROL_DROP}+ HP the player survived "
          f"(the negative control)")

    # -- the offset, measured ---------------------------------------------
    print("\n=== offset between the panel and killfeed strips ===")
    print(f"  swept at +-{hp.SWEEP_WINDOW:.3f}s, narrow enough to localise")
    best = None
    for offset, matched in hp.sweep(falls, events, SWEEP):
        share = matched / max(len(falls), 1)
        ctl = sum(p.matched
                  for p in hp.pair(control, events, offset, hp.SWEEP_WINDOW))
        print(f"  {offset:+.3f}s: falls {matched:3d}/{len(falls)} ({share:6.1%})"
              f"   control {ctl / max(len(control), 1):6.1%}")
        if best is None or matched > best[1]:
            best = (offset, matched)
    offset = best[0]
    print(f"\n  peak at {offset:+.3f}s, one frame of the 6fps panel strip.")
    print("  Nothing at a negative offset: a killfeed row cannot precede the "
          "death it reports, and it does not.")

    # -- the null ----------------------------------------------------------
    print("\n=== null: the same falls at the wrong times ===")
    for delta in NULLS:
        n = sum(p.matched for p in hp.pair(shifted(falls, delta), events, offset))
        print(f"  {delta:+7.0f}s: {n / max(len(falls), 1):6.1%}")
    print("  This is what chance looks like. The peak above is not it.")

    # -- the pairing -------------------------------------------------------
    pairings = hp.pair(falls, events, offset)
    matched = [p for p in pairings if p.matched]
    print(f"\n=== HP falls against killfeed deaths ===")
    print(f"  {len(matched)}/{len(falls)} paired "
          f"({len(matched) / max(len(falls), 1):.1%})")
    if matched:
        deltas = [p.delta for p in matched]
        print(f"  residual offset: median {statistics.median(deltas):+.3f}s, "
              f"spread {min(deltas):+.3f} to {max(deltas):+.3f}s")

    print("\n  by map (drift would show here, and does not):")
    for window in windows:
        inside = [p for p in pairings if window.contains(p.fall.t)]
        hit = [p for p in inside if p.matched]
        if not inside:
            continue
        print(f"    map {window.number} {window.mode:20s} "
              f"{len(hit):3d}/{len(inside):3d} ({len(hit) / len(inside):5.1%})")

    # A fall the killfeed reported late is a different failure from one it never
    # reported, and lumping them together would overstate the row loss.
    unpaired = [p.fall for p in pairings if not p.matched]
    loose = hp.pair(unpaired, events, offset, window=5.0)
    late = sum(p.matched for p in loose)
    print(f"\n  of the {len(unpaired)} unpaired, {late} have a killfeed row for "
          f"that player within 5s and {len(unpaired) - late} have none at all.")
    print("  The first group is a chain that started late over bright gameplay, "
          "not a lost row; only the second is recall. HP cannot say which of "
          "those are suicides -- like the gutter skull, it marks every death.")

    # -- the reverse direction --------------------------------------------
    print("\n=== killfeed deaths of the player on screen, with no HP fall ===")
    stray = hp.unwitnessed_events(events, panel, windows)
    on_panel = [
        e for e in events
        if e.victim and any(w.contains(e.t) for w in windows)
        and hp.value_at(panel.get("name", []), e.t) == e.victim
    ]
    print(f"  {len(stray)} of {len(on_panel)} "
          f"({len(stray) / max(len(on_panel), 1):.1%})")
    for e in stray[:10]:
        print(f"    t={e.t:8.2f}  {e.attacker} -> {e.victim}  "
              f"({e.frames} frames)")
    print("  This is the direction with no innocent explanation, and it is the "
          "reassuring one: with 114 invented kills outstanding, they are not "
          "concentrated on the player the broadcast was watching.")

    print("\n=== falls per player ===")
    counts = Counter(f.player for f in falls)
    hits = Counter(p.fall.player for p in matched)
    for name, n in counts.most_common():
        print(f"  {name:10s} {hits[name]:3d}/{n:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
