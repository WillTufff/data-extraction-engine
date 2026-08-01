"""Build the killfeed name matcher from the OCR-labelled bitmaps, and score it.

Labels come from `tools/label_killfeed.py`, an offline oracle; the matcher
built here compares bitmaps by Hamming distance.

Evaluation splits by time, not at random: templates are built from the first
half of the VOD and scored on the second, so no killfeed entry appears on
both sides.

    uv run python tools/build_killfeed_names.py
    uv run python tools/build_killfeed_names.py --labels data/killfeed_labels_colour.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.names import NameMatcher  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Rosters parsed from the box scores; needed for the cross-team constraint.
BOXSCORES = ("tests/fixtures/boxscore_t3940.json", "tests/fixtures/boxscore_t4920.json")

# Minimum observations of a (player, width) before it gets a template.
MIN_SUPPORT = 3


def decode(payload: dict) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(payload["b"]), dtype=np.uint8)
    bits = np.unpackbits(raw)[: payload["h"] * payload["w"]]
    return bits.reshape(payload["h"], payload["w"]).astype(bool)


def rosters() -> dict[str, str]:
    """Player -> team, cross-checked between the box scores."""
    teams: dict[str, str] = {}
    for rel in BOXSCORES:
        d = json.loads((ROOT / rel).read_text())
        names = [r["name"] for r in d["rows"]]
        if len(names) != 8:
            raise ValueError(f"{rel}: expected 8 players, got {len(names)}")
        mapping = {n: d["team_a"] for n in names[:4]}
        mapping.update({n: d["team_b"] for n in names[4:]})
        if teams and teams != mapping:
            raise ValueError(f"{rel}: roster disagrees with the other box score")
        teams = mapping
    return teams


def observations(path: Path) -> list[tuple[float, str, np.ndarray]]:
    """(timestamp, label, bitmap) for every resolved name slot."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for role in ("attacker", "victim"):
            label = r.get(f"{role}_label")
            if label is None:
                continue
            out.append((r["t"], label, decode(r[f"{role}_bmp"])))
    return out


def full_rows(path: Path) -> list[tuple[float, str, np.ndarray, str, np.ndarray]]:
    """Rows where the oracle resolved both names, for row-level scoring."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        la, lv = r.get("attacker_label"), r.get("victim_label")
        if la is None or lv is None:
            continue
        out.append(
            (r["t"], la, decode(r["attacker_bmp"]), lv, decode(r["victim_bmp"]))
        )
    return out


def build(obs: list[tuple[float, str, np.ndarray]], min_support: int) -> NameMatcher:
    """One averaged template per (player, width) that has enough support."""
    from cdlvision.names import average_masks

    groups: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for _, label, mask in obs:
        groups[(label, mask.shape[1])].append(mask)

    matcher = NameMatcher(teams=rosters())
    for (label, _width), masks in sorted(groups.items()):
        if len(masks) < min_support:
            continue
        matcher.templates.setdefault(label, []).append(average_masks(masks))
    return matcher


def evaluate_rows(matcher: NameMatcher, rows) -> dict:
    """Score both names per row, with and without the cross-team constraint."""
    out = {}
    for mode in ("plain", "cross_team"):
        correct = unknown = wrong = invalid = 0
        for _, la, ma, lv, mv in rows:
            pair = matcher.match_row(
                ma, mv, enforce_cross_team=(mode == "cross_team")
            )
            for got, truth in zip(pair, (la, lv)):
                if not got.resolved:
                    unknown += 1
                elif got.label == truth:
                    correct += 1
                else:
                    wrong += 1
            # A row whose two names land on one team is impossible.
            if matcher.same_team(*pair):
                invalid += 1
        n = max(2 * len(rows), 1)
        out[mode] = {
            "slots": 2 * len(rows),
            "correct": correct,
            "wrong": wrong,
            "unknown": unknown,
            "accuracy": correct / n,
            "same_team_rows": invalid,
        }
    return out


def evaluate(matcher: NameMatcher, obs) -> dict:
    """Top-1 accuracy over resolved slots, and how much is left unknown."""
    correct = unknown = wrong = 0
    confusion: Counter = Counter()
    for _, truth, mask in obs:
        m = matcher.match(mask)
        if not m.resolved:
            unknown += 1
        elif m.label == truth:
            correct += 1
        else:
            wrong += 1
            confusion[(truth, m.label)] += 1
    total = max(len(obs), 1)
    return {
        "n": len(obs),
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "accuracy": correct / total,
        "accuracy_of_resolved": correct / max(correct + wrong, 1),
        "confusion": confusion.most_common(8),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/killfeed_labels.jsonl")
    ap.add_argument("--out", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    args = ap.parse_args()

    obs = observations(ROOT / args.labels)
    if not obs:
        print("no resolved labels", file=sys.stderr)
        return 1
    times = sorted({t for t, _, _ in obs})
    midpoint = times[len(times) // 2]
    train = [o for o in obs if o[0] < midpoint]
    test = [o for o in obs if o[0] >= midpoint]
    print(f"{len(obs)} labelled slots; split at t={midpoint:.0f} "
          f"-> {len(train)} train / {len(test)} test")

    held = build(train, args.min_support)
    print(f"\nheld-out matcher: {held!r}")
    res = evaluate(held, test)
    print(f"  accuracy      {res['accuracy']:.3%} "
          f"({res['correct']}/{res['n']})")
    print(f"  of resolved   {res['accuracy_of_resolved']:.3%}")
    print(f"  unknown       {res['unknown']} ({res['unknown']/res['n']:.1%})")
    print(f"  wrong         {res['wrong']}")
    if res["confusion"]:
        print("  confusions    " + ", ".join(
            f"{a}->{b} x{n}" for (a, b), n in res["confusion"]))

    rows = [r for r in full_rows(ROOT / args.labels) if r[0] >= midpoint]
    print(f"\nrow-level, {len(rows)} held-out rows with both names known:")
    for mode, r in evaluate_rows(held, rows).items():
        print(f"  {mode:11s} acc {r['accuracy']:7.3%}  "
              f"wrong {r['wrong']:3d}  unknown {r['unknown']:4d}  "
              f"same-team rows {r['same_team_rows']}")

    full = build(obs, args.min_support)
    out = ROOT / args.out
    full.save(out)
    print(f"\nshipped matcher built on all {len(obs)} slots: {full!r}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
