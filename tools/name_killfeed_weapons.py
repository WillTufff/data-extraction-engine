"""Name the killfeed's weapon icon clusters using the spectated panel as the
oracle.

When a killfeed row's attacker is the player currently being spectated, the
row's icon and the panel's weapon name are the same gun. Pairings are
collected across the VOD and each icon cluster is named by majority vote,
with runner-ups printed. A cluster contaminated by more than one weapon
(same silhouette, different guns) is named for all of them rather than
picking a winner.

    uv run python tools/name_killfeed_weapons.py
    uv run python tools/name_killfeed_weapons.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision.glyphs import TemplateSet, tight_crop  # noqa: E402
from cdlvision.names import NameMatcher, normalised_distance  # noqa: E402
from scan_killfeed import decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Max time gap between a killfeed row and the panel reading paired with it.
MATCH_WINDOW = 0.25

# Max template distance for an icon to count as that weapon cluster.
MAX_ICON_DISTANCE = 0.28

# A runner-up weapon counts as sharing the cluster's silhouette (rather than
# being a mismatch) if its median distance is this close to the leader's and
# it has at least MIN_CONTAMINANT pairings.
INDISTINGUISHABLE = 0.02
MIN_CONTAMINANT = 20


def panel_at(runs: list[dict], t: float):
    for run in runs:
        if run["start"] - MATCH_WINDOW <= t <= run["end"] + MATCH_WINDOW:
            return run["value"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="data/killfeed_rows_dense.jsonl")
    ap.add_argument("--panel", default="data/spectated_runs.jsonl")
    ap.add_argument("--icons", default="src/cdlvision/config/killfeed_weapons.json")
    ap.add_argument("--names", default="src/cdlvision/config/killfeed_names.json")
    ap.add_argument("--write", action="store_true",
                    help="rename the icon templates in place")
    args = ap.parse_args()

    icons = TemplateSet.load(ROOT / args.icons)
    matcher = NameMatcher.load(ROOT / args.names)
    panel: dict[str, list[dict]] = defaultdict(list)
    for line in (ROOT / args.panel).read_text().splitlines():
        if line.strip():
            run = json.loads(line)
            panel[run["field"]].append(run)
    for runs in panel.values():
        runs.sort(key=lambda r: r["start"])

    records = [
        json.loads(line)
        for line in (ROOT / args.rows).read_text().splitlines() if line.strip()
    ]
    print(f"{len(records)} killfeed rows, {len(icons.templates)} icon clusters, "
          f"{sum(len(v) for v in panel.values())} panel runs\n")

    votes: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rows_seen = spectated_rows = 0
    for record in records:
        weapon = record.get("weapon_bmp")
        if not weapon:
            continue
        icon, distance = _classify(icons, tight_crop(decode(weapon)))
        if icon is None:
            continue
        rows_seen += 1
        t = record["t"]
        name = panel_at(panel["name"], t)
        gun = panel_at(panel["weapon"], t)
        if name is None or gun is None:
            continue
        attacker = matcher.match(decode(record["attacker_bmp"])).label
        if attacker is None or attacker != name:
            continue
        spectated_rows += 1
        votes[icon][gun].append(distance)

    print(f"{rows_seen} rows matched an icon cluster, {spectated_rows} of them "
          f"with the attacker being the player on screen\n")
    # Weapons that lead a cluster of their own are separable; elsewhere they
    # count as mis-pairings rather than merged silhouettes.
    leaders = {
        max(guns.items(), key=lambda kv: len(kv[1]))[0] for guns in votes.values()
    }

    print("=== icon cluster -> weapon name ===")
    naming: dict[str, str] = {}
    for icon in sorted(votes):
        ranked = sorted(votes[icon].items(), key=lambda kv: -len(kv[1]))
        total = sum(len(v) for _, v in ranked)
        winner, winning = ranked[0]
        best = _median(winning)
        holds = [winner]
        for gun, distances in ranked[1:]:
            share = 100 * len(distances) / total
            merged = (abs(_median(distances) - best) <= INDISTINGUISHABLE
                      and len(distances) >= MIN_CONTAMINANT
                      and gun not in leaders)
            why = ("same silhouette" if merged
                   else "separable elsewhere" if gun in leaders
                   else "outlier")
            print(f"  {icon:12}    {gun!r}: {len(distances)} pairings "
                  f"({share:.1f}%), median distance {_median(distances):.3f} "
                  f"vs {best:.3f} -- {why}")
            if merged:
                holds.append(gun)
        label = " / ".join(sorted(holds))
        print(f"  {icon:12} -> {label!r} on {len(winning)}/{total} pairings "
              f"({100 * len(winning) / total:.1f}% for the leader)")
        naming[icon] = label
    unnamed = sorted(set(icons.templates) - set(naming))
    if unnamed:
        print(f"  no pairing for {unnamed} -- these never fired while their "
              f"user was on screen")

    if args.write and naming:
        payload = json.loads((ROOT / args.icons).read_text())
        payload["templates"] = {
            naming.get(k, k): v for k, v in payload["templates"].items()
        }
        payload["note"] = ("named from the spectated panel by "
                           "tools/name_killfeed_weapons.py")
        (ROOT / args.icons).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\nwrote {args.icons}")
    return 0


def _classify(icons: TemplateSet, mask) -> tuple[str | None, float]:
    if mask.size == 0:
        return None, 1.0
    distance, label = min(
        (normalised_distance(mask, t), k) for k, t in icons.templates.items()
    )
    return (label, distance) if distance <= MAX_ICON_DISTANCE else (None, distance)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2)


if __name__ == "__main__":
    raise SystemExit(main())
