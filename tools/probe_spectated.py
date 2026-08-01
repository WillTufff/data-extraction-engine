"""Measure the spectated panel's geometry from the VOD rather than assuming it.

Run this before changing any constant in `spectated.py`. Samples LIVE frames
from the state timeline and reports panel presence rate, name/stat row
positions, label column layout, and weapon-name geometry against what the
module claims.

Usage:
    uv run python tools/probe_spectated.py <video> [--samples 300]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import spectated as sp  # noqa: E402
from cdlvision import video  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Bands generous enough that a row outside the claimed geometry would be found
# rather than cropped away.
PROBE_BAR = (890, 960)
PROBE_BLOCK = (958, 1024)

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE, _INFO = source, video.probe(source)


def _spans(flags: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    out, start = [], None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((offset + start, offset + i - 1))
            start = None
    if start is not None:
        out.append((offset + start, offset + len(flags) - 1))
    return out


def _one(ts: float) -> dict | None:
    try:
        frame = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return None
    out = {"ts": ts, "flatness": sp.flatness(frame), "present": sp.present(frame)}
    if not out["present"]:
        return out

    bar = frame[PROBE_BAR[0]:PROBE_BAR[1], sp.NAME_X[0]:sp.NAME_X[1]]
    out["name_rows"] = _spans(sp.band_ink(bar).sum(axis=1) > 2, PROBE_BAR[0])

    block = frame[PROBE_BLOCK[0]:PROBE_BLOCK[1], sp.PANEL_X[0]:sp.PANEL_X[1]]
    out["block_rows"] = _spans(sp.band_ink(block).sum(axis=1) > 3, PROBE_BLOCK[0])

    labels = frame[sp.LABEL_Y[0]:sp.LABEL_Y[1], sp.PANEL_X[0]:sp.PANEL_X[1]]
    out["label_cols"] = _spans(sp.band_ink(labels).sum(axis=0) > 0, sp.PANEL_X[0])

    weapon = sp.weapon_mask(frame)
    rows = _spans(weapon.sum(axis=1) > 1, sp.WEAPON_Y[0])
    cols = _spans(weapon.sum(axis=0) > 0, sp.WEAPON_X[0])
    out["weapon_rows"] = rows
    out["weapon_right"] = cols[-1][1] if cols else None
    out["team"] = sp.team_of(frame)
    try:
        sp.check_layout(frame)
        out["layout"] = "ok"
    except sp.LayoutError as exc:
        out["layout"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spans = [
        (s["start"] + 2, s["end"] - 2)
        for s in json.loads((ROOT / args.timeline).read_text())
        if s["state"] == "live" and s["end"] - s["start"] > 6
    ]
    rng = random.Random(args.seed)
    stamps = sorted(rng.uniform(*rng.choice(spans)) for _ in range(args.samples))
    with Pool(8, initializer=_init, initargs=(args.video,)) as pool:
        rows = [r for r in pool.map(_one, stamps) if r]

    drawn = [r for r in rows if r["present"]]
    flat_present = sorted(r["flatness"] for r in drawn)
    flat_absent = sorted(r["flatness"] for r in rows if not r["present"])
    print(f"{len(drawn)}/{len(rows)} sampled LIVE frames draw a panel "
          f"({100 * len(drawn) / len(rows):.0f}%)")
    print(f"  flatness: present up to {flat_present[-1]:.1f}, "
          f"absent from {flat_absent[0]:.1f} "
          f"(module cuts at {sp.FLATNESS_CUT})")
    print(f"  layout check: {Counter(r['layout'] for r in drawn).most_common()}")
    print(f"  team from the bar: {Counter(r['team'] for r in drawn).most_common()}")

    _report_band("name row", [r["name_rows"] for r in drawn], sp.NAME_Y)
    _report_rows("stat block rows", [r["block_rows"] for r in drawn],
                 (sp.LABEL_Y, sp.VALUE_Y))

    columns = Counter(tuple(r["label_cols"]) for r in drawn)
    spans_, count = columns.most_common(1)[0]
    print(f"\nlabel columns: modal layout on {count}/{len(drawn)} frames")
    print(f"  {list(spans_)}")
    print(f"  module claims {sp.LABEL_SPANS}")

    heights = Counter(
        b - a + 1 for r in drawn for a, b in r["weapon_rows"] if b - a >= 8
    )
    right = Counter(r["weapon_right"] for r in drawn if r["weapon_right"])
    print(f"\nweapon name: heights {heights.most_common(6)}")
    print(f"  right edge {right.most_common(4)} (module reads to x={sp.WEAPON_X[1]})")
    return 0


def _report_band(label: str, observations: list[list], claimed: tuple[int, int]):
    tall = Counter(
        (a, b) for rows in observations for a, b in rows if b - a >= 12
    )
    print(f"\n{label}: {tall.most_common(5)}")
    print(f"  module reads {claimed}")


def _report_rows(label: str, observations: list[list], claimed):
    modal = Counter(
        tuple((a, b) for a, b in rows if b - a >= 12) for rows in observations
    )
    print(f"\n{label}: {modal.most_common(3)}")
    print(f"  module reads {claimed}")


if __name__ == "__main__":
    raise SystemExit(main())
