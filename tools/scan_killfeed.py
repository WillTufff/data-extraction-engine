"""Cache every killfeed row bitmap over the LIVE segments, in one decode pass.

Stores the attacker, weapon, medal and victim bitmaps for each row, plus its
slot and width signature, so icon clustering, name matching, event dedup and
box-score reconciliation are all tunable against the cache without touching
the video again.

    uv run python tools/scan_killfeed.py <video> --out data/killfeed_rows.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.killfeed import SLOTS, row_structure  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def encode(mask: np.ndarray) -> dict:
    return {
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "b": base64.b64encode(np.packbits(mask).tobytes()).decode(),
    }


def decode(payload: dict) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(payload["b"]), dtype=np.uint8)
    bits = np.unpackbits(raw)[: payload["h"] * payload["w"]]
    return bits.reshape(payload["h"], payload["w"]).astype(bool)


def _init(source: str) -> None:
    global _SOURCE, _INFO
    _SOURCE = source
    _INFO = video.probe(source)


def _one(ts: float) -> list[dict]:
    try:
        bgr = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return []
    out = []
    for top in SLOTS:
        row = row_structure(bgr, top)
        if row is None:
            continue
        record = {
            "t": ts,
            "slot": top,
            "signature": list(row.signature),
            "attacker_bmp": encode(row.attacker.mask),
            "weapon_bmp": encode(row.weapon.mask),
            "victim_bmp": encode(row.victim.mask),
            "medal_bmp": encode(row.medal_token.mask) if row.medal else None,
            # x positions let a later pass check the left edge without the frame.
            "x": [row.attacker.x0, row.weapon.x0, row.victim.x0],
        }
        out.append(record)
    return out


def live_seconds(timeline: Path, step: float) -> list[float]:
    stamps: list[float] = []
    for seg in json.loads(timeline.read_text()):
        if seg["state"] != "live":
            continue
        t = seg["start"] + 1
        while t < seg["end"] - 1:
            stamps.append(float(t))
            t += step
    return stamps


def live_spans(timeline: Path) -> list[tuple[float, float]]:
    """LIVE segments as intervals, trimmed a second at each edge."""
    return [
        (seg["start"] + 1, seg["end"] - 1)
        for seg in json.loads(timeline.read_text())
        if seg["state"] == "live"
    ]


def scan_strip(strip_path: Path, timeline: Path) -> list[dict]:
    """Read every strip frame falling inside a LIVE segment."""
    strip = video.load_strip(strip_path)
    info = video.probe(strip.source)
    canvas = video.Canvas(strip.crop, info.width, info.height)
    spans = live_spans(timeline)

    records: list[dict] = []
    seen = kept = 0
    with Progress("scan_killfeed", total=strip.frames, label="killfeed scan") as bar:
        for ts, frame in video.stream_strip(strip):
            seen += 1
            bar.set(seen, note=f"{kept} LIVE frames, {len(records)} rows")
            if not any(lo <= ts < hi for lo, hi in spans):
                continue
            kept += 1
            full = canvas.place(frame)
            for top in SLOTS:
                row = row_structure(full, top)
                if row is None:
                    continue
                records.append(
                    {
                        "t": round(ts, 4),
                        "slot": top,
                        "signature": list(row.signature),
                        "attacker_bmp": encode(row.attacker.mask),
                        "weapon_bmp": encode(row.weapon.mask),
                        "victim_bmp": encode(row.victim.mask),
                        "medal_bmp": encode(row.medal_token.mask) if row.medal else None,
                        "x": [row.attacker.x0, row.weapon.x0, row.victim.x0],
                    }
                )
            if kept and kept % 5000 == 0:
                print(f"  {kept} LIVE frames, {len(records)} rows", flush=True)

    print(f"read {seen} strip frames, {kept} inside LIVE")
    return records


def scan_seek(source: str, timeline: Path, step: float, limit: int | None,
              workers: int) -> list[dict]:
    """The original path: one parallel seek per sampled second. Needs no strip."""
    stamps = live_seconds(timeline, step)
    if limit:
        stamps = stamps[:limit]
    print(f"scanning {len(stamps)} LIVE seconds", flush=True)

    records: list[dict] = []
    done = 0
    with Pool(workers, initializer=_init, initargs=(source,)) as pool:
        for got in pool.imap(_one, stamps, chunksize=8):
            records.extend(got)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(stamps)} frames, {len(records)} rows", flush=True)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None,
                    help="source VOD; omit when scanning a strip")
    ap.add_argument("--strip", default=None,
                    help="scan a cached strip instead of seeking the VOD")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/killfeed_rows.jsonl")
    args = ap.parse_args()

    timeline = ROOT / args.timeline

    if args.strip:
        records = scan_strip(Path(args.strip), timeline)
    elif args.source:
        records = scan_seek(args.source, timeline, args.step, args.limit, args.workers)
    else:
        ap.error("give a source VOD or --strip")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    medals = sum(1 for r in records if r["medal_bmp"])
    print(f"\n{len(records)} rows, {medals} carrying a medal "
          f"({100 * medals / max(len(records), 1):.1f}%)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
