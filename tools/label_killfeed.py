"""Label killfeed rows with on-device OCR, as an offline oracle for the
matcher's ground truth.

Calibration scaffolding, not pipeline code: nothing in `src/cdlvision/`
imports it. The shipped killfeed reader stays deterministic template
matching; this exists to measure that matcher against something other than
manual review. Apple's Vision is used here because it only needs to bound
another method's error and is run once, offline, with output sanity-checked
by a person -- not because it would be acceptable in the runtime pipeline.

Each name token is rendered in a few variants (cropped, colour-normalised)
and OCR'd; results are pooled by string similarity to the closed player set
and Vision's own confidence (variant voting), then pooled again across
adjacent frames showing the same entry, matched by token-width signature
(temporal voting). Unresolved slots are excluded from the fixture rather
than guessed at.

Output is one record per row per frame: the label, an agreement score, and
the binarised name bitmaps, so the matcher can be rebuilt and re-measured
against this file without re-decoding the video.

Usage:
    uv run python tools/label_killfeed.py <video> [--step 1] [--out data/killfeed_labels.jsonl]
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import video  # noqa: E402
from cdlvision.killfeed import (  # noqa: E402
    MAX_TOKEN_GAP,
    SEARCH_X,
    TOKEN_GAP,
    _runs,
    colour_ink,
    value_ink,
)

ROOT = Path(__file__).resolve().parent.parent

# Fixed grid slots; every slot is probed rather than found by ink.
SLOTS = (569, 539, 509, 479, 449)

PLAYERS = ["ABUZAH", "DRAZAH", "SIMP", "HUKE", "SHOTZZY", "MERCULES", "DASHY", "04"]

WEAPON_WIDTH = (44, 52)
MEDAL_WIDTH = (11, 17)
NAME_WIDTH = (18, 95)

# (mode, scale, pad), best-first.
VARIANTS = (
    ("binarinv", 6, 10),
    ("vnorm", 6, 8),
    ("raw", 4, 8),
)

MIN_SIMILARITY = 0.7

# Which binarisation segments a row into tokens. value_ink is the default;
# --ink switches to colour_ink for comparison.
INKS = {"value": value_ink, "colour": colour_ink}
_INK = value_ink


def _vision():
    import Vision  # noqa: PLC0415
    from Foundation import NSData  # noqa: PLC0415

    return Vision, NSData


def ocr(img: np.ndarray, top_n: int = 3) -> list[tuple[str, float]]:
    Vision, NSData = _vision()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return []
    data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)  # accurate
    req.setUsesLanguageCorrection_(False)
    req.setCustomWords_(PLAYERS)
    req.setRecognitionLanguages_(["en-US"])
    handler.performRequests_error_([req], None)
    out: list[tuple[str, float]] = []
    for obs in req.results() or []:
        for cand in obs.topCandidates_(top_n) or []:
            out.append((cand.string(), float(cand.confidence())))
    return out


def score_candidates(cands: list[tuple[str, float]]) -> dict[str, float]:
    """Weight each player by string similarity and Vision's confidence."""
    scores: dict[str, float] = defaultdict(float)
    for text, conf in cands:
        for piece in text.replace("•", " ").replace("|", " ").split():
            token = "".join(ch for ch in piece.upper() if ch.isalnum())
            if len(token) < 2:
                continue
            for player in PLAYERS:
                ratio = difflib.SequenceMatcher(None, token, player).ratio()
                if ratio >= MIN_SIMILARITY:
                    scores[player] = max(scores[player], ratio * (0.5 + conf))
    return dict(scores)


def render(band: np.ndarray, mode: str, scale: int, pad: int) -> np.ndarray:
    if mode == "raw":
        img = band
    elif mode == "vchan":
        img = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[..., 2]
    elif mode == "vnorm":
        v = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[..., 2]
        img = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    elif mode == "binar":
        img = value_ink(band).astype(np.uint8) * 255
    elif mode == "colourinv":
        # Colour-keyed mask, inverted; survives bright gameplay where the
        # value-based mask floods.
        img = 255 - colour_ink(band).astype(np.uint8) * 255
    elif mode == "binarinv":
        # Dark text on light, colour-free (renders magenta and white names
        # identically).
        img = 255 - value_ink(band).astype(np.uint8) * 255
    else:
        raise ValueError(mode)
    if pad:
        value = 0 if img.ndim == 2 else (0, 0, 0)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def row_structure(bgr: np.ndarray, top: int):
    """Deterministic token spans for one slot, or None if it is not a row."""
    x0, x1 = SEARCH_X
    band = bgr[top - 1 : top + 16, x0:x1]
    mask = _INK(band)
    cols = np.asarray(mask.sum(axis=0)) > 0

    toks: list[tuple[int, int]] = []
    cur: list[int] | None = None
    for a, b in _runs(cols):
        if cur is None:
            cur = [a, b]
            continue
        gap = a - cur[1] - 1
        if gap < TOKEN_GAP:
            cur[1] = b
        elif gap <= MAX_TOKEN_GAP:
            toks.append((cur[0], cur[1]))
            cur = [a, b]
        else:
            break
    if cur is not None:
        toks.append((cur[0], cur[1]))

    if len(toks) < 3:
        return None
    widths = [b - a + 1 for a, b in toks]
    if not WEAPON_WIDTH[0] <= widths[1] <= WEAPON_WIDTH[1]:
        return None
    rest = list(range(2, len(toks)))
    medal = MEDAL_WIDTH[0] <= widths[rest[0]] <= MEDAL_WIDTH[1]
    if medal:
        rest = rest[1:]
    if not rest:
        return None
    ai, vi = 0, rest[0]
    if not (NAME_WIDTH[0] <= widths[ai] <= NAME_WIDTH[1]):
        return None
    if not (NAME_WIDTH[0] <= widths[vi] <= NAME_WIDTH[1]):
        return None
    return {
        "band": band,
        "mask": mask,
        "attacker": toks[ai],
        "victim": toks[vi],
        "medal": medal,
        "signature": (widths[ai], widths[1], widths[vi]),
    }


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


_SOURCE: str | None = None
_INFO: video.VideoInfo | None = None


def _init(source: str, ink: str = "value") -> None:
    global _SOURCE, _INFO, _INK
    _SOURCE = source
    _INFO = video.probe(source)
    _INK = INKS[ink]


def _one(ts: float):
    try:
        bgr = video.frame_at(_SOURCE, ts, _INFO)
    except Exception:
        return []
    rows = []
    for top in SLOTS:
        st = row_structure(bgr, top)
        if st is None:
            continue
        record = {"t": ts, "slot": top, "medal": st["medal"],
                  "signature": list(st["signature"])}
        for role in ("attacker", "victim"):
            a, b = st[role]
            crop = st["band"][:, a : b + 1]
            votes: dict[str, float] = defaultdict(float)
            for mode, scale, pad in VARIANTS:
                for player, weight in score_candidates(
                    ocr(render(crop, mode, scale, pad))
                ).items():
                    votes[player] += weight
            record[role] = dict(sorted(votes.items(), key=lambda kv: -kv[1])[:3])
            record[f"{role}_bmp"] = encode(
                st["mask"][:, a : b + 1][
                    :, : b - a + 1
                ]
            )
        rows.append(record)
    return rows


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


def temporal_vote(records: list[dict]) -> None:
    """Pool votes across adjacent frames showing the same entry, matched by
    token-width signature as the entry ages up the grid one slot at a time."""
    by_time: dict[float, list[dict]] = defaultdict(list)
    for r in records:
        by_time[r["t"]].append(r)

    for r in records:
        pooled = {role: Counter() for role in ("attacker", "victim")}
        for role in ("attacker", "victim"):
            for player, weight in r[role].items():
                pooled[role][player] += weight
        for dt in (-3, -2, -1, 1, 2, 3):
            for other in by_time.get(r["t"] + dt, []):
                if other["signature"] != r["signature"]:
                    continue
                # As an entry ages it rises; slot must move consistently.
                if dt < 0 and other["slot"] > r["slot"]:
                    continue
                if dt > 0 and other["slot"] < r["slot"]:
                    continue
                for role in ("attacker", "victim"):
                    for player, weight in other[role].items():
                        pooled[role][player] += 0.5 * weight
        for role in ("attacker", "victim"):
            if pooled[role]:
                top = pooled[role].most_common(2)
                total = sum(pooled[role].values())
                r[f"{role}_label"] = top[0][0]
                r[f"{role}_agree"] = round(top[0][1] / total, 3)
            else:
                r[f"{role}_label"] = None
                r[f"{role}_agree"] = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="data/killfeed_labels.jsonl")
    ap.add_argument("--ink", choices=sorted(INKS), default="value")
    args = ap.parse_args()

    stamps = live_seconds(ROOT / args.timeline, args.step)
    if args.limit:
        stamps = stamps[: args.limit]
    print(f"labelling {len(stamps)} LIVE seconds", flush=True)

    records: list[dict] = []
    done = 0
    with Pool(
        args.workers, initializer=_init, initargs=(args.source, args.ink)
    ) as pool:
        for got in pool.imap(_one, stamps, chunksize=4):
            records.extend(got)
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(stamps)} frames, {len(records)} rows",
                      flush=True)

    temporal_vote(records)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    resolved = sum(1 for r in records for role in ("attacker", "victim")
                   if r[f"{role}_label"])
    slots = 2 * len(records)
    print(f"\n{len(records)} rows, {slots} name slots")
    print(f"resolved: {resolved}/{slots} ({100 * resolved / max(slots,1):.1f}%)")
    counts = Counter(r[f"{role}_label"] for r in records
                     for role in ("attacker", "victim") if r[f"{role}_label"])
    print("label counts:", counts.most_common())
    strong = sum(1 for r in records for role in ("attacker", "victim")
                 if r[f"{role}_agree"] >= 0.8)
    print(f"agreement >= 0.8: {strong}/{slots} ({100 * strong / max(slots,1):.1f}%)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
