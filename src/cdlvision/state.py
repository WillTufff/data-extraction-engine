"""HUD state classification: LIVE, REPLAY, BOXSCORE, SERIES, or OTHER.

Classification uses a small set of cheap structural probes over the opaque
overlay (name-row layout, a fixed-position replay tag, box-score grid,
series-title colour) rather than whole-frame analysis, since the layout is
independent of the gameplay behind it.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np

from .boxscore import GridDetectionError, _runs, detect_grid
from .glyphs import TemplateSet


class State(str, Enum):
    """States that change downstream behaviour.

    Deliberately coarse: distinguishing the analyst desk from a crowd shot
    changes nothing for any consumer, so both are OTHER.
    """

    LIVE = "live"
    REPLAY = "replay"
    BOXSCORE = "boxscore"
    SERIES = "series"
    OTHER = "other"


# Fixed ROIs for this broadcast layout at 1920x1080, as (x0, y0, x1, y1).
ROI_NAME_COLUMN = (95, 95, 300, 230)
ROI_REPLAY_TAG = (30, 260, 720, 380)
ROI_SERIES_TITLE = (780, 90, 1140, 200)

NAME_ROW_PITCH = (28, 36)
NAME_ROW_HEIGHT = (8, 26)
EXPECTED_NAME_ROWS = 4


@dataclass(frozen=True)
class Probes:
    """Raw probe outputs for one frame, kept for inspection and debugging."""

    name_rows: int
    name_pitches: tuple[int, ...]
    chrome: bool
    tag: str | None
    tag_distance: int | None
    series_orange: float
    boxscore_columns: int | None

    @property
    def replay_tag(self) -> bool:
        return self.tag is not None


def _ink_bands(gray: np.ndarray, roi: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Find text rows in a region, tolerant of inverted (highlighted) rows."""
    x0, y0, x1, y1 = roi
    region = gray[y0:y1, x0:x1].astype(np.int16)
    background = np.median(region, axis=1, keepdims=True)
    ink = np.abs(region - background) > 60
    proj = np.asarray(ink.sum(axis=1), dtype=np.int64)
    lit = [int(y) for y in np.flatnonzero(proj > 4)]
    lo, hi = NAME_ROW_HEIGHT
    return [(a, b) for a, b in _runs(lit, max_gap=3) if lo <= (b - a + 1) <= hi]


def _orange_fraction(bgr: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """Fraction of saturated broadcast-orange pixels in a region."""
    x0, y0, x1, y1 = roi
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float((((h >= 5) & (h <= 22)) & (s > 150) & (v > 120)).mean())


def tag_text_masks(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    min_area: int = 1500,
    min_fill: float = 0.6,
    text_threshold: int = 170,
    inset: int = 8,
) -> list[np.ndarray]:
    """Extract white text from every solid coloured block in a region.

    Colour is only a prefilter; an orange sponsor bug shares this band, so
    blocks must additionally be large and near-solid, and the caller still
    matches the text against known tags. All candidates are returned, not
    just the largest, since the sponsor bug can be wider than a genuine tag.
    The block is inset before reading text, to avoid bright content bleeding
    in at its border.
    """
    x0, y0, x1, y1 = roi
    crop = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    orange = (((h >= 5) & (h <= 22)) & (s > 150) & (v > 120)).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(orange, connectivity=8)
    out: list[np.ndarray] = []
    for i in range(1, count):
        bx, by, bw, bh, area = (int(x) for x in stats[i])
        if area < min_area or bw < 40 or bh < 12:
            continue
        if area / float(bw * bh) < min_fill:  # solid block, not scattered pixels
            continue
        iy0, iy1 = by + inset, by + bh - inset
        ix0, ix1 = bx + inset, bx + bw - inset
        if iy1 - iy0 < 8 or ix1 - ix0 < 20:
            continue
        block = cv2.cvtColor(crop[iy0:iy1, ix0:ix1], cv2.COLOR_BGR2GRAY)
        text = block > text_threshold
        if text.any():
            out.append(text)
    return out


# Acceptance is a fraction of template area, not a flat pixel count, since
# templates differ in size.
MAX_TAG_MISMATCH = 0.05


def probe(
    bgr: np.ndarray,
    gray: np.ndarray | None = None,
    tags: TemplateSet | None = None,
) -> Probes:
    """Run all structural probes over a frame."""
    if gray is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bands = _ink_bands(gray, ROI_NAME_COLUMN)
    tops = [a for a, _ in bands]
    pitches = tuple(b - a for a, b in zip(tops, tops[1:]))
    lo, hi = NAME_ROW_PITCH
    chrome = (
        len(bands) == EXPECTED_NAME_ROWS
        and bool(pitches)
        and all(lo <= p <= hi for p in pitches)
    )

    columns: int | None
    try:
        columns = detect_grid(gray).n_columns
    except GridDetectionError:
        columns = None

    tag: str | None = None
    tag_distance: int | None = None
    if tags is not None:
        for text in tag_text_masks(bgr, ROI_REPLAY_TAG):
            try:
                m = tags.match(text)
            except LookupError:
                continue  # block present but no template of compatible shape
            if tag_distance is None or m.distance < tag_distance:
                tag_distance = m.distance
                limit = MAX_TAG_MISMATCH * tags.templates[m.char].size
                tag = m.char if m.distance <= limit else None

    return Probes(
        name_rows=len(bands),
        name_pitches=pitches,
        chrome=chrome,
        tag=tag,
        tag_distance=tag_distance,
        series_orange=_orange_fraction(bgr, ROI_SERIES_TITLE),
        boxscore_columns=columns,
    )


def state_from_probes(p: Probes) -> State:
    """Decide a state from probe output.

    Split out from `classify` so the decision can be replayed against a
    cached scan. Order matters: the box-score grid is checked first as the
    strictest structure, then the replay tag (checked independently of live
    chrome, since the scoreboard can be absent or off-pitch around a replay
    transition while the tag is still on screen), then live chrome.
    """
    if p.boxscore_columns is not None:
        return State.BOXSCORE
    if p.replay_tag:
        return State.REPLAY
    if p.chrome:
        return State.LIVE
    if p.series_orange > 0.15:
        return State.SERIES
    return State.OTHER


def classify(
    bgr: np.ndarray,
    gray: np.ndarray | None = None,
    tags: TemplateSet | None = None,
) -> tuple[State, Probes]:
    """Classify a single frame."""
    p = probe(bgr, gray, tags)
    return state_from_probes(p), p


# --------------------------------------------------------------------------
# Temporal smoothing
#
# A state occupying a single sample is more likely a classification error
# than a real transition, so a minimum dwell time is enforced per state.
#
# Most states fail by flickering (one misread frame), fixed by absorbing any
# run shorter than its state's minimum dwell into a longer neighbour,
# repeatedly, shortest run first.
#
# REPLAY fails only by dropping out (the tag missed mid-replay), so it is
# exempt from absorption (dwell 0) and instead gets short gaps between two
# REPLAY runs bridged.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A maximal run of one state, in seconds.

    `end` is exclusive and extends one sampling interval past the last sample,
    so segments tile the timeline without gaps.
    """

    state: State
    start: float
    end: float
    n_samples: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class Smoothing:
    """Dwell-time parameters, in seconds."""

    min_dwell: Mapping[State, float] | None = None
    bridge: Mapping[State, float] | None = None

    def __post_init__(self) -> None:
        if self.min_dwell is None:
            object.__setattr__(self, "min_dwell", DEFAULT_MIN_DWELL)
        if self.bridge is None:
            object.__setattr__(self, "bridge", DEFAULT_BRIDGE)

    def dwell_for(self, state: State) -> float:
        """Dwell floor for `state`; absent means no floor.

        A partial `min_dwell` disables smoothing for the states it omits
        rather than falling back to the defaults.
        """
        return self.min_dwell.get(state, 0.0)


# Minimum dwell floors, in seconds, per state. REPLAY is 0 by design: a lone
# replay sample is kept rather than absorbed.
DEFAULT_MIN_DWELL: Mapping[State, float] = {
    State.LIVE: 3.0,
    State.REPLAY: 0.0,
    State.BOXSCORE: 4.0,
    State.SERIES: 8.0,
    State.OTHER: 3.0,
}

# Only REPLAY is bridged: short gaps within this window are filled in.
DEFAULT_BRIDGE: Mapping[State, float] = {State.REPLAY: 4.0}


@dataclass(frozen=True)
class _Run:
    state: State
    t0: float
    t1: float  # timestamp of the last sample in the run, not the end
    n: int


def _sample_interval(times: Sequence[float]) -> float:
    """Median spacing between samples, robust to occasional dropped frames."""
    if len(times) < 2:
        return 1.0
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    return statistics.median(gaps) if gaps else 1.0


def _compress(observations: Sequence[tuple[float, State]]) -> list[_Run]:
    runs: list[_Run] = []
    for t, s in observations:
        if runs and runs[-1].state is s:
            last = runs[-1]
            runs[-1] = replace(last, t1=t, n=last.n + 1)
        else:
            runs.append(_Run(s, t, t, 1))
    return runs


def _merge_adjacent(runs: Sequence[_Run]) -> list[_Run]:
    out: list[_Run] = []
    for r in runs:
        if out and out[-1].state is r.state:
            last = out[-1]
            out[-1] = replace(last, t1=r.t1, n=last.n + r.n)
        else:
            out.append(r)
    return out


def _bridge_gaps(runs: list[_Run], params: Smoothing, interval: float) -> list[_Run]:
    """Relabel short runs that sit between two runs of a bridged state."""
    for state, window in params.bridge.items():
        if window <= 0:
            continue
        changed = True
        while changed:
            changed = False
            for i in range(1, len(runs) - 1):
                run = runs[i]
                if run.state is state:
                    continue
                if runs[i - 1].state is not state or runs[i + 1].state is not state:
                    continue
                if run.t1 - run.t0 + interval > window:
                    continue
                runs[i] = replace(run, state=state)
                runs = _merge_adjacent(runs)
                changed = True
                break
    return runs


def _absorb_short(runs: list[_Run], params: Smoothing, interval: float) -> list[_Run]:
    """Repeatedly fold the shortest sub-dwell run into its longer neighbour."""
    while len(runs) > 1:
        offenders = [
            (r.t1 - r.t0 + interval, i)
            for i, r in enumerate(runs)
            if r.t1 - r.t0 + interval < params.dwell_for(r.state)
        ]
        if not offenders:
            break
        _, i = min(offenders)

        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        if left is None and right is None:
            break
        if left is None:
            winner = right
        elif right is None:
            winner = left
        else:
            # Ties go to the earlier neighbour, so the result does not depend
            # on which end of the timeline the scan started from.
            winner = left if (left.t1 - left.t0) >= (right.t1 - right.t0) else right

        runs[i] = replace(runs[i], state=winner.state)
        runs = _merge_adjacent(runs)
    return runs


def segment(
    observations: Iterable[tuple[float, State]],
    params: Smoothing | None = None,
) -> list[Segment]:
    """Turn raw per-frame classifications into smoothed state segments.

    `observations` are `(timestamp, state)` pairs; they are sorted by time
    here, so callers may pass them in any order.
    """
    params = params or Smoothing()
    obs = sorted(observations, key=lambda o: o[0])
    if not obs:
        return []

    interval = _sample_interval([t for t, _ in obs])
    runs = _compress(obs)
    runs = _bridge_gaps(runs, params, interval)
    runs = _absorb_short(runs, params, interval)

    return [
        Segment(state=r.state, start=r.t0, end=r.t1 + interval, n_samples=r.n)
        for r in runs
    ]


def label_at(segments: Sequence[Segment], t: float) -> State | None:
    """State covering timestamp `t`, or None if outside the scanned range."""
    for s in segments:
        if s.contains(t):
            return s.state
    return None
