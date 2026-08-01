"""Post-game box-score extraction.

Grid geometry is detected from the frame rather than hardcoded, with
invariant checks so a layout change fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .glyphs import GlyphAtlas, Match, TemplateSet, binarize

# Column schemas per mode: a shared combat prefix followed by six mode-specific
# objective columns. All three modes render 13 columns, so mode must come from
# the header title, not the column count.
COMBAT_PREFIX = [
    "player_index", "name",
    "kills_deaths", "assists", "non_traded_kills", "highest_streak", "damage",
]

MODE_SCHEMAS: dict[str, list[str]] = {
    "hardpoint": COMBAT_PREFIX + [
        "hill_time", "average_hill_time", "objective_kills", "contested_hill_time",
        "kills_per_hill", "damage_per_hill",
    ],
    "search_and_destroy": COMBAT_PREFIX + [
        "bombs_planted", "bombs_defused", "first_bloods", "first_deaths",
        "kills_per_round", "damage_per_round",
    ],
    "overload": COMBAT_PREFIX + [
        "overloads", "carrier_kills", "kills_as_carrier", "highest_multikill",
        "accuracy", "score",
    ],
}

# Fields that are not numeric and are handled positionally for now.
NON_NUMERIC = {"name"}


@dataclass(frozen=True)
class Grid:
    """Detected geometry of a box-score table."""

    rows: list[int]              # y of each player row's ink top
    row_height: int
    columns: list[tuple[int, int]]  # (x0, x1) half-open spans, left to right

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_columns(self) -> int:
        return len(self.columns)


class GridDetectionError(RuntimeError):
    """Raised when a frame does not present a well-formed box-score grid."""


def _runs(indices: list[int], max_gap: int) -> list[tuple[int, int]]:
    """Group sorted indices into inclusive runs separated by more than max_gap."""
    if not indices:
        return []
    out: list[tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i - prev > max_gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def detect_grid(
    gray: np.ndarray,
    *,
    threshold: int = 150,
    expected_rows: int = 8,
    row_height_range: tuple[int, int] = (18, 30),
    x_window: tuple[int, int] = (200, 1850),
    min_x_span: float = 0.4,
) -> Grid:
    """Locate player rows and field columns in a box-score frame.

    Player rows are found by horizontal projection, filtered by band height
    (rejects team-header blocks) and horizontal ink extent (rejects text that
    doesn't span the table, like the title card).
    """
    mask = binarize(gray, threshold)
    window = mask[:, x_window[0] : x_window[1]]

    rowsum = np.asarray(window.sum(axis=1), dtype=np.int64)
    lit = [int(y) for y in np.flatnonzero(rowsum > 8)]
    bands = _runs(lit, max_gap=3)

    lo, hi = row_height_range
    span_floor = min_x_span * gray.shape[1]

    def spans_table(y0: int, y1: int) -> bool:
        cols = np.flatnonzero(mask[y0 : y1 + 1, :].any(axis=0))
        return cols.size > 0 and (cols[-1] - cols[0] + 1) >= span_floor

    rows = [
        (y0, y1)
        for y0, y1 in bands
        if lo <= (y1 - y0 + 1) <= hi and spans_table(y0, y1)
    ]
    if len(rows) != expected_rows:
        raise GridDetectionError(
            f"expected {expected_rows} player rows, found {len(rows)} "
            f"(band heights: {[y1 - y0 + 1 for y0, y1 in bands]})"
        )

    heights = {y1 - y0 + 1 for y0, y1 in rows}
    if len(heights) != 1:
        raise GridDetectionError(f"player rows have inconsistent heights: {sorted(heights)}")
    row_height = heights.pop()

    # Rows arrive as two team blocks of four; pitch must be regular within a
    # block. The gap between blocks is not checked.
    tops = [y0 for y0, _ in rows]
    for block in (tops[:4], tops[4:]):
        pitches = {b - a for a, b in zip(block, block[1:])}
        if len(pitches) != 1:
            raise GridDetectionError(f"irregular row pitch within team block: {sorted(pitches)}")

    # Columns: vertical projection accumulated over player rows only, so header
    # text cannot merge adjacent fields.
    acc = np.zeros(mask.shape[1], dtype=np.int64)
    for y0 in tops:
        acc += np.asarray(mask[y0 : y0 + row_height, :].sum(axis=0), dtype=np.int64)
    lit_cols = [int(x) for x in np.flatnonzero(acc > 0)]
    columns = [(x0, x1 + 1) for x0, x1 in _runs(lit_cols, max_gap=12)]

    return Grid(rows=tops, row_height=row_height, columns=columns)


def header_lines(
    gray: np.ndarray,
    *,
    threshold: int = 150,
    x_window: tuple[int, int] = (215, 640),
    y_window: tuple[int, int] = (30, 220),
) -> list[tuple[int, int]]:
    """Find the text lines of the map/mode card in a box-score frame.

    The card carries the mode title, then the map name, then the game clock,
    each on its own line, so the caller can address them by index.
    """
    mask = binarize(gray, threshold)
    region = mask[y_window[0] : y_window[1], x_window[0] : x_window[1]]
    lit = [int(y) for y in np.flatnonzero(np.asarray(region.sum(axis=1), dtype=np.int64) > 2)]
    return [(y0 + y_window[0], y1 + y_window[0]) for y0, y1 in _runs(lit, max_gap=3)]


def detect_mode(
    gray: np.ndarray,
    modes: "TemplateSet",
    *,
    threshold: int = 150,
    x_window: tuple[int, int] = (215, 640),
) -> Match:
    """Identify the game mode from the header card's title line."""
    lines = header_lines(gray, threshold=threshold, x_window=x_window)
    if not lines:
        raise GridDetectionError("no header text found for mode detection")
    y0, y1 = lines[0]
    mask = binarize(gray[y0 : y1 + 1, x_window[0] : x_window[1]], threshold)
    return modes.match(mask)


@dataclass
class Field:
    """One parsed cell."""

    name: str
    text: str
    matches: list[Match] = field(default_factory=list)

    @property
    def min_margin(self) -> int:
        """Smallest best-vs-runner-up separation across the cell's glyphs."""
        return min((m.margin for m in self.matches), default=0)

    @property
    def max_distance(self) -> int:
        """Largest best-match Hamming distance across the cell's glyphs."""
        return max((m.distance for m in self.matches), default=0)


@dataclass
class PlayerRow:
    fields: dict[str, Field]

    def text(self, name: str) -> str:
        return self.fields[name].text

    @property
    def min_margin(self) -> int:
        return min((f.min_margin for f in self.fields.values()), default=0)


@dataclass
class BoxScore:
    mode: str
    grid: Grid
    players: list[PlayerRow]

    @property
    def min_margin(self) -> int:
        return min((p.min_margin for p in self.players), default=0)


def parse(
    gray: np.ndarray,
    atlas: GlyphAtlas,
    *,
    mode: str = "hardpoint",
    grid: Grid | None = None,
    threshold: int = 150,
) -> BoxScore:
    """Parse a box-score frame into structured per-player fields."""
    schema = MODE_SCHEMAS.get(mode)
    if schema is None:
        raise ValueError(f"no column schema for mode {mode!r}")

    grid = grid or detect_grid(gray, threshold=threshold)
    if grid.n_columns != len(schema):
        raise GridDetectionError(
            f"mode {mode!r} expects {len(schema)} columns, detected {grid.n_columns}"
        )

    players: list[PlayerRow] = []
    for y0 in grid.rows:
        fields: dict[str, Field] = {}
        for name, (x0, x1) in zip(schema, grid.columns):
            cell = gray[y0 : y0 + grid.row_height, x0:x1]
            if name in NON_NUMERIC:
                fields[name] = Field(name=name, text="", matches=[])
                continue
            text, matches = atlas.read(cell, threshold=threshold)
            fields[name] = Field(name=name, text=text, matches=matches)
        players.append(PlayerRow(fields=fields))

    return BoxScore(mode=mode, grid=grid, players=players)
