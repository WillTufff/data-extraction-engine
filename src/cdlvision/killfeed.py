"""Killfeed geometry, binarisation and row segmentation.

Rows sit on a bottom-anchored fixed grid, newest row lowest. Within a row the
layout flows left to right rather than columnar, so tokens are found by gaps,
not fixed offsets. Slots are probed and accepted on token structure rather
than found by ink, since ink search misses rows over bright gameplay.

Two binarisations are used: `value_ink` (HSV value threshold) for reading a
row, since it renders white and magenta text identically; `colour_ink` (flat
white/magenta filter) for rendering rows for the offline OCR oracle, where it
gives better contrast inside an already-known row band.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Search window, as (x0, x1). Text starts at x = 60.
SEARCH_X = (50, 430)

ROW_HEIGHT = (11, 17)

# Bottom-anchored: index 0 is the newest row.
GRID_BOTTOM = 569
GRID_PITCH = 30
MAX_ROWS = 5

# Every slot is probed; none is inferred from a neighbour.
SLOTS = tuple(GRID_BOTTOM - i * GRID_PITCH for i in range(MAX_ROWS))

INK_THRESHOLD = 120  # on the HSV value channel


def strip_crop(margin: int = 2):
    """The smallest even crop containing every slot's search window.

    Derived from the grid rather than hardcoded. `segment_row` reads
    `top - 1` upward, so the top edge is taken from the highest slot's band.
    """
    from .video import Crop

    def down(v: int) -> int:
        return v - (v % 2)

    def up(v: int) -> int:
        return v + (v % 2)

    x0 = max(0, down(SEARCH_X[0] - margin))
    x1 = up(SEARCH_X[1] + margin)
    y0 = max(0, down(min(SLOTS) - 1 - margin))
    y1 = up(max(SLOTS) + ROW_HEIGHT[1] - 1 + margin)
    return Crop(x=x0, y=y0, width=x1 - x0, height=y1 - y0)

# Gaps inside a name run 1-4px; gaps between name, weapon icon, medal and
# victim run 10-23px. Anything larger is background, not part of the row.
TOKEN_GAP = 5
MAX_TOKEN_GAP = 40

# Token widths that define a well-formed row. Weapon slot is fixed-width and
# always second; names range from "04" to "MERCULES".
WEAPON_WIDTH = (44, 52)
MEDAL_WIDTH = (11, 17)
NAME_WIDTH = (18, 95)


@dataclass(frozen=True)
class Token:
    """One contiguous glyph group within a row: a name, an icon, a medal."""

    x0: int
    x1: int
    mask: np.ndarray

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1


@dataclass(frozen=True)
class RowStructure:
    """A slot that segments into a well-formed killfeed row.

    Names are exposed as tokens rather than strings: reading identity is the
    matcher's job, and nothing here infers a player from position.
    """

    top: int
    attacker: Token
    weapon: Token
    victim: Token
    medal_token: Token | None

    @property
    def medal(self) -> bool:
        return self.medal_token is not None

    @property
    def signature(self) -> tuple[int, int, int]:
        """Attacker, weapon and victim widths.

        Used to track one entry across frames as it rises up the grid.
        """
        return (self.attacker.width, self.weapon.width, self.victim.width)


def colour_ink(bgr: np.ndarray) -> np.ndarray:
    """Flat white or flat magenta killfeed text.

    For *rendering* a known row band, not for finding one: it matches plenty
    of gameplay content, so it can only be used where the row is already
    known to be.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (v > 165) & (s < 70)
    magenta = (h >= 140) & (h <= 175) & (s > 90) & (v > 110)
    return white | magenta


def value_ink(bgr: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """Binarise for *reading*: white and magenta text render identically."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2] > threshold


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    run: list[int] | None = None
    for i, on in enumerate(flags):
        if on:
            if run:
                run[1] = i
            else:
                run = [i, i]
        elif run:
            out.append((run[0], run[1]))
            run = None
    if run:
        out.append((run[0], run[1]))
    return out


def row_structure(bgr: np.ndarray, top: int, ink=value_ink) -> RowStructure | None:
    """Read one grid slot, or `None` if it does not hold a well-formed row.

    Rejection is on structure alone: token count, then the fixed-width weapon
    slot, then name widths.
    """
    tokens = segment_row(bgr, top, ink)
    if len(tokens) < 3:
        return None
    if not WEAPON_WIDTH[0] <= tokens[1].width <= WEAPON_WIDTH[1]:
        return None  # second token is not the weapon slot; not a row

    rest = tokens[2:]
    medal_token = None
    if MEDAL_WIDTH[0] <= rest[0].width <= MEDAL_WIDTH[1]:
        medal_token, rest = rest[0], rest[1:]
    if not rest:
        return None

    attacker, victim = tokens[0], rest[0]
    lo, hi = NAME_WIDTH
    if not (lo <= attacker.width <= hi and lo <= victim.width <= hi):
        return None
    return RowStructure(
        top=top,
        attacker=attacker,
        weapon=tokens[1],
        victim=victim,
        medal_token=medal_token,
    )


def detect_rows(bgr: np.ndarray, ink=value_ink) -> list[RowStructure]:
    """Every slot holding a well-formed row, newest (lowest) last."""
    found = (row_structure(bgr, top, ink) for top in SLOTS)
    return sorted((r for r in found if r is not None), key=lambda r: r.top)


def segment_row(bgr: np.ndarray, top: int, ink=value_ink) -> list[Token]:
    """Split one row into its glyph groups, left to right.

    Returns the attacker name, the weapon icon, an optional medal icon and
    the victim name, unlabelled; that is the parser's job. Token count varies
    with whether a medal is present. `ink` selects the binarisation used to
    find the gaps.
    """
    x0, x1 = SEARCH_X
    band = bgr[top - 1 : top + ROW_HEIGHT[1] - 1, x0:x1]
    mask = ink(band)
    cols = np.asarray(mask.sum(axis=0)) > 0

    tokens: list[Token] = []
    current: list[int] | None = None
    for a, b in _runs(cols):
        if current is None:
            current = [a, b]
            continue
        gap = a - current[1] - 1
        if gap < TOKEN_GAP:
            current[1] = b
        elif gap <= MAX_TOKEN_GAP:
            tokens.append(_token(mask, current, x0))
            current = [a, b]
        else:
            break  # past the row's right edge; the rest is background
    if current is not None:
        tokens.append(_token(mask, current, x0))
    return tokens


def _token(mask: np.ndarray, span: list[int], x_offset: int) -> Token:
    a, b = span
    return Token(x0=x_offset + a, x1=x_offset + b, mask=mask[:, a : b + 1])
