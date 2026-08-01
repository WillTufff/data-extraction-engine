"""Live scoreboard geometry: the two team panels and the centre header.

Reads the continuous per-player record shown on every LIVE frame: number,
name, K/D, streak, a mode-dependent sixth stat, gutter icon, spectated flag.

The sixth stat column's meaning (TIME/PLNT/OVLD) changes with the mode at
identical geometry, so its schema is read from the column header rather than
assumed. The two panels are an exact translation of one another by MIRROR
px; `check_mirror` re-verifies that invariant on every frame read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import Crop

# Horizontal offset from the left team panel to the right one.
MIRROR = 1224

# Left panel content box, in source coordinates. The right panel is this shifted
# by MIRROR; it is never written down separately.
PANEL_X = (38, 656)
PANEL_Y = (18, 226)

# The four player rows sit on a top-anchored grid. Row 0 is the highest, and on
# the left panel it is player 1; on the right, player 5.
ROW_TOP = 105
ROW_PITCH = 32
ROW_HEIGHT = 16
ROW_COUNT = 4
ROWS = tuple(ROW_TOP + i * ROW_PITCH for i in range(ROW_COUNT))

# The column-label row, one pitch above the grid but not on it. It carries the
# stat labels that name the schema, and a seed line that cycles with other
# text and is not relied on.
LABEL_TOP = 76

# Gutter icons overrun the text band in both directions, so the gutter is
# read from a taller band than the stat columns.
GUTTER_OVERRUN = (6, 6)

# The spectated player's row is drawn inverted, as dark text on a bright bar
# in the team's colour, filling its slot edge to edge.
HIGHLIGHT_RISE = 8
HIGHLIGHT_HEIGHT = ROW_PITCH

# Column windows, as (x0, x1) inclusive, relative to the source frame for the
# left panel. Boundaries sit in the gaps between columns, not on the ink.
# Columns are fixed, not flowed: a long name does not push K/D rightward,
# unlike the killfeed, so cells are sliced at these offsets and segmented
# only within.
COLUMNS = {
    # Tight on the icons themselves rather than on the gutter's full width,
    # since the panel's left border otherwise dominates every gutter cell.
    "gutter": (46, 80),
    "number": (90, 112),
    "name": (114, 455),
    "kd": (457, 529),
    "strk": (531, 593),
    "stat": (594, 650),
}

# The three schemas the stat column takes. Read from the label, never assumed.
STAT_LABELS = ("TIME", "PLNT", "OVLD")

# Centre header panel: best-of, map pips, both team scores, the game clock, an
# objective row, the mode strip, and a sub-strip that carries lives remaining
# in Search and Destroy and the half in Overload.
#
# Unlike the team panels this one is translucent, so it must be addressed at
# its own bounds and not with a margin: a wider band takes in the gameplay
# behind it, which floods the ink threshold over a bright map.
HEADER_X = (724, 1196)
HEADER_Y = (19, 226)

# Sub-regions of the header, measured over LIVE frames in all three modes.
# Written as (x0, x1, y0, y1) inclusive.
HEADER_REGIONS = {
    # Map pips, one per possible map, filling outward from the centre. Only a
    # won map's pip crosses the ink threshold; the rest are dim grey.
    "pips_left": (790, 880, 26, 44),
    "pips_right": (1035, 1130, 26, 44),
    "best_of": (908, 1010, 26, 44),
    # The game clock, and either side of it each team's score in a larger cut
    # of the font than the team panels use.
    "clock": (900, 1030, 68, 88),
    "score_left": (805, 900, 88, 124),
    "score_right": (1028, 1112, 88, 124),
    # Mode-specific: a hill icon and rotation timer in Hardpoint, the two bomb
    # sites in Search and Destroy, the overload marker in Overload.
    "objective": (905, 1025, 105, 145),
    "mode": (726, 1195, 168, 191),
    # Lives remaining in Search and Destroy, the half in Overload, nothing in
    # Hardpoint.
    "substrip": (726, 1195, 196, 225),
}

INK_THRESHOLD = 150  # flat chrome; same cut as the box score


@dataclass(frozen=True)
class Side:
    """One team panel: which side, and where its columns sit in the frame."""

    name: str
    offset: int

    def column(self, key: str) -> tuple[int, int]:
        x0, x1 = COLUMNS[key]
        return x0 + self.offset, x1 + self.offset

    def cell(self, frame: np.ndarray, row: int, key: str) -> np.ndarray:
        """The crop of one row's column, as BGR."""
        x0, x1 = self.column(key)
        top = ROWS[row]
        if key == "gutter":
            up, down = GUTTER_OVERRUN
            return frame[top - up : top + ROW_HEIGHT + down, x0 : x1 + 1]
        return frame[top : top + ROW_HEIGHT, x0 : x1 + 1]

    def label_cell(self, frame: np.ndarray, key: str) -> np.ndarray:
        """The crop of one column's header label."""
        x0, x1 = self.column(key)
        return frame[LABEL_TOP : LABEL_TOP + ROW_HEIGHT, x0 : x1 + 1]

    @property
    def players(self) -> tuple[int, ...]:
        """Roster numbers, which are fixed per side across the whole series."""
        return (1, 2, 3, 4) if self.offset == 0 else (5, 6, 7, 8)


LEFT = Side("left", 0)
RIGHT = Side("right", MIRROR)
SIDES = (LEFT, RIGHT)


def _even_down(v: int) -> int:
    return v - (v % 2)


def _even_up(v: int) -> int:
    return v + (v % 2)


def _crop(x: tuple[int, int], y: tuple[int, int], margin: int = 2) -> Crop:
    x0 = max(0, _even_down(x[0] - margin))
    y0 = max(0, _even_down(y[0] - margin))
    x1 = _even_up(x[1] + margin)
    y1 = _even_up(y[1] + margin)
    return Crop(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def strip_crops() -> dict[str, Crop]:
    """The regions a scoreboard strip must cache, derived from the geometry.

    Three crops rather than one bounding box, since the two panels sit at
    the far edges of the frame with the header between them.
    """
    return {
        "scoreboard_left": _crop(PANEL_X, PANEL_Y),
        "scoreboard_right": _crop(
            (PANEL_X[0] + MIRROR, PANEL_X[1] + MIRROR), PANEL_Y
        ),
        "scoreboard_header": _crop(HEADER_X, HEADER_Y),
    }


class LayoutError(Exception):
    """The frame does not hold the scoreboard layout this module expects."""


def ink(bgr: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    """Binarise flat overlay chrome for reading.

    Correct for every row except the spectated one; see `RowInk` for that.
    """
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > threshold


# Where the ink threshold sits between background and glyph, as a fraction of
# the non-inverted ramp (background 35, glyph 250).
INK_FRACTION = (INK_THRESHOLD - 35) / (250 - 35)

# A row whose background is brighter than this is drawn inverted.
POLARITY_CUT = 100


@dataclass(frozen=True)
class RowInk:
    """The binarisation for one scoreboard row, with its polarity resolved.

    The spectated player's row is drawn inverted (dark glyphs on a bright
    team-coloured bar), so a fixed cut misreads it. The cut here is placed at
    a fixed fraction of the row's own background-to-glyph ramp, estimated
    from the whole row band rather than a single cell, so the same glyph
    renders to the same mask regardless of polarity.
    """

    threshold: float
    inverted: bool

    @classmethod
    def measure(cls, band: np.ndarray) -> RowInk:
        """Derive the rule from a full row band, spanning every column."""
        import cv2

        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        background = float(np.median(gray))
        inverted = background > POLARITY_CUT
        glyph = float(np.percentile(gray, 1 if inverted else 99))
        return cls(
            threshold=background + INK_FRACTION * (glyph - background),
            inverted=inverted,
        )

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray < self.threshold if self.inverted else gray > self.threshold


def row_ink(frame: np.ndarray, side: Side, row: int) -> RowInk:
    """The binarisation rule for one row, measured from that row."""
    x0 = side.column("number")[0]
    x1 = side.column("stat")[1]
    top = ROWS[row]
    return RowInk.measure(frame[top : top + ROW_HEIGHT, x0 : x1 + 1])


def _spans(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


# Least ink a token must carry to be content rather than an artefact.
# Artefacts come from two sources: the column divider on an inverted row
# reading as a 1px "ink" token, and single compression-noise pixels on a flat
# bar. Both sit at the edge of the name column and would otherwise widen a
# name's bounding box.
MIN_TOKEN_INK = 20


def token_spans(
    mask: np.ndarray, gap: int = 6, min_ink: int = MIN_TOKEN_INK
) -> list[tuple[int, int]]:
    """Column spans of the glyph groups in a binarised cell, left to right."""
    merged: list[list[int]] = []
    for a, b in _spans(mask.sum(axis=0) > 0):
        if merged and a - merged[-1][1] - 1 < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if mask[:, a : b + 1].sum() >= min_ink]


# Widest a single glyph in this font gets, and the width at which a token
# must therefore hold more than one.
MAX_GLYPH_WIDTH = 10
MERGED_WIDTH = 13

# Columns at each end of a token that a split is never placed in, since the
# minimum-ink column is often the token's own faint outer edge.
SPLIT_MARGIN = 3


def segment_cell(mask: np.ndarray) -> list[tuple[int, int]]:
    """Split a binarised cell into per-character spans.

    `glyphs.segment_glyphs` cuts on a fully blank column, which is wrong here:
    adjacent characters in this font can interpenetrate with no blank column
    between them. A token is instead split only once it's too wide to be one
    character, at its thinnest interior column.
    """
    from .glyphs import segment_glyphs

    out: list[tuple[int, int]] = []
    todo = list(segment_glyphs(mask, min_gap=1))
    while todo:
        x0, x1 = todo.pop(0)
        if x1 - x0 < MERGED_WIDTH:
            out.append((x0, x1))
            continue
        ink_by_col = mask[:, x0:x1].sum(axis=0)
        interior = range(SPLIT_MARGIN, (x1 - x0) - SPLIT_MARGIN)
        middle = (x1 - x0) / 2
        cut = min(interior, key=lambda i: (ink_by_col[i], abs(i - middle)))
        todo = [(x0, x0 + cut), (x0 + cut, x1)] + todo
    return sorted(out)


def cell_mask(
    frame: np.ndarray, side: Side, row: int, key: str, rule: RowInk | None = None
) -> np.ndarray:
    """One row's column, binarised and trimmed to its real content.

    Everything that reads a cell goes through here rather than binarising it
    directly, so the artefact filter is always applied.
    """
    rule = rule or row_ink(frame, side, row)
    mask = rule.apply(side.cell(frame, row, key))
    spans = token_spans(mask)
    if not spans:
        return np.zeros((mask.shape[0], 0), dtype=bool)
    return mask[:, spans[0][0] : spans[-1][1] + 1]


# -- the gutter icon -----------------------------------------------------
#
# The gutter holds a skull when the player is dead, an objective marker when
# they hold it, or nothing. The icon is tinted (per team colour or white),
# sometimes filled and sometimes an outline, and inverted on the spectated
# row, so neither a value threshold nor a colour filter identifies it. What
# it sits on is flat, so ink is defined as "far from this cell's own
# background colour" in Lab, which is blind to tint and polarity. The
# background is estimated from the cell's left/right border columns rather
# than the whole cell, since a filled disc would otherwise dominate a
# whole-cell median.
#
# The skull masks to a crisp silhouette in every variant and is matched
# against a template; a death with no killfeed row is a suicide. The
# objective icons are not separated from each other, since which objective
# is on screen is already fixed by the mode.
GUTTER_BACKGROUND_MARGIN = 2  # border columns used to estimate the background
GUTTER_INK_DISTANCE = 40      # Lab distance at which a pixel is icon, not chrome
MIN_ICON_INK = 40             # below this the cell holds no icon at all

SKULL = "skull"
OBJECTIVE = "objective"
NO_ICON = "none"

MAX_SKULL_DISTANCE = 0.25


def gutter_mask(cell: np.ndarray) -> np.ndarray:
    """Binarise a gutter cell, blind to the icon's tint and polarity."""
    import cv2

    lab = cv2.cvtColor(cell, cv2.COLOR_BGR2LAB).astype(np.int16)
    m = GUTTER_BACKGROUND_MARGIN
    border = np.concatenate(
        [lab[:, :m].reshape(-1, 3), lab[:, -m:].reshape(-1, 3)]
    )
    background = np.median(border, axis=0)
    return np.linalg.norm(lab - background, axis=2) > GUTTER_INK_DISTANCE


CONFIG = Path(__file__).parent / "config"

# Accept limits, on the same Hamming scale `GlyphAtlas` reports. A cell that
# exceeds one is reported unknown rather than guessed.
MAX_GLYPH_DISTANCE = 30
MAX_NAME_DISTANCE = 40


@dataclass(frozen=True)
class PlayerRow:
    """One player's row. Anything unreadable is None, never a guess."""

    side: str
    slot: int
    number: int | None
    name: str | None
    kills: int | None
    deaths: int | None
    streak: int | None
    stat: str | None
    gutter: str
    spectated: bool
    worst_distance: int

    @property
    def dead(self) -> bool:
        """A skull in the gutter: marks every death, not just an unattributed one."""
        return self.gutter == SKULL

    @property
    def player(self) -> int | None:
        """Roster number, which joins this row to the box score."""
        return self.number


@dataclass(frozen=True)
class Reading:
    """One frame's scoreboard.

    `stat_label` carries the schema (TIME/PLNT/OVLD) as read, not an inferred
    mode name; callers map it to a mode name themselves.
    """

    stat_label: str | None
    rows: tuple[PlayerRow, ...]

    def by_player(self) -> dict[int, PlayerRow]:
        return {r.number: r for r in self.rows if r.number is not None}

    @property
    def roster(self) -> dict[int, str]:
        """The number -> name mapping this frame asserts."""
        return {
            r.number: r.name
            for r in self.rows
            if r.number is not None and r.name is not None
        }


def foreign(reading_roster: dict[int, str], series_roster: dict[int, str]) -> bool:
    """Does this frame's roster contradict the series it is supposed to be in?

    Catches recap clips of other matches: they are classified LIVE (correctly
    -- it is live gameplay) but carry a scoreboard for a different series.
    Works because identity is read: a frame where a roster number maps to a
    different name than the series roster has answered the question.
    """
    return any(
        number in series_roster and name != series_roster[number]
        for number, name in reading_roster.items()
    )


class Reader:
    """Reads player rows off a frame, against the stored bitmaps."""

    def __init__(self, atlas, names, labels, icons) -> None:
        self.atlas = atlas
        self.names = names
        self.labels = labels
        self.icons = icons

    @classmethod
    def load(cls, config: Path = CONFIG) -> Reader:
        from .glyphs import GlyphAtlas, TemplateSet

        return cls(
            GlyphAtlas.load(config / "atlas_scoreboard.json"),
            TemplateSet.load(config / "scoreboard_names.json"),
            TemplateSet.load(config / "scoreboard_labels.json"),
            TemplateSet.load(config / "scoreboard_icons.json"),
        )

    # -- pieces ----------------------------------------------------------

    def read_number(self, mask: np.ndarray) -> tuple[str | None, int]:
        """Read a numeric cell character by character."""
        if mask.size == 0:
            return None, 0
        out: list[str] = []
        worst = 0
        for x0, x1 in segment_cell(mask):
            hit = self.atlas.match(mask[:, x0:x1])
            if hit.distance > MAX_GLYPH_DISTANCE:
                return None, hit.distance
            out.append(hit.char)
            worst = max(worst, hit.distance)
        return ("".join(out) or None), worst

    def read_name(self, mask: np.ndarray) -> tuple[str | None, int]:
        if mask.size == 0:
            return None, 0
        try:
            hit = self.names.match(mask)
        except LookupError:
            return None, MAX_NAME_DISTANCE + 1  # no template of this width
        if hit.distance > MAX_NAME_DISTANCE:
            return None, hit.distance
        return hit.char, hit.distance

    def read_gutter(self, cell: np.ndarray) -> str:
        """Classify one gutter cell as skull, objective or nothing.

        Only the skull is matched against a template; anything else with ink
        is an objective marker, whichever one is already fixed by the mode.
        """
        from .names import normalised_distance

        mask = gutter_mask(cell)
        if mask.sum() < MIN_ICON_INK:
            return NO_ICON
        from .glyphs import tight_crop

        probe = tight_crop(mask)
        skull = self.icons.templates.get(SKULL)
        if skull is not None and normalised_distance(probe, skull) < MAX_SKULL_DISTANCE:
            return SKULL
        return OBJECTIVE

    def read_stat_label(self, frame: np.ndarray) -> str | None:
        """Which of TIME/PLNT/OVLD the sixth column holds on this frame."""
        mask = ink(LEFT.label_cell(frame, "stat"))
        spans = token_spans(mask)
        if not spans:
            return None
        try:
            hit = self.labels.match(mask[:, spans[0][0] : spans[-1][1] + 1])
        except LookupError:
            return None
        return hit.char if hit.char in STAT_LABELS else None

    # -- whole rows ------------------------------------------------------

    def read_row(self, frame: np.ndarray, side: Side, slot: int) -> PlayerRow:
        rule = row_ink(frame, side, slot)
        cells = {
            key: cell_mask(frame, side, slot, key, rule)
            for key in ("number", "name", "kd", "strk", "stat")
        }
        number, d_number = self.read_number(cells["number"])
        name, d_name = self.read_name(cells["name"])
        kd, d_kd = self.read_number(cells["kd"])
        strk, d_strk = self.read_number(cells["strk"])
        stat, d_stat = self.read_number(cells["stat"])

        kills = deaths = None
        if kd and kd.count("/") == 1:
            left, right = kd.split("/")
            if left.isdigit() and right.isdigit():
                kills, deaths = int(left), int(right)

        return PlayerRow(
            side=side.name,
            slot=slot,
            number=int(number) if number and number.isdigit() else None,
            name=name,
            kills=kills,
            deaths=deaths,
            streak=int(strk) if strk and strk.isdigit() else None,
            stat=stat,
            gutter=self.read_gutter(side.cell(frame, slot, "gutter")),
            spectated=rule.inverted,
            worst_distance=max(d_number, d_name, d_kd, d_strk, d_stat),
        )

    def read(self, frame: np.ndarray, check: bool = True) -> Reading:
        """Read every player row on a frame.

        `check` runs the mirror invariant first; turn it off only when reading
        a strip that carries one panel and not the other.
        """
        if check:
            check_mirror(frame)
        return Reading(
            stat_label=self.read_stat_label(frame),
            rows=tuple(
                self.read_row(frame, side, slot)
                for side in SIDES
                for slot in range(ROW_COUNT)
            ),
        )


def check_mirror(frame: np.ndarray, tolerance: int = 2) -> None:
    """Verify both panels' stat labels sit MIRROR apart, or raise.

    The labels are the same glyphs drawn twice, so their spans must agree to
    the pixel.
    """
    for key in ("kd", "strk", "stat"):
        found = []
        for side in SIDES:
            spans = token_spans(ink(side.label_cell(frame, key)))
            if not spans:
                raise LayoutError(f"no {key} label on the {side.name} panel")
            found.append((spans[0][0], spans[-1][1]))
        (l0, l1), (r0, r1) = found
        if abs(l0 - r0) > tolerance or abs(l1 - r1) > tolerance:
            raise LayoutError(
                f"{key} label is not mirrored: left {l0}..{l1}, right {r0}..{r1} "
                f"(offsets differ by {abs(l0 - r0)}, {abs(l1 - r1)}px)"
            )
