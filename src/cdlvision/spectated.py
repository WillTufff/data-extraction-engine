"""The spectated player's panel, bottom right, and the weapon name above it.

The panel (a coloured name bar over a four-column stat block: HP / K/D / STRK
/ a mode-dependent sixth) has the same polarity problem as the scoreboard: the
bar is magenta for one team and white for the other. `PanelInk` resolves it
the way `scoreboard.RowInk` does, cutting at a fixed fraction of the band's
own background-to-glyph ramp.

The stat values share the centre header's clock font cut, plus `/` for K/D.
Name (28px) and weapon name (13px) are separate cuts. The panel and ammo are
absent on free-camera frames, which reads as None rather than a failure.

The weapon name renders on raw gameplay, right-aligned under the ammo
counter, and is read as whole-region templates against a small closed set
rather than glyph by glyph.

`check_layout` re-checks the load-bearing geometry on every frame `read`
touches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .glyphs import GlyphAtlas, TemplateSet, segment_glyphs, tight_crop
from .names import NameMatcher, normalised_distance
from .scoreboard import INK_FRACTION

# -- geometry, all measured -----------------------------------------------

# The panel's content box. The name bar and the stat block share it exactly;
# the facecam to its left is not read.
PANEL_X = (1412, 1719)

# The name bar: borders at 895 and 953, an accent line at 955-957, and the
# name glyphs on a 28px cut between. NAME_Y is wider than the glyphs so a
# taller one is seen rather than clipped.
BAR_Y = (895, 953)
BAR_TOP = (897, 911)          # flat strip above the glyphs; the presence test
NAME_Y = (908, 946)
NAME_X = (1418, 1712)

# The stat block. Both text rows are 18px and sit at a pitch of 29.
LABEL_Y = (967, 986)
VALUE_Y = (995, 1016)

# Four columns, centred rather than left-aligned.
LABEL_SPANS = {
    "hp": (1435, 1463), "kd": (1506, 1547),
    "strk": (1576, 1634), "stat": (1653, 1713),
}
COLUMN_PITCH = 78
COLUMNS = {
    "hp": (1412, 1487), "kd": (1488, 1565),
    "strk": (1566, 1643), "stat": (1644, 1719),
}

# The weapon name, right-aligned at x=1875 on a 13px cut, and the ammo counter
# above it. Both sit on raw gameplay.
WEAPON_Y = (848, 868)
WEAPON_X = (1730, 1881)
AMMO_Y = (813, 841)
AMMO_X = (1770, 1881)

# The bar's top strip is flat chrome when the panel is drawn and gameplay
# when it is not; `present` is a cheap prefilter and `check_layout` is the
# real gate, since the panel fades in and out and the flatness distribution
# is continuous through the middle.
FLATNESS_CUT = 10.0

# Acceptance limits, from the distance distributions the atlas build reports.
# A value glyph is accepted on its margin to the runner-up, with the absolute
# distance kept only as a loose backstop: the block is translucent, so a
# bright scene lifts the ramp and the same digit renders a pixel thinner.
MAX_GLYPH_DISTANCE = 45
MIN_GLYPH_MARGIN = 8

# Least ink a span must carry to be a glyph rather than an artefact (below
# this, single bright pixels on the block's edge segment as their own glyph).
MIN_GLYPH_INK = 5

MAX_LABEL_DISTANCE = 40

# The weapon name is scored on the ink-normalised scale rather than
# absolutely.
MAX_WEAPON_DISTANCE = 0.35

CONFIG = Path(__file__).parent / "config"


def strip_crop():
    """The region a spectated strip must cache, derived from the geometry above.

    One box rather than two: the panel and the weapon name overlap in x and
    are close in y, so a single crop wastes almost nothing.
    """
    from .video import Crop

    x0, x1 = PANEL_X[0] - 2, WEAPON_X[1] + 2
    y0, y1 = AMMO_Y[0] - 2, VALUE_Y[1] + 4
    x0, y0 = x0 - x0 % 2, y0 - y0 % 2
    x1, y1 = x1 + x1 % 2, y1 + y1 % 2
    return Crop(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


class LayoutError(RuntimeError):
    """The frame does not have the geometry this module was measured on."""


# -- binarisation ----------------------------------------------------------


@dataclass(frozen=True)
class PanelInk:
    """The binarisation for one band of the panel, with polarity resolved.

    Same purpose as `scoreboard.RowInk`: the same glyph must render to the
    same mask whichever polarity it is drawn in. Unlike `RowInk`'s absolute
    cut, polarity here is decided by which extreme the band actually runs to,
    since the name bar can carry white text on a dark (magenta) background as
    well as dark text on a light one.
    """

    threshold: float
    inverted: bool

    @classmethod
    def measure(cls, band: np.ndarray) -> PanelInk:
        gray = _gray(band)
        background = float(np.median(gray))
        dark = float(np.percentile(gray, 2))
        bright = float(np.percentile(gray, 98))
        inverted = (background - dark) > (bright - background)
        glyph = dark if inverted else bright
        return cls(
            threshold=background + INK_FRACTION * (glyph - background),
            inverted=inverted,
        )

    def apply(self, band: np.ndarray) -> np.ndarray:
        gray = _gray(band)
        return gray < self.threshold if self.inverted else gray > self.threshold


def band_ink(band: np.ndarray) -> np.ndarray:
    """Binarise a band using a rule measured from that same band."""
    return PanelInk.measure(band).apply(band)


def _gray(band: np.ndarray) -> np.ndarray:
    if band.ndim == 2:
        return band.astype(float)
    return cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(float)


# -- presence --------------------------------------------------------------


def flatness(frame: np.ndarray) -> float:
    """Standard deviation of the name bar's top strip."""
    y0, y1 = BAR_TOP
    return float(_gray(frame[y0:y1, 1420:1710]).std())


def present(frame: np.ndarray) -> bool:
    """Is the panel drawn at all? Absence (free camera) is normal, not a failure."""
    return flatness(frame) < FLATNESS_CUT


def check_layout(frame: np.ndarray) -> None:
    """Fail loudly if the panel is drawn but not where it was measured.

    Checks the stat block's two text rows, 18px tall at a pitch of 29.
    """
    block = frame[LABEL_Y[0] - 8:VALUE_Y[1] + 4, PANEL_X[0]:PANEL_X[1]]
    rows = _spans(band_ink(block).sum(axis=1) > 3, LABEL_Y[0] - 8)
    tall = [(a, b) for a, b in rows if b - a >= 12]
    if len(tall) != 2:
        raise LayoutError(f"expected two text rows in the stat block, got {rows}")
    (la, lb), (va, vb) = tall
    if not (LABEL_Y[0] <= la and lb <= LABEL_Y[1]
            and VALUE_Y[0] <= va and vb <= VALUE_Y[1]):
        raise LayoutError(
            f"stat rows at {(la, lb)}, {(va, vb)}; expected inside "
            f"{LABEL_Y} and {VALUE_Y}"
        )


# -- reading ---------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One frame's spectated panel. Any field may be None: see the module note."""

    name: str | None
    hp: int | None
    kills: int | None
    deaths: int | None
    streak: int | None
    stat: str | None
    stat_label: str | None
    team: str | None
    weapon: str | None

    @property
    def fields(self) -> dict[str, object]:
        return {
            "name": self.name, "hp": self.hp, "kills": self.kills,
            "deaths": self.deaths, "streak": self.streak, "stat": self.stat,
            "stat_label": self.stat_label, "team": self.team,
            "weapon": self.weapon,
        }


class Reader:
    """Reads the panel with the atlases and template sets it was calibrated on."""

    def __init__(self, values: GlyphAtlas, names: NameMatcher,
                 labels: TemplateSet, weapons: TemplateSet | None = None):
        self.values = values
        self.names = names
        self.labels = labels
        self.weapons = weapons

    @classmethod
    def load(cls, config: Path = CONFIG) -> Reader:
        weapons = config / "spectated_weapons.json"
        return cls(
            values=GlyphAtlas.load(config / "atlas_spectated.json"),
            names=NameMatcher.load(config / "spectated_names.json"),
            labels=TemplateSet.load(config / "spectated_labels.json"),
            weapons=TemplateSet.load(weapons) if weapons.exists() else None,
        )

    def read(self, frame: np.ndarray) -> Panel | None:
        if not present(frame):
            return None
        check_layout(frame)
        label = self.read_label(frame)
        kills, deaths = self._read_kd(frame)
        return Panel(
            name=self.read_name(frame),
            hp=_as_int(self.read_value(frame, "hp")),
            kills=kills,
            deaths=deaths,
            streak=_as_int(self.read_value(frame, "strk")),
            stat=self.read_value(frame, "stat"),
            stat_label=label,
            team=team_of(frame),
            weapon=self.read_weapon(frame),
        )

    # -- one field at a time, so a scan can report each separately --------

    def read_name(self, frame: np.ndarray) -> str | None:
        """Classify the name bar against the closed set of players.

        Uses `NameMatcher` rather than `TemplateSet`: requiring shape
        agreement leaves many observations with no candidate at all.
        """
        mask = name_mask(frame)
        if mask.size == 0 or not mask.any():
            return None
        return self.names.match(mask).label

    def read_label(self, frame: np.ndarray) -> str | None:
        mask = cell_mask(frame, "stat", LABEL_Y)
        return _template(self.labels, mask, MAX_LABEL_DISTANCE)

    def read_value(self, frame: np.ndarray, column: str) -> str | None:
        """Read one numeric cell, or None if any glyph is not confidently known.

        All or nothing per cell: a partial cell is worse than none, since it
        parses as a plausible but wrong value.
        """
        mask = cell_mask(frame, column, VALUE_Y)
        spans = [
            (x0, x1) for x0, x1 in segment_glyphs(mask)
            if mask[:, x0:x1].sum() >= MIN_GLYPH_INK
        ]
        if not spans:
            return None
        out = []
        for x0, x1 in spans:
            try:
                hit = self.values.match(mask[:, x0:x1])
            except ValueError:
                # Larger than the canvas: two digits merged into one span.
                return None
            if hit.distance > MAX_GLYPH_DISTANCE or hit.margin < MIN_GLYPH_MARGIN:
                return None
            out.append(hit.char)
        return "".join(out) or None

    def read_weapon(self, frame: np.ndarray) -> str | None:
        """Classify the weapon name, on the ink-normalised scale.

        Not through `TemplateSet.match`: drawn on raw gameplay, the weapon
        name renders solid over a dark scene and hollow over a bright one, so
        its tight height varies for the same word, defeating shape agreement.
        """
        if self.weapons is None:
            return None
        mask = tight_crop(weapon_mask(frame))
        if mask.size == 0:
            return None
        distance, label = min(
            (normalised_distance(mask, t), k)
            for k, t in self.weapons.templates.items()
        )
        return label.split("#")[0] if distance <= MAX_WEAPON_DISTANCE else None

    def _read_kd(self, frame: np.ndarray) -> tuple[int | None, int | None]:
        text = self.read_value(frame, "kd")
        if not text or text.count("/") != 1:
            return None, None
        kills, _, deaths = text.partition("/")
        return _as_int(kills), _as_int(deaths)


# -- the crops each field is read from -------------------------------------


def name_mask(frame: np.ndarray) -> np.ndarray:
    """The name, binarised against the bar's own ramp."""
    y0, y1 = NAME_Y
    return band_ink(frame[y0:y1, NAME_X[0]:NAME_X[1]])


def cell_mask(frame: np.ndarray, column: str, rows: tuple[int, int]) -> np.ndarray:
    """One stat cell, binarised against the whole row band it sits in.

    Against the band, not the cell: a streak column holding a single "0" has
    too little ink to estimate a ramp from.
    """
    y0, y1 = rows
    rule = PanelInk.measure(frame[y0:y1, PANEL_X[0]:PANEL_X[1]])
    x0, x1 = COLUMNS[column]
    return rule.apply(frame[y0:y1, x0:x1])


def weapon_mask(frame: np.ndarray) -> np.ndarray:
    y0, y1 = WEAPON_Y
    return band_ink(frame[y0:y1, WEAPON_X[0]:WEAPON_X[1]])


def team_of(frame: np.ndarray) -> str | None:
    """Which side of the scoreboard the spectated player is on, from the
    bar's fill colour."""
    y0, y1 = BAR_TOP
    fill = float(np.median(_gray(frame[y0:y1, 1420:1710])))
    if fill > 200:
        return "right"
    if 100 < fill < 170:
        return "left"
    return None


def _template(templates: TemplateSet, mask: np.ndarray, limit: int) -> str | None:
    """Match a whole region, returning None rather than a poor guess.

    `LookupError` (no template of a compatible shape) is a reading, not an
    error.
    """
    if mask.size == 0 or not mask.any():
        return None
    try:
        hit = templates.match(mask)
    except LookupError:
        return None
    return hit.char if hit.distance <= limit else None


def _as_int(text: str | None) -> int | None:
    return int(text) if text and text.isdigit() else None


def _spans(flags: np.ndarray, offset: int = 0) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(flags):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((offset + start, offset + i - 1))
            start = None
    if start is not None:
        out.append((offset + start, offset + len(flags) - 1))
    return out
