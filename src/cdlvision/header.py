"""The live scoreboard's centre header: scores, clock, pips, mode, substrip.

Geometry is `scoreboard.HEADER_REGIONS`, refined here into the sub-windows a
reader actually addresses.

The mode strip is a notification band, not a fixed mode label: it often shows
a kill notification or the event title instead, so the mode is read as a
closed-set template match and anything else is None. The substrip is absent
in Hardpoint, and reads as None on gameplay rather than as a failure. The pip
count gives the best-of and series score from block brightness alone, no font
needed.

Score and clock atlases are separate font cuts (34px and 18px) from the team
panels' 12px font, built by `tools/build_header_atlas.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .glyphs import GlyphAtlas, TemplateSet, segment_glyphs, tight_crop
from .scoreboard import HEADER_REGIONS, INK_THRESHOLD, LayoutError

CONFIG = Path(__file__).parent / "config"

# -- sub-windows -------------------------------------------------------------
#
# `HEADER_REGIONS` gives the outer bounds of each band. Two of them hold more
# than one thing and are split here.

# The substrip's three cells. In Search and Destroy a dark centre panel
# carries "LIVES REMAINING" between two lighter side panels each holding a
# count; in Overload the whole band is one flat panel carrying "FIRST HALF".
SUBSTRIP_Y = (202, 224)
SUBSTRIP_LABEL_X = (826, 1101)
SUBSTRIP_LEFT_X = (726, 820)
SUBSTRIP_RIGHT_X = (1107, 1195)

# The objective row holds an icon and, in Hardpoint only, the hill rotation
# timer, windowed away from the icon since the icon has no stable width. The
# other two modes draw their own icons here (site letters, Overload marker),
# which fail the height gate rather than being recognised.
ROTATION_X = (955, 1025)
ROTATION_Y = (114, 143)

# Pip blocks are found, not indexed, so a three-pip best-of-5 panel isn't
# misread as five-pip with two missing.
PIP_PROFILE_CUT = 45   # block interior brightness vs. gap between blocks
PIP_MIN_WIDTH = 6      # blocks are ~12px wide
PIP_LIT = 110          # unlit blocks mean ~64, lit ones ~155-175

# Glyph heights, used to tell a token from a fade frame or a stray. Each band
# renders at exactly one height when drawn.
SCORE_HEIGHT = 34
CLOCK_HEIGHT = 18
ROTATION_HEIGHT = 22
# Lives counts are a separate font cut from the clock, at the substrip
# label's own height rather than the clock's.
LIVES_HEIGHT = 15

# Canvases; each must exceed its font's largest glyph.
SCORE_CANVAS = (38, 32)
CLOCK_CANVAS = (22, 19)
ROTATION_CANVAS = (26, 21)
LIVES_CANVAS = (19, 16)

# Width at which a score token holds more than one glyph.
SCORE_MERGED_WIDTH = 34

# Least ink a token must carry to be a glyph rather than a compression artefact.
MIN_TOKEN_INK = 8

# Accept limits on the Hamming scale `GlyphAtlas` reports.
MAX_SCORE_DISTANCE = 65
MAX_CLOCK_DISTANCE = 30
MAX_LIVES_DISTANCE = 22

# Whole-region template limits, on the same scale.
MAX_MODE_DISTANCE = 80
MAX_SUBSTRIP_DISTANCE = 60

MODES = ("CDL HARDPOINT", "CDL SEARCH AND DESTROY", "CDL OVERLOAD")


def region(frame: np.ndarray, key: str) -> np.ndarray:
    """One header band, at its own bounds.

    Addressed with no margin: the panel is translucent at its edge, so a
    band even slightly larger picks up gameplay behind it.
    """
    x0, x1, y0, y1 = HEADER_REGIONS[key]
    return frame[y0 : y1 + 1, x0 : x1 + 1]


def ink(bgr: np.ndarray, threshold: int = INK_THRESHOLD) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > threshold


def _split_wide(mask: np.ndarray, x0: int, x1: int, limit: int) -> list[tuple[int, int]]:
    """Split a token too wide to be one glyph, at its thinnest interior column.

    Same rule as `scoreboard.segment_cell`.
    """
    if x1 - x0 <= limit:
        return [(x0, x1)]
    column = mask[:, x0:x1].sum(axis=0)
    interior = range(3, (x1 - x0) - 3)
    middle = (x1 - x0) / 2
    cut = min(interior, key=lambda i: (column[i], abs(i - middle)))
    return _split_wide(mask, x0, x0 + cut, limit) + _split_wide(
        mask, x0 + cut, x1, limit
    )


def _glyphs(mask: np.ndarray, merged_width: int | None) -> list[np.ndarray]:
    spans: list[tuple[int, int]] = []
    for x0, x1 in segment_glyphs(mask, min_gap=1):
        if mask[:, x0:x1].sum() < MIN_TOKEN_INK:
            continue
        spans += (
            _split_wide(mask, x0, x1, merged_width) if merged_width else [(x0, x1)]
        )
    return [tight_crop(mask[:, x0:x1]) for x0, x1 in spans]


def tokens(
    mask: np.ndarray, height: int, merged_width: int | None = None
) -> list[np.ndarray] | None:
    """A number band's glyphs, or None if the band is not cleanly drawn.

    Height is checked on every token, not just the tallest, since a band
    fading in or out reads as a plausible smaller number otherwise.
    """
    if not mask.any():
        return None
    out = _glyphs(mask, merged_width)
    if not out or any(t.shape[0] != height for t in out):
        return None
    return out


# A colon against its own font's digits: narrower and shorter than a digit.
COLON_MAX_WIDTH = 5
COLON_MAX_HEIGHT = 0.8

# The final thirty seconds are a third counter, not a late state of the clock.
# It renders red rather than white, counts seconds and tenths rather than
# minutes and seconds, separates them with a decimal point rather than a colon,
# and is two or three digits wide rather than three: "28.5", "6.5". Its digits
# are the *same* 18px font cut as the clock's -- measured, the existing clock
# atlas matches them at distance 1 to 15 against a limit of 30 -- so this needs
# no atlas of its own, only its own ink and its own token rule.
REDNESS_CUT = 40          # white scores -12..-5, this counter up to 98
REDNESS_MARGIN = 25       # above the band median, to find the lit pixels
REDNESS_MIN_PIXELS = 20
TENTHS_INK_FRACTION = 0.40
TENTHS_MIN_DIGIT_HEIGHT = 10
TENTHS_DIGITS = (2, 3)
RAMP_MIN_CONTRAST = 30


def redness(band: np.ndarray) -> float | None:
    """How red the band's bright pixels are: R minus the larger of B and G.

    What tells the two clock formats apart, and it is not a close call. The
    ordinary white clock scores -12 to -5 over the whole VOD; the final-thirty
    counter scores up to 98. The cut sits in an empty gap 40 wide.

    This is colour used the way this project permits it -- as a prefilter that
    chooses *which* reader runs, with the bitmaps still deciding what the glyphs
    say. It is not colour deciding a value.
    """
    import cv2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    lit = gray > np.median(gray) + REDNESS_MARGIN
    if lit.sum() < REDNESS_MIN_PIXELS:
        return None
    blue, green, red = (band[..., i][lit].mean() for i in range(3))
    return float(red - max(blue, green))


def ramp_ink(band: np.ndarray, fraction: float = TENTHS_INK_FRACTION):
    """Binarise at a fraction of the band's own background-to-glyph ramp.

    The fixed `INK_THRESHOLD` of 150 is right for white glyphs and useless for
    these: the red ones peak at 142 and vanish entirely. Cutting relative to what
    the band actually contains is the same reasoning `killfeed.value_ink` and
    `scoreboard.RowInk` already use, for the same reason -- an absolute cut
    encodes an assumption about brightness that the broadcast is free to break.
    """
    import cv2

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(int)
    background, peak = float(np.median(gray)), float(gray.max())
    if peak - background < RAMP_MIN_CONTRAST:
        return None
    return gray > background + fraction * (peak - background)


def tenths_tokens(mask, min_height: int = TENTHS_MIN_DIGIT_HEIGHT):
    """The digits of an "SS.T" countdown, decimal point discarded.

    Two or three digits, and the separator is *ignored* rather than checked. The
    first version required it, at the cost of a quarter of the readings: the
    decimal point carries three or four ink pixels and drops out at the least
    provocation, which is the same trap the spectated panel's colon set when a
    min-ink filter of 12 ate its eight pixels. Anything tall enough to be a digit
    is a digit; anything shorter is punctuation and is not needed, because the
    last digit is the tenths by the format's own definition.

    Height is a floor rather than an equality for the same reason. The red glyphs
    render 17 to 19 px tall depending on how much of the glow clears the cut, and
    demanding exactly 18 cost 7% of the frames on its own.
    """
    if mask is None or not mask.any():
        return None
    digits = [g for g in _glyphs(mask, None) if g.shape[0] >= min_height]
    if not TENTHS_DIGITS[0] <= len(digits) <= TENTHS_DIGITS[1]:
        return None
    return digits


def clock_tokens(mask: np.ndarray, height: int) -> list[np.ndarray] | None:
    """A countdown band's glyphs as [M, :, S, S], or None.

    Stricter than `tokens`: exactly four tokens, three digits at the band's
    height and the second a colon (identified by position, not a template).
    """
    if not mask.any():
        return None
    out = _glyphs(mask, None)
    if len(out) != 4:
        return None
    if any(out[i].shape[0] != height for i in (0, 2, 3)):
        return None
    colon = out[1]
    if colon.shape[1] > COLON_MAX_WIDTH or colon.shape[0] > COLON_MAX_HEIGHT * height:
        return None
    return out


def pip_blocks(band: np.ndarray) -> list[tuple[int, int]]:
    """Column spans of the pip blocks in one pip band, found by their profile."""
    import cv2

    profile = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).mean(axis=0)
    on = profile > PIP_PROFILE_CUT
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for x, v in enumerate(on):
        if v and start is None:
            start = x
        elif not v and start is not None:
            spans.append((start, x - 1))
            start = None
    if start is not None:
        spans.append((start, len(on) - 1))
    return [s for s in spans if s[1] - s[0] + 1 >= PIP_MIN_WIDTH]


def read_pips(frame: np.ndarray, key: str) -> tuple[int, int] | None:
    """(maps won, maps needed) for one side, or None if the band is not drawn.

    Pips fill outward from the centre, so lit blocks must be contiguous from
    the inner end; otherwise returns None.
    """
    import cv2

    band = region(frame, key)
    blocks = pip_blocks(band)
    if not blocks:
        return None
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    lit = [gray[:, a : b + 1].mean() > PIP_LIT for a, b in blocks]
    if key == "pips_left":  # the inner end is the right-hand one
        lit = lit[::-1]
    if any(lit[i] and not lit[i - 1] for i in range(1, len(lit))):
        return None
    return int(sum(lit)), len(blocks)  # int(): numpy bool sum isn't JSON-serialisable


@dataclass(frozen=True)
class Header:
    """One frame's centre header. Anything unread is None, never a guess."""

    best_of: int | None
    maps_left: int | None
    maps_right: int | None
    score_left: int | None
    score_right: int | None
    clock: str | None
    rotation: str | None
    mode: str | None
    substrip: str | None
    lives_left: int | None
    lives_right: int | None
    worst_distance: int

    @property
    def clock_seconds(self) -> int | None:
        return seconds(self.clock)

    @property
    def rotation_seconds(self) -> int | None:
        return seconds(self.rotation)


def seconds(clock: str | None) -> float | None:
    """Either drawn form of the countdown, in seconds.

    "M:SS" for the ordinary clock and "SS.T" for the final thirty seconds, which
    the game draws as a different counter entirely. Returns a float because the
    second form carries tenths; whole seconds come back as whole floats.
    """
    if not clock:
        return None
    if "." in clock:
        seconds, _, tenths = clock.partition(".")
        if not seconds.isdigit() or not tenths.isdigit() or len(tenths) != 1:
            return None
        return int(seconds) + int(tenths) / 10
    if ":" not in clock:
        return None
    minutes, _, rest = clock.partition(":")
    if not minutes.isdigit() or not rest.isdigit() or len(rest) != 2:
        return None
    return float(int(minutes) * 60 + int(rest))


class Reader:
    """Reads the centre header off a frame, against the stored bitmaps."""

    def __init__(self, score, clock, rotation, lives, labels) -> None:
        self.score = score
        self.clock = clock
        self.rotation = rotation
        self.lives = lives
        self.labels = labels

    @classmethod
    def load(cls, config: Path = CONFIG) -> Reader:
        return cls(
            GlyphAtlas.load(config / "atlas_header_score.json"),
            GlyphAtlas.load(config / "atlas_header_clock.json"),
            GlyphAtlas.load(config / "atlas_header_rotation.json"),
            GlyphAtlas.load(config / "atlas_header_lives.json"),
            TemplateSet.load(config / "header_labels.json"),
        )

    # -- pieces ----------------------------------------------------------

    def _read_glyphs(self, glyphs, atlas, limit) -> tuple[str | None, int]:
        if glyphs is None:
            return None, 0
        out: list[str] = []
        worst = 0
        for glyph in glyphs:
            hit = atlas.match(glyph, max_shift=2)
            if hit.distance > limit:
                return None, hit.distance
            out.append(hit.char)
            worst = max(worst, hit.distance)
        return "".join(out), worst

    def read_score(self, frame: np.ndarray, key: str) -> tuple[int | None, int]:
        glyphs = tokens(
            ink(region(frame, key)), SCORE_HEIGHT, SCORE_MERGED_WIDTH
        )
        text, worst = self._read_glyphs(glyphs, self.score, MAX_SCORE_DISTANCE)
        if text is None or not text.isdigit():
            return None, worst
        return int(text), worst

    def read_clock(
        self, frame: np.ndarray, key: str = "clock"
    ) -> tuple[str | None, int]:
        """A countdown as "M:SS", or as "SS.T" in the final thirty seconds.

        Which of the two is drawn is decided by colour, before any matching: the
        counters are white and red and nothing sits between them. Reading the red
        one with the white one's ink threshold returns an empty mask, which looks
        exactly like a clock that is not drawn -- and a clock that is not drawn is
        what the end of a round looks like, which is why 100% unread below 0:30
        went unnoticed until something needed the value.
        """
        if key == "clock":
            band, height, atlas = region(frame, "clock"), CLOCK_HEIGHT, self.clock
        else:
            band = frame[
                ROTATION_Y[0] : ROTATION_Y[1] + 1,
                ROTATION_X[0] : ROTATION_X[1] + 1,
            ]
            height, atlas = ROTATION_HEIGHT, self.rotation

        red = redness(band)
        if red is not None and red > REDNESS_CUT:
            glyphs = tenths_tokens(ramp_ink(band))
            text, worst = self._read_glyphs(glyphs, atlas, MAX_CLOCK_DISTANCE)
            if text is None or not text.isdigit() or len(text) < 2:
                return None, worst
            text = f"{text[:-1]}.{text[-1]}"
        else:
            glyphs = clock_tokens(ink(band), height)
            text, worst = self._read_glyphs(glyphs, atlas, MAX_CLOCK_DISTANCE)
        if text is None or seconds(text) is None:
            return None, worst
        return text, worst

    def read_mode(self, frame: np.ndarray) -> tuple[str | None, int]:
        """One of the three mode titles, or None.

        None covers the strip showing the event title, a kill notification,
        or mid-transition; callers don't need to distinguish these.
        """
        mask = tight_crop(ink(region(frame, "mode")))
        if mask.size == 0:
            return None, 0
        try:
            hit = self.labels.match(mask, tolerance=2)
        except LookupError:
            return None, MAX_MODE_DISTANCE + 1
        if hit.char not in MODES or hit.distance > MAX_MODE_DISTANCE:
            return None, hit.distance
        return hit.char, hit.distance

    def read_substrip(self, frame: np.ndarray) -> tuple[str | None, int]:
        """The band under the mode strip, or None when nothing is drawn there.

        Hardpoint draws no substrip, so the window holds gameplay; the
        template match doubles as the presence test.
        """
        y0, y1 = SUBSTRIP_Y
        x0, x1 = SUBSTRIP_LABEL_X
        mask = tight_crop(ink(frame[y0 : y1 + 1, x0 : x1 + 1]))
        if mask.size == 0:
            return None, 0
        try:
            hit = self.labels.match(mask, tolerance=2)
        except LookupError:
            return None, MAX_SUBSTRIP_DISTANCE + 1
        if hit.char in MODES or hit.distance > MAX_SUBSTRIP_DISTANCE:
            return None, hit.distance
        return hit.char, hit.distance

    def read_lives(self, frame: np.ndarray, x: tuple[int, int]) -> int | None:
        """One flanking count of the "LIVES REMAINING" band, at its own atlas
        and height."""
        y0, y1 = SUBSTRIP_Y
        mask = ink(frame[y0 : y1 + 1, x[0] : x[1] + 1])
        glyphs = tokens(mask, LIVES_HEIGHT)
        text, _ = self._read_glyphs(glyphs, self.lives, MAX_LIVES_DISTANCE)
        return int(text) if text and text.isdigit() else None

    # -- whole header ----------------------------------------------------

    def read(self, frame: np.ndarray, check: bool = True) -> Header:
        """Read the whole header off one frame.

        `check` verifies both pip bands hold the same number of blocks (the
        same best-of, drawn twice); the header's counterpart to
        `scoreboard.check_mirror`.
        """
        left = read_pips(frame, "pips_left")
        right = read_pips(frame, "pips_right")
        if check and left and right and left[1] != right[1]:
            raise LayoutError(
                f"pip bands disagree on the best-of: {left[1]} blocks on the "
                f"left, {right[1]} on the right"
            )
        best_of = 2 * left[1] - 1 if left else None

        score_left, d_left = self.read_score(frame, "score_left")
        score_right, d_right = self.read_score(frame, "score_right")
        clock, d_clock = self.read_clock(frame, "clock")
        rotation, d_rotation = self.read_clock(frame, "rotation")
        mode, d_mode = self.read_mode(frame)
        substrip, d_sub = self.read_substrip(frame)

        lives = (None, None)
        if substrip == "LIVES REMAINING":
            lives = (
                self.read_lives(frame, SUBSTRIP_LEFT_X),
                self.read_lives(frame, SUBSTRIP_RIGHT_X),
            )

        return Header(
            best_of=best_of,
            maps_left=left[0] if left else None,
            maps_right=right[0] if right else None,
            score_left=score_left,
            score_right=score_right,
            clock=clock,
            rotation=rotation,
            mode=mode,
            substrip=substrip,
            lives_left=lives[0],
            lives_right=lives[1],
            worst_distance=max(
                d_left, d_right, d_clock, d_rotation, d_mode, d_sub
            ),
        )
