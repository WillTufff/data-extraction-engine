"""Glyph atlas and matcher for machine-rendered broadcast text.

Recognition is exact-shape matching against binarised bitmaps rather than
OCR. Atlas files are plain JSON with bitmaps encoded as '.'/'#' rows, so they
stay readable and diffable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Binarisation cut for greyscale -> ink mask.
DEFAULT_THRESHOLD = 150

# Default canvas that a tight-cropped glyph is anchored into before comparison.
# Must exceed the largest glyph in the font class it's used for; GlyphAtlas
# instances may override it per font.
CANVAS = (32, 28)  # (height, width)


def binarize(gray: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Threshold a greyscale crop into a boolean ink mask."""
    return gray > threshold


def segment_glyphs(
    mask: np.ndarray, min_gap: int = 1, min_ink: int = 1
) -> list[tuple[int, int]]:
    """Split a binarised text cell into per-character column spans.

    Returns `(x0, x1)` half-open spans in left-to-right order. A run of at
    least `min_gap` fully blank columns separates characters.
    """
    ink = mask.sum(axis=0)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    blank = 0
    for x, value in enumerate(ink):
        if value >= min_ink:
            if start is None:
                start = x
            blank = 0
        elif start is not None:
            blank += 1
            if blank >= min_gap:
                spans.append((start, x - blank + 1))
                start = None
    if start is not None:
        spans.append((start, len(ink)))
    return spans


def tight_crop(mask: np.ndarray) -> np.ndarray:
    """Crop a mask to its ink bounding box."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def to_canvas(mask: np.ndarray, canvas: tuple[int, int] = CANVAS) -> np.ndarray:
    """Anchor a tight-cropped glyph top-left into a fixed canvas."""
    h, w = canvas
    out = np.zeros(canvas, dtype=bool)
    gh, gw = mask.shape
    if gh > h or gw > w:
        raise ValueError(f"glyph {mask.shape} exceeds canvas {canvas}")
    out[:gh, :gw] = mask
    return out


def _encode(mask: np.ndarray) -> list[str]:
    return ["".join("#" if v else "." for v in row) for row in mask]


def _decode(rows: list[str]) -> np.ndarray:
    return np.array([[c == "#" for c in row] for row in rows], dtype=bool)


@dataclass(frozen=True)
class Match:
    """Outcome of matching one glyph against an atlas.

    `margin` is the separation between the best and runner-up candidates, and
    is the confidence signal: a small margin means genuine ambiguity.
    """

    char: str
    distance: int
    runner_up: str | None
    runner_up_distance: int

    @property
    def margin(self) -> int:
        return self.runner_up_distance - self.distance


class GlyphAtlas:
    """A set of labelled glyph bitmaps for one font class."""

    def __init__(
        self,
        font: str,
        glyphs: dict[str, np.ndarray] | None = None,
        canvas: tuple[int, int] = CANVAS,
    ):
        self.font = font
        self.canvas = tuple(canvas)
        self.glyphs: dict[str, np.ndarray] = {}
        for char, mask in (glyphs or {}).items():
            self.add(char, mask)

    def add(self, char: str, mask: np.ndarray) -> None:
        """Register a glyph, anchoring it into this atlas's canvas."""
        self.glyphs[char] = to_canvas(tight_crop(mask), self.canvas)

    def match(self, mask: np.ndarray, max_shift: int = 1) -> Match:
        """Identify a single glyph.

        Compares Hamming distance on the canvas, probing +/-`max_shift` pixel
        alignments to tolerate small crop offsets.
        """
        if not self.glyphs:
            raise ValueError(f"atlas '{self.font}' is empty")
        probe = to_canvas(tight_crop(mask), self.canvas)

        scored: list[tuple[int, str]] = []
        for char, template in self.glyphs.items():
            best = None
            for dy in range(-max_shift, max_shift + 1):
                for dx in range(-max_shift, max_shift + 1):
                    shifted = np.roll(np.roll(probe, dy, axis=0), dx, axis=1)
                    d = int(np.count_nonzero(shifted ^ template))
                    best = d if best is None else min(best, d)
            scored.append((best, char))

        scored.sort()
        distance, char = scored[0]
        runner_up_distance, runner_up = (
            scored[1] if len(scored) > 1 else (distance, None)
        )
        return Match(char, distance, runner_up, runner_up_distance)

    def read(
        self, gray: np.ndarray, threshold: int = DEFAULT_THRESHOLD, min_gap: int = 1
    ) -> tuple[str, list[Match]]:
        """Read a whole text cell, returning the string and per-glyph matches."""
        mask = binarize(gray, threshold)
        matches = [
            self.match(mask[:, x0:x1]) for x0, x1 in segment_glyphs(mask, min_gap)
        ]
        return "".join(m.char for m in matches), matches

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "font": self.font,
            "canvas": list(self.canvas),
            "threshold": DEFAULT_THRESHOLD,
            "glyphs": {
                char: _encode(tight_crop(mask))
                for char, mask in sorted(self.glyphs.items())
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> GlyphAtlas:
        payload = json.loads(Path(path).read_text())
        atlas = cls(payload["font"], canvas=tuple(payload.get("canvas", CANVAS)))
        for char, rows in payload["glyphs"].items():
            atlas.add(char, _decode(rows))
        return atlas

    def __len__(self) -> int:
        return len(self.glyphs)

    def __repr__(self) -> str:
        chars = "".join(sorted(self.glyphs))
        return (
            f"GlyphAtlas(font={self.font!r}, canvas={self.canvas}, "
            f"glyphs={chars!r})"
        )


class TemplateSet:
    """Whole-region bitmap templates for fixed strings and icons.

    For overlay text drawn from a closed set of fixed strings (mode titles,
    stage names, team names), matched as one bitmap rather than character by
    character. Unlike `GlyphAtlas`, templates vary in size, so shape must
    agree within a pixel of slack before comparison.
    """

    def __init__(self, name: str, templates: dict[str, np.ndarray] | None = None):
        self.name = name
        self.templates: dict[str, np.ndarray] = dict(templates or {})

    def add(self, label: str, mask: np.ndarray) -> None:
        self.templates[label] = tight_crop(mask)

    def match(self, mask: np.ndarray, tolerance: int = 1) -> Match:
        """Identify a region against the template set."""
        if not self.templates:
            raise ValueError(f"template set '{self.name}' is empty")
        probe = tight_crop(mask)

        scored: list[tuple[int, str]] = []
        for label, template in self.templates.items():
            dh = abs(template.shape[0] - probe.shape[0])
            dw = abs(template.shape[1] - probe.shape[1])
            if dh > tolerance or dw > tolerance:
                continue
            h = max(template.shape[0], probe.shape[0])
            w = max(template.shape[1], probe.shape[1])
            a = np.zeros((h, w), dtype=bool)
            b = np.zeros((h, w), dtype=bool)
            a[: probe.shape[0], : probe.shape[1]] = probe
            b[: template.shape[0], : template.shape[1]] = template
            best = min(
                int(np.count_nonzero(np.roll(np.roll(a, dy, 0), dx, 1) ^ b))
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
            scored.append((best, label))

        if not scored:
            raise LookupError(
                f"no template in '{self.name}' has a compatible shape for "
                f"a {probe.shape} region"
            )
        scored.sort()
        distance, label = scored[0]
        runner_distance, runner = scored[1] if len(scored) > 1 else (distance, None)
        return Match(label, distance, runner, runner_distance)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "templates": {k: _encode(v) for k, v in sorted(self.templates.items())},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> TemplateSet:
        payload = json.loads(Path(path).read_text())
        return cls(
            payload["name"],
            {k: _decode(v) for k, v in payload["templates"].items()},
        )

    def __len__(self) -> int:
        return len(self.templates)
