"""Build the centre header's atlases from the counters themselves, not from labels.

The header uses two cuts of the font not covered elsewhere: the team scores at
34px and the clock at 18px. Both fonts only ever draw a counter, so each
counter's own structure labels its digits: the score counts up, the clock
counts down, and a step that leaves every digit but one unchanged names that
digit's successor. Those successors chain into a path over ten classes (not a
cycle -- the wrap step also carries into the next place and is unobservable).
The score's carry (9->10, 99->100) and the clock's tens position (0..5,
decrementing under the ones labels) then confirm the path independently.

Checks use each class's modal step rather than every step seen, since a few
frames disagree with their neighbours.

The mode and substrip titles are a closed set, built as whole-region templates
from the fixtures. "SECOND HALF" appears in no fixture and is taken from a
named frame instead. The best-of needs no template: it is the pip count from
`header.read_pips`.

    uv run python tools/build_header_atlas.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cdlvision import header as hd  # noqa: E402
from cdlvision import scoreboard as sb  # noqa: E402
from cdlvision import video  # noqa: E402
from cdlvision.glyphs import GlyphAtlas, TemplateSet, tight_crop  # noqa: E402
from cdlvision.progress import Progress  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "src/cdlvision/config"

FIXTURES = sorted((ROOT / "tests/fixtures").glob("scoreboard_t*.json"))

# Titles not covered by any fixture; read off a named frame instead.
EXTRA_SUBSTRIPS = {"SECOND HALF": 3610.0}

# Adjacent strip frames are 1/6s apart; a wider gap is a cut and no step can
# be inferred across it.
MAX_GAP = 0.25

# Hamming radius at which two glyph observations count as the same character.
CLUSTER_RADIUS = {"score": 65, "clock": 24, "rotation": 30, "lives": 20}

# Radius at which an observation is admitted to a label template's majority.
REFINE_RADIUS = 140

BANDS = {
    # name -> (atlas file, glyph height, canvas, merged-width limit)
    "score": ("atlas_header_score.json", hd.SCORE_HEIGHT, hd.SCORE_CANVAS,
              hd.SCORE_MERGED_WIDTH),
    "clock": ("atlas_header_clock.json", hd.CLOCK_HEIGHT, hd.CLOCK_CANVAS, None),
    "rotation": ("atlas_header_rotation.json", hd.ROTATION_HEIGHT,
                 hd.ROTATION_CANVAS, None),
    "lives": ("atlas_header_lives.json", hd.LIVES_HEIGHT, hd.LIVES_CANVAS, None),
}


class Clusters:
    """Online clustering of glyph observations, with a running majority.

    An exact-bitmap cache answers most lookups without a distance comparison.
    """

    def __init__(self, radius: int, canvas: tuple[int, int]) -> None:
        self.radius = radius
        self.canvas = canvas
        self.reps: list[np.ndarray] = []
        self.counts: list[Counter] = []
        self.sums: list[dict[tuple[int, int], np.ndarray]] = []
        self.cache: dict[bytes, int] = {}

    def _canvas(self, mask: np.ndarray) -> np.ndarray:
        out = np.zeros(self.canvas, dtype=bool)
        out[: mask.shape[0], : mask.shape[1]] = mask
        return out

    def _distance(self, a: np.ndarray, b: np.ndarray) -> int:
        return min(
            int(np.count_nonzero(np.roll(np.roll(a, dy, 0), dx, 1) ^ b))
            for dy in (-2, -1, 0, 1, 2)
            for dx in (-2, -1, 0, 1, 2)
        )

    def add(self, mask: np.ndarray) -> int:
        key = mask.shape[1].to_bytes(2, "little") + mask.tobytes()
        hit = self.cache.get(key)
        if hit is None:
            probe = self._canvas(mask)
            hit = next(
                (
                    i
                    for i, rep in enumerate(self.reps)
                    if self._distance(rep, probe) < self.radius
                ),
                None,
            )
            if hit is None:
                hit = len(self.reps)
                self.reps.append(probe)
                self.counts.append(Counter())
                self.sums.append({})
            self.cache[key] = hit
        self.counts[hit][mask.shape] += 1
        total = self.sums[hit].get(mask.shape)
        self.sums[hit][mask.shape] = mask.astype(np.int32) + (
            0 if total is None else total
        )
        return hit

    def template(self, index: int) -> np.ndarray:
        """Pixelwise majority over every observation, at the modal shape.

        Observations of another shape are dropped rather than padded.
        """
        shape, n = self.counts[index].most_common(1)[0]
        return self.sums[index][shape] > n / 2

    def size(self, index: int) -> int:
        return sum(self.counts[index].values())


def collect(strip, spans, limit: int | None, labels=None):
    """One pass over the header strip, clustering every band's glyphs.

    Returns the clusterings and, per band, the time-ordered sequence of cluster
    tuples the counter took. That sequence is the input to the labelling below.
    """
    canvas = video.Canvas(strip.crop, 1920, 1080)
    clusters = {
        name: Clusters(CLUSTER_RADIUS[name], BANDS[name][2]) for name in BANDS
    }
    series: dict[str, list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
    majority = LabelMajority()
    total = limit or strip.frames

    with Progress("build_header_atlas", total=total, label="read header") as bar:
        for i, (ts, strip_frame) in enumerate(video.stream_strip(strip)):
            if limit and i >= limit:
                break
            if i % 200 == 0:
                bar.set(i)
            if not any(a <= ts <= b for a, b in spans):
                continue
            frame = canvas.place(strip_frame)
            majority.observe(labels, hd.region(frame, "mode"))
            substrip, _ = majority.observe(labels, _substrip_band(frame))
            # Lives band is only drawn in Search and Destroy; the substrip label
            # says whether it's open. Outside that mode the window falls on raw
            # gameplay and must be skipped rather than fed to the clusterer.
            lives_open = substrip == "LIVES REMAINING"
            for name, (_, height, _, merged) in BANDS.items():
                if name == "lives" and not lives_open:
                    continue
                for band, key in _bands(frame, name):
                    # Countdown bands hold a colon (shorter than the digits),
                    # so they need clock_tokens rather than the flat gate in
                    # tokens.
                    glyphs = (
                        hd.tokens(hd.ink(band), height, merged)
                        if name in ("score", "lives")
                        else hd.clock_tokens(hd.ink(band), height)
                    )
                    if glyphs is None:
                        continue
                    series[f"{name}:{key}"].append(
                        (ts, tuple(clusters[name].add(g) for g in glyphs))
                    )
    return clusters, series, majority


class LabelMajority:
    """Accumulates whole-region label observations into pixelwise majorities."""

    def __init__(self) -> None:
        self.counts: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
        self.sums: dict[tuple[str, tuple[int, int]], np.ndarray] = {}

    def add(self, label: str, mask: np.ndarray) -> None:
        key = (label, mask.shape)
        self.counts[key] += 1
        total = self.sums.get(key)
        self.sums[key] = mask.astype(np.int32) + (0 if total is None else total)

    def observe(self, templates, band: np.ndarray) -> tuple[str | None, int]:
        mask = tight_crop(hd.ink(band))
        if mask.size == 0:
            return None, 0
        try:
            hit = templates.match(mask, tolerance=2)
        except LookupError:
            return None, 0
        if hit.distance > REFINE_RADIUS:
            return None, hit.distance
        self.add(hit.char, mask)
        return hit.char, hit.distance

    def rebuild(self, templates, report) -> None:
        labels = {label for label, _ in self.counts}
        for label in sorted(labels):
            shapes = {s: n for (l, s), n in self.counts.items() if l == label}
            shape, n = max(shapes.items(), key=lambda kv: kv[1])
            templates.templates[label] = self.sums[(label, shape)] > n / 2
            report(f"  {label:24s} n={n:6d} at {shape}, "
                   f"{len(shapes)} shapes seen")


def _bands(frame, name):
    if name == "score":
        yield hd.region(frame, "score_left"), "left"
        yield hd.region(frame, "score_right"), "right"
    elif name == "lives":
        y0, y1 = hd.SUBSTRIP_Y
        for key, x in (("left", hd.SUBSTRIP_LEFT_X), ("right", hd.SUBSTRIP_RIGHT_X)):
            yield frame[y0 : y1 + 1, x[0] : x[1] + 1], key
    elif name == "clock":
        yield hd.region(frame, "clock"), "clock"
    else:
        yield (
            frame[
                hd.ROTATION_Y[0] : hd.ROTATION_Y[1] + 1,
                hd.ROTATION_X[0] : hd.ROTATION_X[1] + 1,
            ],
            "rotation",
        )


def steps(series: list[tuple[float, tuple[int, ...]]]):
    """Adjacent contiguous samples where the reading changed."""
    out = []
    for (t0, a), (t1, b) in zip(series, series[1:]):
        if 0 < t1 - t0 <= MAX_GAP and a != b:
            out.append((a, b))
    return out


def _modal_map(votes: dict[int, Counter]) -> dict[int, int]:
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def _evidence(votes: dict[int, Counter]) -> str:
    """Every observed edge with its weight, strongest first."""
    edges = sorted(
        ((n, a, b) for a, c in votes.items() for b, n in c.items()), reverse=True
    )
    return " ".join(f"{a}->{b}x{n}" for n, a, b in edges)


def _chain(successor: dict[int, int], expected: int = 10) -> list[int] | None:
    """The single path through every digit class, or None if it is not one.

    Successors form nine edges over ten nodes -- a Hamiltonian path, not a
    cycle, since the wrap step is unobservable (it always carries).
    """
    nodes = set(successor) | set(successor.values())
    if len(nodes) != expected:
        return None
    ends = set(successor.values())
    starts = [c for c in nodes if c not in ends]
    if len(starts) != 1:
        return None
    order = [starts[0]]
    while len(order) < expected:
        nxt = successor.get(order[-1])
        if nxt is None or nxt in order:
            return None
        order.append(nxt)
    return order


def label_counter(series, report) -> dict[int, str] | None:
    """Label an ascending decimal counter's clusters -- the score.

    A step that leaves every digit but the last unchanged names an ascending
    successor; those chain into a path from zero to nine. A step that adds a
    digit is a carry (9->10 or 99->100), which names nine, one and zero
    independently and checks the path.
    """
    successors: dict[int, Counter] = defaultdict(Counter)
    anchors: Counter = Counter()
    for a, b in series:
        if len(a) == len(b) and a[:-1] == b[:-1]:
            successors[a[-1]][b[-1]] += 1
        elif len(b) == len(a) + 1 and set(b[1:]) == {b[-1]} and b[0] != b[-1]:
            anchors[(a[-1], b[0], b[-1])] += 1

    report(f"ones edges: {_evidence(successors)}")
    order = _chain(_modal_map(successors))
    if order is None:
        report("the ascending successors are not one path over ten classes")
        return None
    labels = {c: str(i) for i, c in enumerate(order)}
    report(f"ascending path over ten classes: {order} = 0..9")

    if not anchors:
        report("no carry into a new digit was observed; the path is unconfirmed")
        return None
    (nine, one, zero), n = anchors.most_common(1)[0]
    report(f"carry seen {n} times: ...{nine} -> {one}{zero}...  "
           f"({len(anchors)} distinct carries, the rest {anchors.most_common()[1:]})")
    read = (labels.get(nine), labels.get(one), labels.get(zero))
    if read != ("9", "1", "0"):
        report(f"the carry disagrees with the path: it reads ...{read[0]} -> "
               f"{read[1]}{read[2]}..., which is not ...9 -> 10...")
        return None
    report("the carry confirms the path independently")
    return labels


def label_countdown(series, report) -> dict[int, str] | None:
    """Label a descending M:SS countdown's clusters -- the clock.

    The ones digit's descending successors form a path from nine to zero. The
    seconds' tens position (0..5, same font and classes) then checks it: each
    step must decrement by one under the ones labels. Its own wrap (0->5) is
    not used as evidence, since it only happens on a minute decrement and is
    filtered out with any other step that changes a higher digit.
    """
    ones: dict[int, Counter] = defaultdict(Counter)
    tens: dict[int, Counter] = defaultdict(Counter)
    positions: dict[int, set[int]] = defaultdict(set)
    for a, b in series:
        for i, c in enumerate(a):
            positions[c].add(i)
        if len(a) != 4 or len(b) != 4:
            continue
        if a[:-1] == b[:-1]:
            ones[a[-1]][b[-1]] += 1
        elif a[:-2] == b[:-2] and a[-1] != b[-1]:
            tens[a[-2]][b[-2]] += 1

    colons = [c for c, seen in positions.items() if seen == {1}]
    if len(colons) != 1:
        report(f"expected exactly one class to appear only in position 1, got {colons}")
        return None

    report(f"ones edges: {_evidence(ones)}")
    order = _chain(_modal_map(ones))
    if order is None:
        report("the descending successors are not one path over ten classes")
        return None
    labels = {c: str(9 - i) for i, c in enumerate(order)}
    report(f"descending path over ten classes: {order} = 9..0")

    tens_successor = _modal_map(tens)
    if not tens_successor:
        report("no tens step was observed; the path is unconfirmed")
        return None
    report(f"tens edges: {_evidence(tens)}")
    # Uses each digit's modal successor rather than every edge seen, since a
    # few frames disagree with identical neighbours on both sides.
    steps_taken = {}
    for a, b in tens_successor.items():
        if labels.get(a) is None or labels.get(b) is None:
            report(f"a tens class {a} is not in the ones path")
            return None
        steps_taken[labels[a]] = labels[b]

    # "0" is exempt: its wrap step moves the minute too and is filtered out
    # before being counted, so any edge recorded from 0 is noise.
    bad = sorted(
        (a, b) for a, b in steps_taken.items()
        if a != "0" and int(b) != int(a) - 1
    )
    if bad:
        report(f"the tens position contradicts the path, stepping {bad}")
        return None
    confirmed = sorted(steps_taken.items())
    if len(confirmed) < 5:
        report(f"only {len(confirmed)} tens digits stepped; too few to confirm")
        return None
    report(f"the tens position steps {confirmed}, confirming the path "
           f"independently ('0' exempt: {steps_taken.get('0', 'unobserved')}); "
           f"colon is cluster {colons[0]}")
    labels[colons[0]] = ":"
    return labels


def label_lives(series, report) -> dict[int, str] | None:
    """Label the "LIVES REMAINING" counters -- one digit, counting down.

    Has no carry or second position to anchor against, so it uses the game's
    own floor and ceiling instead: a team starts with one life per player
    (`scoreboard.ROW_COUNT`) and counts down to zero, so the path must be
    exactly ROW_COUNT + 1 classes long, labelled from its end.
    """
    successors: dict[int, Counter] = defaultdict(Counter)
    for a, b in series:
        if len(a) == len(b) == 1:
            successors[a[0]][b[0]] += 1

    report(f"edges: {_evidence(successors)}")
    expected = sb.ROW_COUNT + 1
    order = _chain(_modal_map(successors), expected)
    if order is None:
        report(f"the descending successors are not one path over {expected} "
               f"classes ({sb.ROW_COUNT} players a side, plus zero)")
        return None
    labels = {c: str(len(order) - 1 - i) for i, c in enumerate(order)}
    report(f"descending path over {expected} classes: {order} = "
           f"{sb.ROW_COUNT}..0, anchored on the roster being {sb.ROW_COUNT} a side")
    return labels


def build_labels(source: str, report) -> TemplateSet:
    """Whole-region templates for the mode titles and substrip titles."""
    info = video.probe(source)
    templates = TemplateSet("header_labels")
    seen: dict[str, list[np.ndarray]] = defaultdict(list)

    for path in FIXTURES:
        spec = json.loads(path.read_text())
        frame = video.frame_at(source, spec["timestamp"], info)
        if spec.get("mode") in hd.MODES:
            seen[spec["mode"]].append(tight_crop(hd.ink(hd.region(frame, "mode"))))
        if spec.get("substrip"):
            seen[spec["substrip"]].append(_substrip_mask(frame))
    for label, ts in EXTRA_SUBSTRIPS.items():
        seen[label].append(_substrip_mask(video.frame_at(source, ts, info)))

    for label, masks in sorted(seen.items()):
        templates.templates[label] = _majority(masks)
        report(f"  {label:24s} n={len(masks)} "
               f"shapes={sorted({m.shape for m in masks})}")
    return templates


def _substrip_band(frame: np.ndarray) -> np.ndarray:
    y0, y1 = hd.SUBSTRIP_Y
    x0, x1 = hd.SUBSTRIP_LABEL_X
    return frame[y0 : y1 + 1, x0 : x1 + 1]


def _substrip_mask(frame: np.ndarray) -> np.ndarray:
    return tight_crop(hd.ink(_substrip_band(frame)))


def _majority(masks: list[np.ndarray]) -> np.ndarray:
    modal = Counter(m.shape for m in masks).most_common(1)[0][0]
    stack = np.stack([m for m in masks if m.shape == modal])
    return stack.mean(axis=0) > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", default="data/strip_scoreboard_header.mp4")
    ap.add_argument("--timeline", default="data/state_timeline.json")
    ap.add_argument(
        "--source",
        default="@OpTicTexas vs @FaZeVegas ｜ Championship Weekend ｜ "
        "Monster Grand Finals [OtkL2tJOpIc].f399.mp4",
        help="the VOD, for the fixture frames the label templates come from",
    )
    ap.add_argument("--limit", type=int, default=None, help="frames, for smoke tests")
    args = ap.parse_args()

    strip = video.load_strip(ROOT / args.strip)
    spans = [
        (s["start"] + 1, s["end"] - 1)
        for s in json.loads((ROOT / args.timeline).read_text())
        if s["state"] == "live" and s["end"] - s["start"] > 2
    ]
    print(f"{strip.frames} strip frames at {strip.fps}fps, "
          f"{len(spans)} LIVE spans")

    # Label templates come first: the collection pass needs the substrip label
    # to know whether the lives band is open.
    print("\n=== label templates")
    labels_set = build_labels(args.source, print)
    labels_set.save(CONFIG / "header_labels.json")
    print(f"wrote {(CONFIG / 'header_labels.json').relative_to(ROOT)}\n")

    clusters, series, majority = collect(strip, spans, args.limit, labels_set)

    print("\n=== label templates, re-voted over every frame that drew them")
    majority.rebuild(labels_set, print)
    labels_set.save(CONFIG / "header_labels.json")
    print(f"wrote {(CONFIG / 'header_labels.json').relative_to(ROOT)}")

    problems: list[str] = []
    for name in BANDS:
        cluster = clusters[name]
        print(f"\n=== {name}: {len(cluster.reps)} clusters over "
              f"{sum(cluster.size(i) for i in range(len(cluster.reps)))} glyphs")
        for i in range(len(cluster.reps)):
            print(f"   c{i:<2d} n={cluster.size(i):6d} "
                  f"shapes={sorted(cluster.counts[i])}")

        transitions = [
            step
            for key, values in series.items()
            if key.startswith(f"{name}:")
            for step in steps(values)
        ]
        print(f"   {len(transitions)} contiguous changes")

        def report(line: str, name=name) -> None:
            print(f"   [{name}] {line}")

        labeller = {
            "score": label_counter,
            "lives": label_lives,
        }.get(name, label_countdown)
        labels = labeller(transitions, report)
        if labels is None or len(labels) != len(cluster.reps):
            problems.append(
                f"{name}: labelled {0 if labels is None else len(labels)} of "
                f"{len(cluster.reps)} clusters"
            )
            continue

        filename, _, canvas, _ = BANDS[name]
        atlas = GlyphAtlas(f"header_{name}", canvas=canvas)
        for index, char in labels.items():
            atlas.add(char, cluster.template(index))
        atlas.save(CONFIG / filename)
        print(f"   wrote {(CONFIG / filename).relative_to(ROOT)}: {atlas!r}")

    if problems:
        print("\nthe bootstrap did not close:")
        print("\n".join("  " + p for p in problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
