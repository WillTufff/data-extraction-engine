# cdlvision

Reads structured data out of competitive Call of Duty broadcast VODs — box
scores, the live scoreboard, the centre header, the killfeed, and the
spectated player's panel — directly from the video, with no manual
transcription and no model anywhere in the pipeline.

This is a companion project to
[cdlhub](https://github.com/WillTufff/COD-DataAnalysis), which does era
adjustment, rating systems, and round-level modelling on data that already
exists (the public `cwl-data` archive). cdlvision exists to produce data that
doesn't: kill-by-kill, round-by-round state pulled straight from broadcast
footage, for feeding into that same kind of modelling later.

## The core idea

The broadcast overlay is machine-rendered: fixed positions, a fixed font,
flat colours, no perspective or lighting to account for. That means
recognition can be exact-shape matching against stored bitmaps instead of
OCR. Nothing here is a trained model — every read is a pixel comparison
against a template built from the broadcast's own frames, so it's
deterministic and free to re-run.

Two things fell out of actually measuring the pixels, both counter to the
naive approach:

- **Binarise before comparing.** Two renderings of the same character agree
  on only 11% of raw greyscale pixels — that's compression noise — but 99.9%
  once both are thresholded to black-and-white. Never compare raw pixels.
- **Never trust colour on its own.** Sponsor graphics and bright gameplay
  can match any colour rule you write. Colour narrows down where to look;
  a bitmap match is what actually decides.

## What it reads, and how well

**Box scores** (the post-map summary screen) — 1,415 glyphs across six
scores in three modes, zero below the confidence floor. Cross-checked
against the game's own arithmetic (a team's kills must equal the other
team's deaths minus suicides) rather than hand transcription alone.

**Live/replay/other state** — the classifier that everything else gates on,
since a replay re-shows kills that already happened and a killfeed reader
that can't tell replay from live will double-count. Validated against all
six box-score screens landing in the right segment, and 23/23 hand-labelled
frames, across the full 2h43m VOD.

**The live scoreboard** — both team panels, every stat, for every live
frame, off 6,707 run-length records covering the whole VOD. 94 of 96
per-player totals match the box scores exactly (the two misses are off by
one kill at a map's last second).

**The centre header** — both scores, the clock, the map pips, the mode, and
the round-count fields for Search & Destroy. Both team scores are monotone
across all 443 increments recorded — zero decreases anywhere in the VOD.

**The killfeed** — who killed whom, with what weapon, chained together
across the frames a kill fades over. Recovers 96.7% of kills and 95.7% of
deaths against the box scores, and the shortfall is mostly explained (lost
video frames, suicides) rather than unexplained.

**The spectated player's panel** (bottom-right HUD) — name, HP, K/D, streak,
team colour. Checked directly against what the live scoreboard says about
the same player at the same instant: 99.9% agreement across 24,080 sampled
frames.

**Map boundaries** — where one map ends and the next begins, found three
independent ways (scoreboard reset, pip advance, map score reset) that all
agree to the second across all seven maps in this VOD.


## Layout

```
src/cdlvision/
  video.py            frame access — single seek, or one sampled decode pass
  glyphs.py           per-character atlas + whole-region template matching
  boxscore.py         box-score grid detection, mode detection, parsing
  state.py            live/replay/other classification and smoothing
  scoreboard.py        live scoreboard geometry + reader
  scoreboard_runs.py   scan cache loading, roster and foreign-frame filtering
  header.py           centre header (scores, clock, pips, mode, S&D lives)
  killfeed.py         killfeed row reading and chaining into kill events
  events.py           chains killfeed rows into entries
  names.py            closed-set player name matching for the killfeed
  spectated.py        the bottom-right panel and the weapon name
  maps.py             map boundary derivation
  validate.py         structural / game-rule consistency checks
  config/             atlases and templates, all JSON, human-readable
tools/          build_* (calibration), scan_*/probe_* (scanning the VOD),
                validate_* (checking results against game rules)
tests/          fixtures/ holds hand-transcribed ground truth and cached
                whole-VOD scan output, so the tests don't need the video
```

Geometry (grid positions, column widths, panel boundaries) is always
detected from the frame and checked against expected invariants — never
hardcoded — so a layout change in a future broadcast fails loudly instead of
silently producing wrong numbers.

## Usage

```sh
uv sync --extra dev
uv run pytest

uv run python tools/find_boxscores.py <video> --step 10
uv run python tools/scan_scoreboard.py           # whole VOD -> scoreboard runs
uv run python tools/scan_header.py               # whole VOD -> header runs
uv run python tools/scan_killfeed.py <video>      # whole VOD -> killfeed rows
uv run python tools/scan_spectated.py            # whole VOD -> spectated runs
uv run python tools/reconcile_killfeed.py        # check killfeed against box scores
uv run python tools/validate_scoreboard.py --strict
uv run python tools/validate_header.py
uv run python tools/validate_spectated.py
```

Source footage is not committed — it's broadcast material with no
redistribution rights, and it's large. Only derived data (scan caches, hand
labels, calibration atlases) is tracked, which is enough to reproduce every
result above without the video.
