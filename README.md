# cdlvision

Reads structured data out of competitive Call of Duty broadcast VODs — box
scores, the live scoreboard, the centre header, the killfeed, and the
spectated player's panel — directly from the video, with no manual
transcription and no model anywhere in the pipeline. On top of those readings
it derives what actually happened: where each map starts and ends, and how
each map divides into rounds, hills or halves.

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
The clock band is really three counters sharing one position: the game clock,
the bomb timer that replaces it on a plant, and a red seconds-and-tenths counter
below 0:30 that reads `28.5` rather than `0:28`. All three are read; the last
one is why time expiry is observable at all.

**The killfeed** — who killed whom, with what weapon, chained together
across the frames a kill fades over. Recovers 96.7% of kills and 95.7% of
deaths against the box scores, and the shortfall is mostly explained (lost
video frames, suicides) rather than unexplained.

**The spectated player's panel** (bottom-right HUD) — name, K/D, streak,
team colour. Checked directly against what the live scoreboard says about
the same player at the same instant: 99.9% agreement across 24,080 sampled
frames.

**The panel's HP, as death events** — a fall to 0 while the panel stays on the
same player is a death, and the killfeed should draw a row naming that player.
70.8% of falls pair with one, against a 0.7–4.1% floor from the same falls
displaced in time, and the peak sits at +0.167s — exactly one frame, with
nothing pairing at any negative offset, since a killfeed row cannot precede the
death it reports. Damage the player survived pairs at 1.2%.

**Map boundaries** — where one map ends and the next begins, found three
independent ways (scoreboard reset, pip advance, map score reset) that all
agree to the second across all seven maps in this VOD.

**Rounds, hills and halves** — the first output here that is game semantics
rather than screen contents. 11 rounds on each Search & Destroy map from three
different header fields, 12 hills per Hardpoint map off the rotation timer, and
2 halves per Overload map. Every round gets an outcome: 19 wipes, 5 time
expiries, 3 detonations, and 6 plants whose ending the broadcast never draws.
Overload turning out *not* to be round-based was measured, against the
assumption going in.


## Known limits

Stated rather than smoothed over, because a characterised error of known size is
worth more than an uncharacterised zero:

- **Everything is validated against one VOD.** The atlases, geometry invariants
  and thresholds have never seen a second broadcast. The design intends a layout
  change to fail loudly rather than emit wrong numbers, but that is a claim
  awaiting a second VOD.
- **Two weapons draw the same killfeed silhouette**, so one cluster is named
  `JÄGER 45 / MPC-25` rather than inventing a distinction the pixels don't carry.
- **A defuse can't be told from the planters being killed**, because that needs
  to know which side planted and the overlay never draws it. Left unclassified.
- Small enumerated residuals: 9 duplicate killfeed events across 35 rounds (chain
  fragmentation, sitting on the minimum-chain-length floor), 2 of 96 scoreboard
  totals off by one at a map's final kill, 0.1% of clock samples unread, and one
  map's final-round winner undecided because the closing pips are never drawn.

None of these are tuned away. Where a threshold and a game rule disagree, the
rule wins and the disagreement gets recorded.

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
  hp.py               deaths off the panel's HP, matched against the killfeed
  maps.py             map boundary derivation
  rounds.py           rounds / hills / halves, and the killfeed's ordering
  validate.py         structural / game-rule consistency checks
  progress.py         progress reporting for long jobs
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
uv run python tools/validate_hp.py               # panel HP vs killfeed deaths
uv run python tools/validate_rounds.py           # rounds vs the game's arithmetic
```

The validators run off the committed scan caches, so `uv run pytest` and every
number above reproduce from a clean checkout with no video present.

Source footage is not committed — it's broadcast material with no
redistribution rights, and it's large. Only derived data (scan caches, hand
labels, calibration atlases) is tracked, which is enough to reproduce every
result above without the video.
