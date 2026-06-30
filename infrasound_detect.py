"""Pure, display-independent detection logic for ``infrasound_detections.py``.

The detection algorithm (frequency-clump grouping, false-positive filtering, and
the start/stop state machine) is separated from the app here so it can be
unit-tested without a matplotlib display or an audio device. ``infrasound_detections.py``
imports these helpers and only handles argv wiring, plotting, and the animation
loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from infrasound_core import UPDATE_RATE, FREQ_RESOLUTION

# -----------------------------
# DETECTION TUNING
# -----------------------------
# A detection starts when a clump of power is:
#   * at least ``threshold_db`` above the noise floor (red-bar level), and
#   * more than MIN_GROUP_HZ wide in frequency, and
#   * sustained for at least MIN_DURATION_S.
DEFAULT_THRESHOLD_DB = 16.0   # dB above noise floor (matches the bar display's red tier)
MIN_GROUP_HZ = 2.0            # a clump must be wider than this to count
MIN_DURATION_S = 1.5          # a clump must persist at least this long to start an event
MAX_EVENT_S = 20.0            # cap a single event's chart width (seconds)

# False-positive filtering (optional, default ON). Real rumble harmonics are
# narrow and tonal; wind gusts and broadband clicks smear energy across a wide
# chunk of the band. These two conservative checks reject only such obvious
# non-rumble frames; narrow rumble harmonics never trip them.
MAX_GROUP_HZ = 15.0           # a single qualifying clump wider than this is a smear
BROADBAND_FRACTION = 0.40     # if this fraction of the whole band is hot -> broadband


@dataclass
class DetectionConfig:
    """Detection thresholds and the bin/frame counts derived from them.

    The derived counts default to values computed from the shared engine
    constants (``UPDATE_RATE``/``FREQ_RESOLUTION``) but may be overridden
    explicitly, which keeps the state machine easy to unit-test.
    """

    threshold_db: float = DEFAULT_THRESHOLD_DB
    filter_fp: bool = True
    min_consec_bins: Optional[int] = None
    start_frames: Optional[int] = None
    end_frames: int = 1
    max_columns: Optional[int] = None
    max_group_bins: Optional[int] = None
    broadband_fraction: float = BROADBAND_FRACTION

    def __post_init__(self):
        # "More than MIN_GROUP_HZ wide" -> strictly more bins than it spans.
        if self.min_consec_bins is None:
            self.min_consec_bins = int(MIN_GROUP_HZ / FREQ_RESOLUTION) + 1
        # Start gate floor = MIN_DURATION_S, rounded UP to a whole frame so the
        # reported minimum length never falls below MIN_DURATION_S at any rate.
        if self.start_frames is None:
            self.start_frames = math.ceil(MIN_DURATION_S * UPDATE_RATE)
        if self.max_columns is None:
            self.max_columns = int(MAX_EVENT_S * UPDATE_RATE)
        if self.max_group_bins is None:
            self.max_group_bins = int(MAX_GROUP_HZ / FREQ_RESOLUTION)


def find_runs(mask, min_len):
    """Return ``(start, stop)`` index pairs for runs of >= *min_len* True values."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


def strongest_group_column(relative_db, config):
    """Build a heatmap column showing only the strongest valid group.

    A valid group is a run of >= ``config.min_consec_bins`` adjacent bins above
    threshold. Among all valid groups in this frame, only the one containing the
    highest single-bin dB is rendered; every other bin is NaN (blank).

    Returns ``(column, active, broadband)``:
        active    -- True if a narrow, tonal qualifying group was found.
        broadband -- True if false-positive filtering is on and the frame looks
                     like wind/broadband noise (too much of the band is hot, or
                     the strongest clump is implausibly wide for a rumble).
    A broadband frame is never ``active`` (so it cannot *start* a detection), but
    the caller treats it as an interior gap so it does not *end* one either.
    """
    relative_db = np.asarray(relative_db, dtype=float)
    mask = relative_db >= config.threshold_db
    n_bins = len(relative_db)
    col = np.full(n_bins, np.nan, dtype=float)

    # Filter A: energy smeared across a large fraction of the band == broadband.
    if config.filter_fp and np.count_nonzero(mask) >= config.broadband_fraction * n_bins:
        return col, False, True

    runs = find_runs(mask, config.min_consec_bins)
    if not runs:
        return col, False, False

    # Strength of a group = its peak bin dB; pick the strongest group.
    best = max(runs, key=lambda r: float(np.max(relative_db[r[0]:r[1]])))
    s, e = best

    # Filter B: a single qualifying clump wider than a plausible rumble harmonic
    # is a smear, not a tone.
    if config.filter_fp and (e - s) > config.max_group_bins:
        return col, False, True

    col[s:e] = relative_db[s:e]
    return col, True, False


@dataclass
class FrameOutcome:
    """What changed after feeding one frame to the state machine."""

    redraw: bool = False
    event_started: bool = False
    event_ended: bool = False


class DetectionStateMachine:
    """Start/stop state machine that turns per-frame columns into event charts.

    The machine waits in ``'IDLE'`` until ``config.start_frames`` consecutive
    active frames confirm an event, then records columns in ``'RECORDING'``
    until ``config.end_frames`` quiet frames end it. Broadband frames mid-event
    are held as interior gaps and never end an event; trailing blanks are
    discarded so the recorded length excludes the end-detection delay.
    """

    def __init__(self, config):
        self.config = config
        self.state = 'IDLE'
        self.above_count = 0      # consecutive active frames (while IDLE)
        self.below_count = 0      # consecutive quiet frames (while RECORDING)
        self.columns = []         # committed per-bin columns for the current/last event
        self.pending = []         # below-threshold columns held back from the chart
        self.start_buffer = []    # active columns accumulated while passing the start gate
        self.frozen = False       # True when a completed event chart is held on screen

    def process(self, relative_db):
        """Feed one frame's ``relative_db`` and advance the state machine."""
        cfg = self.config
        _col, active, broadband = strongest_group_column(relative_db, cfg)
        outcome = FrameOutcome()

        if self.state == 'IDLE':
            if active:
                # Accumulate qualifying frames so the chart (and reported length)
                # includes the full start-gate window, not just the trigger frame.
                self.above_count += 1
                self.start_buffer.append(_col)
            else:
                # Wind/broadband or quiet frames do not start an event.
                self.above_count = 0
                self.start_buffer = []
            if self.above_count >= cfg.start_frames:
                # Start a fresh chart, seeded with every frame that passed the gate.
                self.state = 'RECORDING'
                self.below_count = 0
                self.columns = self.start_buffer[:cfg.max_columns]
                self.pending = []
                self.start_buffer = []
                self.frozen = False
                outcome.event_started = True
                outcome.redraw = True
        else:  # RECORDING
            if active:
                # Commit any held-back blank columns (interior gaps) plus this one.
                for held in self.pending:
                    if len(self.columns) < cfg.max_columns:
                        self.columns.append(held)
                self.pending = []
                if len(self.columns) < cfg.max_columns:
                    self.columns.append(_col)
                self.below_count = 0
                outcome.redraw = True
            elif broadband:
                # A broadband frame (e.g. a wind gust over an ongoing rumble) must
                # not end the event. Hold it as an interior gap; below_count is NOT
                # incremented so a brief gust lets the event survive. But a gust
                # that never clears would record forever, so cap the held-back run
                # at the event width and end the event once it is exceeded.
                self.pending.append(_col)
                if len(self.pending) >= cfg.max_columns:
                    self.pending = []
                    self.state = 'IDLE'
                    self.above_count = 0
                    self.below_count = 0
                    self.frozen = True
                    outcome.redraw = True
                    outcome.event_ended = True
            else:
                # Genuinely quiet: hold the blank back and count toward the end.
                self.pending.append(_col)
                self.below_count += 1
                if self.below_count >= cfg.end_frames:
                    # Event over: discard held-back trailing blanks so the chart
                    # ends on the last active frame.
                    self.pending = []
                    self.state = 'IDLE'
                    self.above_count = 0
                    self.below_count = 0
                    self.frozen = True
                    outcome.redraw = True
                    outcome.event_ended = True

        return outcome


# -----------------------------
# ARGUMENT PARSING
# -----------------------------
class DetectionArgError(ValueError):
    """Raised on a malformed command line for the detection app."""


@dataclass
class DetectionArgs:
    test_mode: bool
    filter_fp: bool
    show_bars: bool
    threshold_db: float
    file_mode: bool
    file_path: Optional[str]
    speed: int = 1


def parse_detection_args(argv):
    """Parse the detection app's command line from an ``argv`` list.

    ``argv`` follows ``sys.argv`` conventions (``argv[0]`` is the program name).
    Raises :class:`DetectionArgError` on a malformed command line.
    """
    test_mode = '--test' in argv
    filter_fp = '--no-fp-filter' not in argv   # false-positive filtering on by default
    show_bars = '--no-bars' not in argv        # raw bar spectrum shown by default
    threshold_db = DEFAULT_THRESHOLD_DB
    speed = 1

    thr_idx = argv.index('--threshold') if '--threshold' in argv else -1
    if thr_idx != -1:
        if thr_idx + 1 >= len(argv):
            raise DetectionArgError("--threshold requires a following number.")
        try:
            threshold_db = float(argv[thr_idx + 1])
        except ValueError:
            raise DetectionArgError("--threshold value must be a number.") from None

    spd_idx = argv.index('--speed') if '--speed' in argv else -1
    if spd_idx != -1:
        if spd_idx + 1 >= len(argv):
            raise DetectionArgError("--speed requires a following number.")
        try:
            speed = int(argv[spd_idx + 1])
        except ValueError:
            raise DetectionArgError("--speed value must be a whole number.") from None
        if speed < 1:
            raise DetectionArgError("--speed must be >= 1.")

    file_mode = False
    file_path = None
    file_idx = argv.index('--file') if '--file' in argv else -1
    if file_idx != -1:
        if file_idx + 1 >= len(argv):
            raise DetectionArgError("--file requires a following path argument.")
        file_mode = True
        file_path = argv[file_idx + 1]
    else:
        # Accept a bare path argument; skip any value consumed by --threshold or
        # --speed so it isn't mistaken for a path.
        skip = {idx + 1 for idx in (thr_idx, spd_idx) if idx != -1}
        for i, arg in enumerate(argv[1:], start=1):
            if i in skip:
                continue
            if not arg.startswith('-'):
                file_mode = True
                file_path = arg
                break

    if test_mode and file_mode:
        raise DetectionArgError(
            "--test cannot be combined with --file or a file path.")

    return DetectionArgs(test_mode, filter_fp, show_bars, threshold_db,
                         file_mode, file_path, speed)


if __name__ == '__main__':
    import sys
    print(
        "infrasound_detect.py is a shared library module, not a runnable app.\n"
        "Run the detection display instead:\n"
        "  python infrasound_detections.py [PATH|--file PATH|--test]",
        file=sys.stderr,
    )
    sys.exit(2)
