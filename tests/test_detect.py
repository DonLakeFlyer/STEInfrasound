"""Unit tests for the pure detection logic in ``infrasound_detect``."""

import numpy as np
import pytest

from infrasound_detect import (
    DetectionConfig,
    DetectionStateMachine,
    DetectionArgError,
    find_runs,
    strongest_group_column,
    parse_detection_args,
)

N_BINS = 101  # matches the 0-50 Hz band at 0.5 Hz resolution


def _cfg(**kw):
    """Small, explicit config so the state machine is easy to drive in tests."""
    defaults = dict(
        threshold_db=16.0,
        filter_fp=True,
        min_consec_bins=5,
        start_frames=3,
        end_frames=1,
        max_columns=5,
        max_group_bins=30,
        broadband_fraction=0.40,
    )
    defaults.update(kw)
    return DetectionConfig(**defaults)


def _narrow(level=20.0, start=40, width=6, n=N_BINS):
    col = np.zeros(n)
    col[start:start + width] = level
    return col


def _wide(level=20.0, width=35, n=N_BINS):
    col = np.zeros(n)
    col[10:10 + width] = level
    return col


def _broadband(level=20.0, n=N_BINS):
    col = np.zeros(n)
    col[::2] = level  # ~50% of bins hot, scattered
    return col


def _quiet(n=N_BINS):
    return np.zeros(n)


# -----------------------------
# find_runs
# -----------------------------
def test_find_runs_basic():
    mask = np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)
    assert find_runs(mask, 2) == [(1, 4)]


def test_find_runs_min_len_excludes_short():
    mask = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    assert find_runs(mask, 3) == [(3, 6)]


def test_find_runs_trailing_run():
    mask = np.array([0, 0, 1, 1, 1], dtype=bool)
    assert find_runs(mask, 3) == [(2, 5)]


# -----------------------------
# strongest_group_column
# -----------------------------
def test_narrow_group_is_active():
    col, active, broadband = strongest_group_column(_narrow(), _cfg())
    assert active and not broadband
    assert np.allclose(col[40:46], 20.0)
    assert np.all(np.isnan(col[:40]))


def test_short_group_is_not_active():
    # 3 bins < min_consec_bins (5)
    col, active, broadband = strongest_group_column(_narrow(width=3), _cfg())
    assert not active and not broadband
    assert np.all(np.isnan(col))


def test_broadband_fraction_rejected():
    col, active, broadband = strongest_group_column(_broadband(), _cfg())
    assert not active and broadband


def test_wide_single_clump_rejected_by_filter_b():
    # 35 contiguous bins < broadband fraction (40) but > max_group_bins (30).
    col, active, broadband = strongest_group_column(_wide(width=35), _cfg())
    assert not active and broadband


def test_filters_disabled_allows_wide_clump():
    col, active, broadband = strongest_group_column(
        _wide(width=35), _cfg(filter_fp=False)
    )
    assert active and not broadband


def test_strongest_group_selected_among_many():
    col = np.zeros(N_BINS)
    col[10:16] = 18.0   # weaker group
    col[60:66] = 25.0   # stronger group
    out, active, broadband = strongest_group_column(col, _cfg())
    assert active
    assert np.allclose(out[60:66], 25.0)
    assert np.all(np.isnan(out[10:16]))


# -----------------------------
# DetectionStateMachine
# -----------------------------
def test_event_starts_after_start_frames():
    sm = DetectionStateMachine(_cfg(start_frames=3))
    outcomes = [sm.process(_narrow()) for _ in range(3)]
    assert sm.state == 'RECORDING'
    assert outcomes[-1].event_started
    assert len(sm.columns) == 3  # seeded with the full start-gate window


def test_start_gate_resets_on_gap():
    sm = DetectionStateMachine(_cfg(start_frames=3))
    sm.process(_narrow())
    sm.process(_narrow())
    sm.process(_quiet())            # gap resets the gate
    assert sm.state == 'IDLE'
    assert sm.above_count == 0
    sm.process(_narrow())
    sm.process(_narrow())
    out = sm.process(_narrow())     # now three in a row
    assert sm.state == 'RECORDING'
    assert out.event_started


def test_quiet_frame_ends_event():
    sm = DetectionStateMachine(_cfg(start_frames=3, end_frames=1))
    for _ in range(3):
        sm.process(_narrow())
    assert sm.state == 'RECORDING'
    out = sm.process(_quiet())
    assert out.event_ended
    assert sm.state == 'IDLE'
    assert sm.frozen
    assert sm.below_count == 0       # state fully reset on termination
    assert len(sm.columns) == 3     # trailing blank discarded


def test_broadband_frame_does_not_end_event():
    sm = DetectionStateMachine(_cfg(start_frames=3, end_frames=1))
    for _ in range(3):
        sm.process(_narrow())
    out = sm.process(_broadband())
    assert not out.event_ended
    assert sm.state == 'RECORDING'
    assert sm.below_count == 0       # broadband never counts toward the end
    assert len(sm.pending) == 1      # held as an interior gap


def test_sustained_broadband_eventually_ends_event():
    sm = DetectionStateMachine(_cfg(start_frames=3, max_columns=5))
    for _ in range(3):
        sm.process(_narrow())
    assert sm.state == 'RECORDING'
    # A gust that never clears: the held-back run cannot grow past max_columns.
    out = None
    for _ in range(5):
        out = sm.process(_broadband())
    assert out.event_ended
    assert sm.state == 'IDLE'
    assert sm.frozen
    assert sm.pending == []          # held-back run discarded on termination


def test_interior_gap_is_committed_on_next_active():
    sm = DetectionStateMachine(_cfg(start_frames=3, end_frames=2, max_columns=50))
    for _ in range(3):
        sm.process(_narrow())
    sm.process(_quiet())             # below_count=1, not yet ended (end_frames=2)
    assert sm.state == 'RECORDING'
    assert len(sm.pending) == 1
    sm.process(_narrow())            # commits the held gap + this frame
    assert sm.pending == []
    assert len(sm.columns) == 5      # 3 + held gap + new active


def test_columns_capped_at_max_columns():
    sm = DetectionStateMachine(_cfg(start_frames=3, max_columns=5))
    for _ in range(3):
        sm.process(_narrow())        # columns == 3
    for _ in range(4):
        sm.process(_narrow())        # would grow to 7, capped at 5
    assert len(sm.columns) == 5


# -----------------------------
# parse_detection_args
# -----------------------------
def test_parse_defaults():
    args = parse_detection_args(['prog'])
    assert not args.test_mode
    assert args.filter_fp
    assert args.show_bars
    assert args.threshold_db == 16.0
    assert not args.file_mode
    assert args.file_path is None


def test_parse_flags():
    args = parse_detection_args(['prog', '--test', '--no-fp-filter', '--no-bars'])
    assert args.test_mode
    assert not args.filter_fp
    assert not args.show_bars


def test_parse_threshold():
    args = parse_detection_args(['prog', '--threshold', '10'])
    assert args.threshold_db == 10.0


def test_parse_threshold_missing_value():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--threshold'])


def test_parse_threshold_bad_value():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--threshold', 'abc'])


def test_parse_file_flag():
    args = parse_detection_args(['prog', '--file', 'rec.wav'])
    assert args.file_mode
    assert args.file_path == 'rec.wav'


def test_parse_file_flag_missing_value():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--file'])


def test_parse_bare_path():
    args = parse_detection_args(['prog', 'rec.wav'])
    assert args.file_mode
    assert args.file_path == 'rec.wav'


def test_parse_threshold_value_not_mistaken_for_path():
    args = parse_detection_args(['prog', '--threshold', '10', 'rec.wav'])
    assert args.threshold_db == 10.0
    assert args.file_path == 'rec.wav'


def test_parse_threshold_value_alone_is_not_a_path():
    args = parse_detection_args(['prog', '--threshold', '10'])
    assert not args.file_mode
    assert args.file_path is None


# -----------------------------
# parse_detection_args: --speed
# -----------------------------
def test_parse_speed_default():
    args = parse_detection_args(['prog'])
    assert args.speed == 1


def test_parse_speed_value():
    args = parse_detection_args(['prog', '--file', 'rec.wav', '--speed', '8'])
    assert args.speed == 8


def test_parse_speed_missing_value():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--speed'])


def test_parse_speed_bad_value():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--speed', 'fast'])


def test_parse_speed_below_one_rejected():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--speed', '0'])


def test_parse_speed_non_integer_rejected():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--speed', '1.5'])


def test_parse_speed_value_not_mistaken_for_path():
    args = parse_detection_args(['prog', '--speed', '8', 'rec.wav'])
    assert args.speed == 8
    assert args.file_path == 'rec.wav'


def test_parse_test_with_file_flag_rejected():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--test', '--file', 'rec.wav'])


def test_parse_test_with_bare_path_rejected():
    with pytest.raises(DetectionArgError):
        parse_detection_args(['prog', '--test', 'rec.wav'])

