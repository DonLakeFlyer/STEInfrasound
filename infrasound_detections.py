"""Event-triggered infrasound detection display.

Unlike the scrolling bar display (``infrasound.py``), this app draws one chart
*per detection event*. A detection starts when any frequency bin rises above a
dB threshold (relative to the adaptive noise floor) for a short confirmation
window, and the chart's X axis then grows for the full duration of that event.
When the signal falls back below threshold (after a brief hangover) the chart
freezes; the next threshold crossing clears it and starts a fresh chart.

Axes:
    X = elapsed time within the current detection event (not real-time scrolling)
    Y = frequency (Hz)
    Greyscale intensity = dB above noise floor (threshold = light, louder = darker)

Shares the audio/FFT backend (``infrasound_core``) and the splash/error/menu
chrome (``infrasound_ui``) with ``infrasound.py``.

Usage:
    python infrasound_detections.py                  # live microphone
    python infrasound_detections.py --test           # synthetic rumble, no mic
    python infrasound_detections.py --file REC.wav   # replay and analyze a WAV
    python infrasound_detections.py --threshold 10   # set detection level (dB)
    python infrasound_detections.py --file REC.wav --speed 8   # silent 8x fast-scan
"""

import signal
import sys
import time

import numpy as np

# Import UI before pyplot so the matplotlib backend is selected first.
import infrasound_ui as ui
from infrasound_ui import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DPI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from infrasound_core import (
    InfrasoundEngine,
    log,
    IS_RPI,
    UPDATE_RATE,
    LOW_HZ,
    HIGH_HZ,
    FREQ_RESOLUTION,
    DB_MAX,
    SAMPLE_RATE,
)
from infrasound_bar import BarChartView
from infrasound_detect import (
    DetectionConfig,
    DetectionStateMachine,
    DetectionArgError,
    parse_detection_args,
    MAX_EVENT_S,
)

if IS_RPI:
    print("Running on Raspberry Pi")
else:
    print("Running on desktop/other platform")

# -----------------------------
# MODE SELECTION
# -----------------------------
# Detection thresholds, the false-positive filters, and the start/stop state
# machine all live in infrasound_detect.py so they can be unit-tested without a
# display. This file only parses the command line, builds the engine, and draws.
try:
    _args = parse_detection_args(sys.argv)
except DetectionArgError as exc:
    print(f"Usage error: {exc}", file=sys.stderr)
    sys.exit(2)

TEST_MODE = _args.test_mode
THRESHOLD_DB = _args.threshold_db
FILE_MODE = _args.file_mode
FILE_PATH = _args.file_path
SHOW_BARS = _args.show_bars

# Faster-than-real-time scan (file mode only). At speed > 1 the audio output is
# disabled and the animation loop steps the detector SPEED times per tick, so
# every 0.1 s hop is analyzed the same way as normal playback.
SPEED = _args.speed
SILENT_FAST = FILE_MODE and SPEED > 1
STEPS_PER_TICK = SPEED if SILENT_FAST else 1
HOP = int(SAMPLE_RATE / UPDATE_RATE)
if SPEED > 1 and not FILE_MODE:
    log("Note: --speed only applies to file playback; ignoring.")

config = DetectionConfig(threshold_db=_args.threshold_db, filter_fp=_args.filter_fp)
sm = DetectionStateMachine(config)

if FILE_MODE:
    engine = InfrasoundEngine(mode='file', file_path=FILE_PATH)
elif TEST_MODE:
    engine = InfrasoundEngine(mode='test')
else:
    engine = InfrasoundEngine(mode='mic')


# -----------------------------
# CLEAN SHUTDOWN
# -----------------------------
def signal_handler(sig, frame):
    log(f"Received signal {sig}, shutting down...")
    engine.stop()
    plt.close('all')
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# -----------------------------
# START AUDIO
# -----------------------------
if FILE_MODE:
    try:
        engine.start(play_audio=not SILENT_FAST)
    except Exception as exc:
        log(f"ERROR starting file mode: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
elif TEST_MODE:
    engine.start()
elif not ui.show_splash_and_connect(engine):
    print(f"Error opening audio stream: {engine.audio_error}")
    ui.show_error_screen(engine.audio_error if engine.audio_error else "Connection timeout")
    sys.exit(1)


# -----------------------------
# SETUP PLOT
# -----------------------------
# With the bar view enabled (default) the raw spectrum sits on top and the
# detection heatmap below it, so raw data and detections are visible together.
if SHOW_BARS:
    fig, (ax_bar, ax) = plt.subplots(
        2, 1, figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI
    )
    bar_view = BarChartView(ax_bar, engine.infra_freqs)
else:
    fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)
    ax_bar = None
    bar_view = None
fig.patch.set_facecolor('white')

# Greyscale: threshold maps to white (light), louder maps to black (dark).
# Bins below threshold are NaN and render as blank/white.
cmap = plt.get_cmap('Greys').copy()
cmap.set_bad('white')

# Power/exit menu on click (works on either chart).
ui.attach_power_menu(fig, [a for a in (ax, ax_bar) if a is not None])


# -----------------------------
# DETECTION STATE
# -----------------------------
# The state machine itself (state, above/below counts, column buffers) lives in
# the DetectionStateMachine `sm` created above. These globals only track the
# display annotations that the renderer reads.
playback_str = ''  # "mm:ss / mm:ss" file-replay position (empty unless file mode)
event_when = ''   # when the current/last detection began (clock time or file mm:ss)


def _fmt_mmss(seconds):
    """Format a number of seconds as mm:ss."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def _decorate():
    ax.set_facecolor('white')
    ax.set_xlabel("Time in detection (s)", fontsize=7)
    ax.set_ylabel("Frequency (Hz)", fontsize=7)
    ax.tick_params(labelsize=6)

    suffix = f"  |  {playback_str}" if playback_str else ""
    when = f"  |  {event_when}" if event_when else ""
    if sm.state == 'RECORDING':
        elapsed = len(sm.columns) / UPDATE_RATE
        ax.set_title(f"DETECTION  |  {elapsed:.1f}s{when}  (≥ {THRESHOLD_DB:.0f} dB){suffix}",
                     fontsize=8, color='#aa0000')
    elif sm.frozen and sm.columns:
        elapsed = len(sm.columns) / UPDATE_RATE
        ax.set_title(f"Last detection: {elapsed:.1f}s{when}  (≥ {THRESHOLD_DB:.0f} dB){suffix}",
                     fontsize=8, color='black')
    else:
        ax.set_title(f"Listening…  (threshold ≥ {THRESHOLD_DB:.0f} dB){suffix}",
                     fontsize=8, color='gray')


def _render():
    ax.clear()
    if sm.columns:
        img = np.column_stack(sm.columns)  # (n_freq, n_cols)
        elapsed = max(len(sm.columns) / UPDATE_RATE, 1.0 / UPDATE_RATE)
        # Extend the y-extent by half a bin on each side so the 101 rows have
        # centers exactly on the FFT bin frequencies (0, 0.5, ... HIGH_HZ).
        y_bottom = LOW_HZ - FREQ_RESOLUTION / 2
        y_top = HIGH_HZ + FREQ_RESOLUTION / 2
        ax.imshow(
            img,
            origin='lower',
            aspect='auto',
            extent=[0.0, elapsed, y_bottom, y_top],
            cmap=cmap,
            vmin=THRESHOLD_DB,
            vmax=DB_MAX,
        )
        # Zoom the frequency axis to the band that actually contains data.
        valid_rows = np.where(np.any(~np.isnan(img), axis=1))[0]
        if valid_rows.size:
            step = FREQ_RESOLUTION
            pad = step  # one-bin margin so the data isn't flush against the edge
            y_lo = max(LOW_HZ, LOW_HZ + valid_rows[0] * step - pad)
            y_hi = min(HIGH_HZ, LOW_HZ + valid_rows[-1] * step + pad)
            ax.set_ylim(y_lo, y_hi)
        else:
            ax.set_ylim(LOW_HZ, HIGH_HZ)
    else:
        ax.set_xlim(0, MAX_EVENT_S)
        ax.set_ylim(LOW_HZ, HIGH_HZ)
    _decorate()


def _event_start_label(result):
    """Describe when the current detection began.

    The event actually started (start_frames - 1) frames before this triggering
    frame (the start-gate window). In file-replay mode this is the file position
    (``@ mm:ss``); for live/mic/test it is the wall-clock time (``@ HH:MM:SS``).
    """
    onset_offset_s = (config.start_frames - 1) / UPDATE_RATE
    if result.total_samples:
        onset_s = max(0.0, result.file_position / SAMPLE_RATE - onset_offset_s)
        return f"@ {_fmt_mmss(onset_s)}"
    return "@ " + time.strftime('%H:%M:%S', time.localtime(time.time() - onset_offset_s))


# -----------------------------
# UPDATE FUNCTION
# -----------------------------
def update_plot(frame):
    global playback_str, event_when

    # Normally one analysis step per tick (driven by real-time audio in file
    # mode, or the test/mic source). In silent fast-scan we step the file by a
    # full hop STEPS_PER_TICK times so every 0.1 s window is still analyzed.
    redraw = False
    result = None
    for _ in range(STEPS_PER_TICK):
        if SILENT_FAST:
            engine.advance_file(HOP)
        result = engine.process_frame()
        outcome = sm.process(result.relative_db)
        if outcome.event_started:
            event_when = _event_start_label(result)
        if outcome.event_ended:
            log(f"Detection ended: {len(sm.columns) / UPDATE_RATE:.1f}s")
        redraw = redraw or outcome.redraw

    # The bar spectrum and playback timer reflect the latest analyzed window.
    if result is not None and bar_view is not None:
        bar_view.update(result)
    if result is not None and result.total_samples:
        rate_tag = f"  {SPEED}\u00d7" if SILENT_FAST else ""
        playback_str = (f"{_fmt_mmss(result.file_position / SAMPLE_RATE)}"
                        f" / {_fmt_mmss(result.total_samples / SAMPLE_RATE)}{rate_tag}")

    if redraw:
        _render()
    elif playback_str:
        # No image change, but keep the playback timer ticking.
        _decorate()
    return []


# Initial idle frame
_render()

# -----------------------------
# RUN ANIMATION
# -----------------------------
ani = FuncAnimation(
    fig,
    update_plot,
    interval=int(1000 / UPDATE_RATE),
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()

# Clean up
engine.stop()
