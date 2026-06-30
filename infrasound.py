"""Real-time infrasound bar display.

Live spectrum analyzer for the 0-50 Hz band (infrasound + elephant-rumble
harmonics), designed for a Raspberry Pi with a 3.5" 320x480 display. Shows a
color-coded bar chart of signal strength relative to an adaptive noise floor.

Audio capture, FFT, and noise-floor tracking live in ``infrasound_core`` and the
splash/error/menu chrome lives in ``infrasound_ui`` so this file only handles the
bar visualization.

Usage:
    python infrasound.py                  # live microphone
    python infrasound.py --test           # synthetic 18 Hz rumble, no mic needed
    python infrasound.py --file REC.wav   # replay and analyze a WAV file
"""

import signal
import sys
import traceback

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
)
from infrasound_bar import BarChartView

if IS_RPI:
    print("Running on Raspberry Pi")
else:
    print("Running on desktop/other platform")

# -----------------------------
# MODE SELECTION
# -----------------------------
TEST_MODE = '--test' in sys.argv

FILE_MODE = False
FILE_PATH = None
_file_idx = sys.argv.index('--file') if '--file' in sys.argv else -1
if _file_idx != -1:
    if _file_idx + 1 >= len(sys.argv):
        print("Usage error: --file requires a following path argument.", file=sys.stderr)
        sys.exit(2)
    FILE_MODE = True
    FILE_PATH = sys.argv[_file_idx + 1]
else:
    # Accept a bare path argument (e.g. `python infrasound.py recording.wav`).
    # Any non-flag argument is treated as a file to replay. No path => mic mode.
    for _arg in sys.argv[1:]:
        if not _arg.startswith('-'):
            FILE_MODE = True
            FILE_PATH = _arg
            break

if TEST_MODE and FILE_MODE:
    print("Usage error: --test cannot be combined with --file or a file path.",
          file=sys.stderr)
    sys.exit(2)

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
        engine.start()
    except Exception as exc:
        log(f"ERROR starting file mode: {exc}")
        traceback.print_exc()
        sys.exit(1)
elif TEST_MODE:
    engine.start()
elif not ui.show_splash_and_connect(engine):
    # Connection failed after timeout
    print(f"Error opening audio stream: {engine.audio_error}")
    ui.show_error_screen(engine.audio_error if engine.audio_error else "Connection timeout")
    sys.exit(1)


# -----------------------------
# SETUP PLOT
# -----------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)

# Color-coded bar spectrum with peak-hold decay (shared with the detection app).
bar_view = BarChartView(ax, engine.infra_freqs)

# Power/exit menu on click
ui.attach_power_menu(fig, ax)


# -----------------------------
# UPDATE FUNCTION
# -----------------------------
def update_plot(frame):
    result = engine.process_frame()
    # BarChartView updates the bars, noise text, and (in file mode) the
    # current/total playback time in the title.
    return bar_view.update(result)


# -----------------------------
# RUN ANIMATION
# -----------------------------
ani = FuncAnimation(
    fig,
    update_plot,
    interval=int(1000 / UPDATE_RATE),  # milliseconds (100ms for 10 Hz)
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()

# Clean up
engine.stop()
