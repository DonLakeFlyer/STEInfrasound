"""Reusable live infrasound bar-spectrum view.

The color-coded bar chart that shows per-bin signal strength relative to the
adaptive noise floor was originally inlined in ``infrasound.py``. It now lives
here as :class:`BarChartView` so that both the bar app (``infrasound.py``) and
the detection app (``infrasound_detections.py``) can render the same raw view
without duplicating the drawing/colour logic.

This module performs no audio work; it is fed :class:`infrasound_core.FrameResult`
objects produced by :class:`infrasound_core.InfrasoundEngine`.
"""

import numpy as np

from infrasound_core import (
    FREQ_RESOLUTION,
    LOW_HZ,
    HIGH_HZ,
    DB_MIN,
    DB_MAX,
    SAMPLE_RATE,
)

# Peak-hold decay applied each update (0.90 = 10% decay per frame).
DECAY_RATE = 0.90


def _fmt_mmss(seconds):
    """Format a number of seconds as mm:ss."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


class BarChartView:
    """Color-coded bar spectrum with peak-hold decay and a noise-floor readout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw the bars into.
    infra_freqs : numpy.ndarray
        Frequency axis for the infrasound band (from the engine).
    """

    def __init__(self, ax, infra_freqs):
        self.ax = ax
        self.infra_freqs = infra_freqs
        self.peak_values = np.zeros(len(infra_freqs))
        self.base_title = f"Infrasound {LOW_HZ}-{HIGH_HZ} Hz"

        self.bars = ax.bar(
            infra_freqs,
            np.zeros(len(infra_freqs)),
            width=FREQ_RESOLUTION * 0.8,
            color='cyan',
        )
        ax.set_xlim(-FREQ_RESOLUTION / 2, HIGH_HZ)  # show the 0 Hz bar fully
        ax.set_ylim(DB_MIN, DB_MAX)
        ax.set_xlabel("Frequency (Hz)", fontsize=7)
        ax.set_ylabel("dB above background", fontsize=7)
        ax.set_title(self.base_title, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

        self.noise_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            ha='left', va='top', fontsize=5, color='gray',
            fontfamily='monospace',
        )

    def update(self, result):
        """Update the bars + noise text from a :class:`FrameResult`.

        In file-replay mode the title gains an ``mm:ss / mm:ss`` current/total
        playback readout. Returns the list of artists that changed.
        """
        relative_db = result.relative_db

        # Peak hold with decay.
        self.peak_values *= DECAY_RATE
        self.peak_values = np.maximum(self.peak_values, relative_db)

        display_values = np.clip(self.peak_values, DB_MIN, DB_MAX)

        for bar, height in zip(self.bars, display_values):
            bar.set_height(height)
            if height >= 16:
                bar.set_color('#ff4444')    # Red: strong signal
            elif height >= 8:
                bar.set_color('#ffaa00')    # Orange: moderate signal
            elif height >= 3:
                bar.set_color('#ffff44')    # Yellow: weak signal
            else:
                bar.set_color('cyan')       # Cyan: background

        avg_floor = np.mean(result.noise_floor)
        self.noise_text.set_text(f"floor: {avg_floor:.0f} dB SPL")

        # File-replay playback position (mm:ss / mm:ss) in the title.
        if result.total_samples:
            pos = _fmt_mmss(result.file_position / SAMPLE_RATE)
            total = _fmt_mmss(result.total_samples / SAMPLE_RATE)
            self.ax.set_title(f"{self.base_title}  |  {pos} / {total}", fontsize=8)
        else:
            # Non-file source: keep the plain title (avoids a stale file-mode
            # playback position lingering if a view is reused across modes).
            self.ax.set_title(self.base_title, fontsize=8)

        return list(self.bars) + [self.noise_text]


if __name__ == '__main__':
    import sys
    print(
        "infrasound_bar.py is a shared library module, not a runnable app.\n"
        "Run one of the display apps instead:\n"
        "  python infrasound.py [PATH|--file PATH|--test]              # bar display\n"
        "  python infrasound_detections.py [PATH|--file PATH|--test]   # detection display\n"
        "A bare PATH analyzes a WAV file; no path uses the live microphone.",
        file=sys.stderr,
    )
    sys.exit(2)
