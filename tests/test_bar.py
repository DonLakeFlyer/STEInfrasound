"""Unit tests for the reusable bar-spectrum view in ``infrasound_bar``."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

import infrasound_bar as bar
from infrasound_bar import BarChartView, _fmt_mmss
from infrasound_core import FrameResult, DB_MIN, DB_MAX


def _make_result(relative_db, noise_floor=None, file_position=0, total_samples=0):
    relative_db = np.asarray(relative_db, dtype=float)
    n = len(relative_db)
    freqs = np.arange(n, dtype=float) * 0.5
    if noise_floor is None:
        noise_floor = np.full(n, 25.0)
    return FrameResult(
        infra_freqs=freqs,
        infra_db=relative_db + np.asarray(noise_floor, dtype=float),
        relative_db=relative_db,
        noise_floor=np.asarray(noise_floor, dtype=float),
        magnitude=np.zeros(n),
        file_position=file_position,
        total_samples=total_samples,
    )


@pytest.fixture
def ax():
    fig, axis = plt.subplots()
    yield axis
    plt.close(fig)


def test_fmt_mmss():
    assert _fmt_mmss(0) == "00:00"
    assert _fmt_mmss(65) == "01:05"
    assert _fmt_mmss(3600) == "60:00"


def test_bar_heights_match_clipped_relative_db(ax):
    result = _make_result([5.0, 12.0, 40.0, -3.0])
    view = BarChartView(ax, result.infra_freqs)
    view.update(result)
    heights = [b.get_height() for b in view.bars]
    expected = np.clip([5.0, 12.0, 40.0, -3.0], DB_MIN, DB_MAX)
    assert heights == pytest.approx(list(expected))


def test_bar_color_tiers(ax):
    # 20 -> red, 10 -> orange, 5 -> yellow, 1 -> cyan
    result = _make_result([20.0, 10.0, 5.0, 1.0])
    view = BarChartView(ax, result.infra_freqs)
    view.update(result)
    colors = [b.get_facecolor() for b in view.bars]
    import matplotlib.colors as mcolors
    assert colors[0] == pytest.approx(mcolors.to_rgba('#ff4444'))
    assert colors[1] == pytest.approx(mcolors.to_rgba('#ffaa00'))
    assert colors[2] == pytest.approx(mcolors.to_rgba('#ffff44'))
    assert colors[3] == pytest.approx(mcolors.to_rgba('cyan'))


def test_peak_hold_decay(ax):
    result_hi = _make_result([20.0, 20.0])
    result_lo = _make_result([0.0, 0.0])
    view = BarChartView(ax, result_hi.infra_freqs)
    view.update(result_hi)
    view.update(result_lo)
    heights = [b.get_height() for b in view.bars]
    # Peak decays by DECAY_RATE when the new frame is lower.
    assert heights == pytest.approx([20.0 * bar.DECAY_RATE] * 2)


def test_noise_text_shows_floor(ax):
    result = _make_result([1.0, 1.0], noise_floor=[40.0, 50.0])
    view = BarChartView(ax, result.infra_freqs)
    view.update(result)
    assert "floor: 45 dB SPL" == view.noise_text.get_text()


def test_title_static_without_file(ax):
    result = _make_result([1.0, 1.0])
    view = BarChartView(ax, result.infra_freqs)
    view.update(result)
    assert ax.get_title() == view.base_title


def test_title_shows_playback_position_in_file_mode(ax):
    from infrasound_core import SAMPLE_RATE
    result = _make_result(
        [1.0, 1.0],
        file_position=65 * SAMPLE_RATE,
        total_samples=130 * SAMPLE_RATE,
    )
    view = BarChartView(ax, result.infra_freqs)
    view.update(result)
    assert "01:05 / 02:10" in ax.get_title()


def test_title_resets_to_base_when_leaving_file_mode(ax):
    from infrasound_core import SAMPLE_RATE
    file_result = _make_result(
        [1.0, 1.0],
        file_position=65 * SAMPLE_RATE,
        total_samples=130 * SAMPLE_RATE,
    )
    view = BarChartView(ax, file_result.infra_freqs)
    view.update(file_result)
    assert "01:05 / 02:10" in ax.get_title()
    # A subsequent non-file frame must clear the stale playback position.
    view.update(_make_result([1.0, 1.0]))
    assert ax.get_title() == view.base_title
