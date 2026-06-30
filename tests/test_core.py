"""Unit tests for the signal-processing backend in ``infrasound_core``."""

import numpy as np
import pytest

import infrasound_core as core
from infrasound_core import InfrasoundEngine, FrameResult


# -----------------------------
# CONFIGURATION CONSISTENCY
# -----------------------------
def test_fft_size_matches_resolution():
    assert core.FFT_SIZE == int(core.SAMPLE_RATE / core.FREQ_RESOLUTION)
    assert core.BUFFER_SIZE == core.FFT_SIZE


def test_noise_floor_constants_derive_from_time_constants():
    # Alphas/warmup are expressed so wall-clock behavior is independent of rate.
    assert core.NOISE_FLOOR_ALPHA_UP == pytest.approx(
        1.0 / (core.NOISE_FLOOR_TAU_UP_S * core.UPDATE_RATE)
    )
    assert core.NOISE_FLOOR_ALPHA_DOWN == pytest.approx(
        1.0 / (core.NOISE_FLOOR_TAU_DOWN_S * core.UPDATE_RATE)
    )
    assert core.NOISE_FLOOR_WARMUP_FRAMES == int(
        core.NOISE_FLOOR_WARMUP_S * core.UPDATE_RATE
    )
    # Slow rise must be slower than fast drop.
    assert core.NOISE_FLOOR_ALPHA_UP < core.NOISE_FLOOR_ALPHA_DOWN


def test_infra_band_within_range():
    engine = InfrasoundEngine(mode='mic')
    assert engine.infra_freqs[0] >= core.LOW_HZ
    assert engine.infra_freqs[-1] <= core.HIGH_HZ
    # 0.5 Hz spacing.
    assert np.allclose(np.diff(engine.infra_freqs), core.FREQ_RESOLUTION)


# -----------------------------
# CONSTRUCTION
# -----------------------------
def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        InfrasoundEngine(mode='bogus')


@pytest.mark.parametrize("mode", ["mic", "test", "file"])
def test_valid_modes_construct(mode):
    engine = InfrasoundEngine(mode=mode)
    assert engine.mode == mode


def test_file_mode_start_without_path_raises():
    engine = InfrasoundEngine(mode='file')
    assert engine.file_path is None
    with pytest.raises(ValueError):
        engine.start()
    assert engine.audio_buffer.shape == (core.BUFFER_SIZE,)


# -----------------------------
# RING BUFFER
# -----------------------------
def test_ring_write_wraps_around():
    engine = InfrasoundEngine(mode='mic')
    engine.audio_buffer[:] = 0
    engine.audio_buffer_write_idx = core.BUFFER_SIZE - 3
    engine._ring_write(np.array([1, 2, 3, 4, 5], dtype='float32'))
    assert list(engine.audio_buffer[-3:]) == [1, 2, 3]
    assert list(engine.audio_buffer[:2]) == [4, 5]
    assert engine.audio_buffer_write_idx == 2


def test_ring_write_truncates_oversized_chunk():
    engine = InfrasoundEngine(mode='mic')
    engine.audio_buffer[:] = 0
    engine.audio_buffer_write_idx = 0
    data = np.arange(core.BUFFER_SIZE + 10, dtype='float32')
    engine._ring_write(data)
    # Only the most recent BUFFER_SIZE samples are kept.
    assert engine.audio_buffer[0] == 10
    assert engine.audio_buffer[-1] == core.BUFFER_SIZE + 9


# -----------------------------
# TEST-SIGNAL GENERATOR
# -----------------------------
def test_generate_test_signal_advances_time():
    engine = InfrasoundEngine(mode='test')
    engine.test_time_s = 0.0
    engine.generate_test_signal()
    block_s = int(core.SAMPLE_RATE / core.UPDATE_RATE) / core.SAMPLE_RATE
    assert engine.test_time_s == pytest.approx(block_s)


def test_generate_test_signal_rumble_phase_is_nonzero():
    engine = InfrasoundEngine(mode='test')
    engine.audio_buffer[:] = 0
    engine.test_time_s = 0.0  # cycle_time 0 -> rumble ON
    engine.generate_test_signal()
    assert np.any(engine.audio_buffer != 0)


# -----------------------------
# FRAME PROCESSING / FFT
# -----------------------------
def _inject_tone(engine, freq_hz):
    t = np.arange(core.BUFFER_SIZE) / core.SAMPLE_RATE
    sig = np.sin(2 * np.pi * freq_hz * t).astype('float32')
    engine.audio_buffer_write_idx = 0
    engine._ring_write(sig)


def test_process_frame_returns_frameresult_with_shapes():
    engine = InfrasoundEngine(mode='mic')
    _inject_tone(engine, 20.0)
    result = engine.process_frame()
    assert isinstance(result, FrameResult)
    n = len(engine.infra_freqs)
    assert len(result.infra_db) == n
    assert len(result.relative_db) == n
    assert len(result.noise_floor) == n
    assert result.file_position == 0
    assert result.total_samples == 0


def test_process_frame_peak_at_injected_tone():
    engine = InfrasoundEngine(mode='mic')
    _inject_tone(engine, 20.0)  # 20 Hz -> bin 40 at 0.5 Hz spacing
    result = engine.process_frame()
    assert int(np.argmax(result.magnitude)) == 40


def test_first_frame_relative_db_is_zero():
    engine = InfrasoundEngine(mode='mic')
    _inject_tone(engine, 18.0)
    result = engine.process_frame()
    # First frame seeds the noise floor to the current spectrum.
    assert engine.noise_floor_warmup == 1
    assert np.allclose(result.relative_db, 0.0)


def test_noise_floor_asymmetric_adaptation():
    engine = InfrasoundEngine(mode='mic')
    _inject_tone(engine, 18.0)
    first = engine.process_frame()
    infra_db = first.infra_db.copy()

    # Jump past warmup so the asymmetric EMA is used.
    engine.noise_floor_warmup = core.NOISE_FLOOR_WARMUP_FRAMES

    # Signal everywhere ABOVE the floor -> floor rises slowly (alpha_up).
    engine.noise_floor = infra_db - 10.0
    engine.process_frame()  # same buffer -> identical infra_db
    expected_up = (infra_db - 10.0) + core.NOISE_FLOOR_ALPHA_UP * 10.0
    assert np.allclose(engine.noise_floor, expected_up)

    # Signal everywhere BELOW the floor -> floor drops faster (alpha_down).
    engine.noise_floor = infra_db + 10.0
    engine.process_frame()
    expected_down = (infra_db + 10.0) - core.NOISE_FLOOR_ALPHA_DOWN * 10.0
    assert np.allclose(engine.noise_floor, expected_down)


# -----------------------------
# WAV LOADING
# -----------------------------
def _write_wav(path, data, rate):
    from scipy.io import wavfile
    wavfile.write(str(path), rate, data)


def test_load_wav_int16_mono(tmp_path):
    engine = InfrasoundEngine(mode='file')
    n = 4410
    t = np.arange(n) / core.SAMPLE_RATE
    data = (np.sin(2 * np.pi * 18 * t) * 30000).astype(np.int16)
    p = tmp_path / "mono.wav"
    _write_wav(p, data, core.SAMPLE_RATE)

    mono, output, channels = engine._load_wav_for_replay(str(p))
    assert channels == 1
    assert mono.dtype == np.float32
    assert len(mono) == n
    assert output.shape == (n, 1)
    assert np.abs(mono).max() <= 1.0


def test_load_wav_float32_stereo_mixes_to_mono(tmp_path):
    engine = InfrasoundEngine(mode='file')
    n = 4410
    left = np.full(n, 0.5, dtype=np.float32)
    right = np.full(n, -0.1, dtype=np.float32)
    data = np.column_stack([left, right])
    p = tmp_path / "stereo.wav"
    _write_wav(p, data, core.SAMPLE_RATE)

    mono, output, channels = engine._load_wav_for_replay(str(p))
    assert channels == 2
    assert output.shape == (n, 2)
    assert np.allclose(mono, 0.2, atol=1e-6)  # (0.5 + -0.1) / 2


def test_load_wav_resamples_to_engine_rate(tmp_path):
    engine = InfrasoundEngine(mode='file')
    src_rate = 48000
    n = 4800
    t = np.arange(n) / src_rate
    data = (np.sin(2 * np.pi * 18 * t) * 30000).astype(np.int16)
    p = tmp_path / "resample.wav"
    _write_wav(p, data, src_rate)

    mono, _output, _channels = engine._load_wav_for_replay(str(p))
    expected = n * core.SAMPLE_RATE / src_rate
    assert abs(len(mono) - expected) <= 2


# -----------------------------
# FAST-SCAN FILE ADVANCE
# -----------------------------
def test_advance_file_writes_hop_and_advances_position():
    engine = InfrasoundEngine(mode='file')
    engine.file_audio = np.arange(100, dtype='float32')
    engine.file_position = 0
    engine.audio_buffer[:] = 0
    engine.audio_buffer_write_idx = 0
    engine.advance_file(10)
    assert engine.file_position == 10
    assert list(engine.audio_buffer[:10]) == list(range(10))


def test_advance_file_wraps_at_end():
    engine = InfrasoundEngine(mode='file')
    engine.file_audio = np.arange(100, dtype='float32')
    engine.file_position = 95
    engine.advance_file(10)
    # 5 samples to the end, then wrap to the start for the remaining 5.
    assert engine.file_position == 5


def test_advance_file_noop_without_file():
    engine = InfrasoundEngine(mode='file')
    assert engine.file_audio is None
    engine.advance_file(10)  # must not raise
    assert engine.file_position == 0

