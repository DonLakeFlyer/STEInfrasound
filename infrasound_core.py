"""Shared infrasound backend: audio capture, FFT, and adaptive noise floor.

This module contains the signal-processing engine extracted from the original
monolithic ``infrasound.py`` so that multiple display applications can share a
single backend. It owns audio acquisition (live mic, synthetic test signal, or
WAV file replay), the rolling sample ring buffer, the windowed FFT, and the
asymmetric adaptive noise-floor tracker.

Display/UI concerns (matplotlib windows, splash/error/menu screens) live in
``infrasound_ui.py``; this module performs no drawing.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from collections import namedtuple
from math import gcd

import numpy as np
import sounddevice as sd

# -----------------------------
# LOGGING
# -----------------------------
def log(msg):
    """Print a timestamped log line (flushed for service/journald capture)."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


# -----------------------------
# PLATFORM DETECTION
# -----------------------------
def is_raspberry_pi():
    """Detect if running on Raspberry Pi."""
    try:
        # Check for Raspberry Pi specific hardware
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                if 'Raspberry Pi' in model:
                    return True
        # Alternative check using platform
        if platform.machine() in ['armv7l', 'aarch64'] and platform.system() == 'Linux':
            return True
    except Exception:
        pass
    return False


IS_RPI = is_raspberry_pi()


# -----------------------------
# CONFIGURATION
# -----------------------------
SAMPLE_RATE = 44100  # Hz
UPDATE_RATE = 10  # Updates per second (0.1 s hop => ~95% window overlap)
FREQ_RESOLUTION = 0.5  # Hz resolution

# Calculate FFT size needed for 0.5 Hz resolution
# Resolution = sample_rate / fft_size, so fft_size = sample_rate / resolution
FFT_SIZE = int(SAMPLE_RATE / FREQ_RESOLUTION)

# Buffer size should be enough for smooth updates
BUFFER_SIZE = FFT_SIZE  # Keep full FFT size for resolution

# Infrasound + low-frequency range (includes elephant rumble harmonics up to 50 Hz)
LOW_HZ = 0
HIGH_HZ = 50

# Display settings - relative to noise floor
DB_MIN = 0
DB_MAX = 30

# Microphone calibration
# LS-P5 max SPL is ~120 dB, meaning 0 dBFS ≈ 120 dB SPL
# Adjust this value if you calibrate against a known reference
MIC_CAL_OFFSET = 120  # dB SPL at full scale (0 dBFS)

# Noise floor tracking - asymmetric adaptation.
# Alphas are derived from wall-clock time constants so behavior is independent
# of UPDATE_RATE: for an EMA, time_constant_s ~= 1 / (alpha * UPDATE_RATE).
NOISE_FLOOR_TAU_UP_S = 100.0    # Slow rise - signals don't inflate floor
NOISE_FLOOR_TAU_DOWN_S = 10.0   # Fast drop - floor settles quickly after signal ends
NOISE_FLOOR_WARMUP_S = 5.0      # Use fast adaptation for the first few seconds
NOISE_FLOOR_ALPHA_UP = 1.0 / (NOISE_FLOOR_TAU_UP_S * UPDATE_RATE)
NOISE_FLOOR_ALPHA_DOWN = 1.0 / (NOISE_FLOOR_TAU_DOWN_S * UPDATE_RATE)
NOISE_FLOOR_WARMUP_FRAMES = int(NOISE_FLOOR_WARMUP_S * UPDATE_RATE)


# Per-frame analysis result handed to display apps.
FrameResult = namedtuple(
    'FrameResult',
    [
        'infra_freqs',   # frequency axis for the infrasound band (Hz)
        'infra_db',      # absolute magnitude per bin (dB SPL)
        'relative_db',   # dB above the adaptive noise floor
        'noise_floor',   # current noise-floor estimate per bin (dB SPL)
        'magnitude',     # full rfft magnitude spectrum (for detectors)
        'file_position', # current sample index in file replay (0 otherwise)
        'total_samples', # total audio samples in the loaded file (0 otherwise)
    ],
)


class InfrasoundEngine:
    """Audio capture + FFT + adaptive noise-floor engine.

    Modes
    -----
    'mic'  : live capture from an input device (connect via :meth:`try_connect_mic`).
    'test' : synthetic elephant-rumble generator (no hardware required).
    'file' : replay a WAV file through the speakers while analysing it.

    Typical usage::

        engine = InfrasoundEngine(mode='test')
        engine.start()
        result = engine.process_frame()   # call once per display frame
        ...
        engine.stop()
    """

    def __init__(self, mode='mic', file_path=None):
        if mode not in ('mic', 'test', 'file'):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.file_path = file_path

        # Rolling buffer - always keep the last BUFFER_SIZE samples
        self.audio_buffer = np.zeros(BUFFER_SIZE, dtype='float32')
        # ring-buffer write position; analysis reconstructs order via np.roll
        self.audio_buffer_write_idx = 0

        # File replay state
        self.file_audio = None          # float32 mono array for analysis
        self.file_audio_output = None   # float32 (N, n_channels) for playback
        self.file_output_channels = 0
        self.file_position = 0
        self._file_cb_status = None     # last status from file callback (logged on UI thread)
        self._mic_cb_status = None      # last status from mic callback (logged on UI thread)

        # Test mode state (start in OFF cycle so prefill is noise-only)
        self.test_time_s = 8.0

        # Streams
        self.stream = None       # mic InputStream
        self.file_stream = None  # file OutputStream

        # Connection status (mic mode)
        self.audio_connected = False
        self.audio_error = None

        # Precompute frequency axis, band mask, and window
        self.freqs = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
        self.infrasound_mask = (self.freqs >= LOW_HZ) & (self.freqs <= HIGH_HZ)
        self.infra_freqs = self.freqs[self.infrasound_mask]
        self.window = np.hanning(FFT_SIZE)
        self.window_sum = np.sum(self.window)

        # Noise floor (adapts from the first few frames)
        self.noise_floor = None
        self.noise_floor_warmup = 0

    # -----------------------------
    # RING BUFFER
    # -----------------------------
    def _ring_write(self, data):
        """Write data into the ring buffer (called from audio callback thread)."""
        n = len(data)
        if n > BUFFER_SIZE:
            # Chunk is larger than the ring buffer; keep only the most recent samples.
            data = data[-BUFFER_SIZE:]
            n = BUFFER_SIZE
        end = (self.audio_buffer_write_idx + n) % BUFFER_SIZE
        if end > self.audio_buffer_write_idx:
            self.audio_buffer[self.audio_buffer_write_idx:end] = data
        else:
            split = BUFFER_SIZE - self.audio_buffer_write_idx
            self.audio_buffer[self.audio_buffer_write_idx:] = data[:split]
            self.audio_buffer[:end] = data[split:]
        self.audio_buffer_write_idx = end

    # -----------------------------
    # AUDIO CALLBACKS
    # -----------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self._mic_cb_status = str(status)  # defer to UI thread; no I/O in callback
        self._ring_write(indata[:, 0])

    def generate_test_signal(self):
        """Generate synthetic elephant rumble + noise for test mode.

        Cycles: 8s rumble ON, 8s rumble OFF.
        Rumble = 18 Hz fundamental + 36 Hz (2nd harmonic) + 54 Hz (3rd harmonic)
        with amplitudes modelling EM272 mic rolloff below 20 Hz (fundamental
        attenuated, 2nd harmonic strongest, 3rd intermediate).
        Background = low-level white Gaussian noise.
        """
        block_size = int(SAMPLE_RATE / UPDATE_RATE)
        t = np.arange(block_size) / SAMPLE_RATE + self.test_time_s

        # Cycle: 8s on, 8s off (decide before advancing time so boundary is correct)
        cycle_time = self.test_time_s % 16.0
        self.test_time_s += block_size / SAMPLE_RATE

        # Background noise (low amplitude)
        noise = np.random.randn(block_size).astype('float32') * 0.002

        if cycle_time < 8.0:
            # Elephant rumble: 18 Hz fundamental + harmonics
            fundamental = 18.0
            rumble = (
                0.003 * np.sin(2 * np.pi * fundamental * t) +       # f0
                0.006 * np.sin(2 * np.pi * 2 * fundamental * t) +   # 2f
                0.004 * np.sin(2 * np.pi * 3 * fundamental * t)     # 3f
            ).astype('float32')
            signal_data = noise + rumble
        else:
            signal_data = noise

        # Write block into ring buffer (no allocation; called from main thread in test mode)
        self._ring_write(signal_data)

    def _file_stream_callback(self, outdata, frames, time_info, status):
        """Write file audio to speakers and analysis buffer in sync."""
        if status:
            self._file_cb_status = str(status)  # defer to UI thread; no I/O in callback

        total = len(self.file_audio)
        start_pos = self.file_position
        filled = 0
        while filled < frames:
            remaining = total - self.file_position
            needed = frames - filled
            take = min(needed, remaining)
            outdata[filled:filled + take] = \
                self.file_audio_output[self.file_position:self.file_position + take]
            filled += take
            self.file_position += take
            if self.file_position >= total:
                self.file_position = 0  # loop

        # Write the same mono samples to the analysis ring buffer.
        # Replay the position arithmetic using start_pos to avoid a staging buffer,
        # so any PortAudio-delivered frame count is handled without allocation.
        pos = start_pos
        remaining_frames = frames
        while remaining_frames > 0:
            chunk = min(remaining_frames, total - pos)
            self._ring_write(self.file_audio[pos:pos + chunk])
            pos = (pos + chunk) % total
            remaining_frames -= chunk

    def advance_file(self, hop):
        """Advance file replay by *hop* samples for analysis only (no audio).

        Mirrors the buffer-fill side of :meth:`_file_stream_callback` so a faster-
        than-real-time offline scan produces exactly the same FFT windows as
        real-time playback. Used when the audio output stream is disabled (silent
        fast-scan). Only meaningful in file mode; a no-op until a file is loaded.
        """
        if self.file_audio is None:
            return
        total = len(self.file_audio)
        pos = self.file_position
        remaining = hop
        while remaining > 0:
            chunk = min(remaining, total - pos)
            self._ring_write(self.file_audio[pos:pos + chunk])
            pos = (pos + chunk) % total
            remaining -= chunk
        self.file_position = pos

    # -----------------------------
    # WAV LOADING
    # -----------------------------
    def _load_wav_for_replay(self, path):
        """Load a WAV file. Returns (mono_arr, output_arr_2d, n_channels)."""
        from scipy.io import wavfile as sp_wavfile
        from scipy import signal as sp_signal

        # mmap=True maps the raw sample data lazily; the normalisation and
        # resampling steps below still allocate full float32 arrays.
        try:
            framerate, data = sp_wavfile.read(path, mmap=True)
        except (TypeError, ValueError):
            framerate, data = sp_wavfile.read(path)

        # Normalise to float32 in [-1, 1].
        # Use np.float32 divisors so all arithmetic stays in float32 (Python float
        # would silently upcast the result to float64, doubling memory use).
        if data.dtype == np.float32:
            # Force an in-RAM copy: if mmap=True succeeded, data is a memory-mapped
            # array and reads in the real-time callback would trigger disk page faults.
            arr = np.array(data, dtype=np.float32)
        elif data.dtype == np.float64:
            arr = data.astype(np.float32)
        elif data.dtype == np.int16:
            arr = data.astype(np.float32) / np.float32(np.iinfo(np.int16).max + 1)
        elif data.dtype == np.int32:
            # scipy reads both 24-bit and 32-bit PCM as int32; check actual bit depth
            import wave as _wave
            try:
                with _wave.open(path, 'r') as _wf:
                    _bits = _wf.getsampwidth() * 8
            except Exception:
                _bits = 32
            arr = data.astype(np.float32) / np.float32(1 << (_bits - 1))
        elif data.dtype == np.uint8:
            u8_scale = np.float32(np.iinfo(np.uint8).max + 1)
            arr = (data.astype(np.float32) - (u8_scale / np.float32(2.0))) / (u8_scale / np.float32(2.0))
        else:
            raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

        # Ensure 2D output array (n_frames, n_channels)
        if arr.ndim == 1:
            output_arr = arr[:, np.newaxis]
        else:
            output_arr = arr
        n_channels = output_arr.shape[1]

        # Resample to detector SAMPLE_RATE if needed
        if framerate != SAMPLE_RATE:
            g = gcd(SAMPLE_RATE, framerate)
            up = SAMPLE_RATE // g
            down = framerate // g
            output_arr = sp_signal.resample_poly(output_arr, up, down, axis=0).astype(np.float32)

        # Mono mix derived from (resampled) output array.
        # dtype=np.float32 keeps the accumulator in float32, avoiding a float64 temporary
        # that would double memory use for large recordings.
        mono_arr = output_arr.mean(axis=1, dtype=np.float32)

        if len(mono_arr) == 0:
            raise ValueError("WAV file contains no audio frames")
        duration = len(mono_arr) / SAMPLE_RATE
        log(f"FILE MODE: loaded '{path}' — {data.dtype}, {n_channels}ch, "
            f"{framerate} Hz → {SAMPLE_RATE} Hz, {duration:.1f}s")
        return mono_arr, output_arr, n_channels

    # -----------------------------
    # LIFECYCLE
    # -----------------------------
    def try_connect_mic(self):
        """Attempt to open and start the mic input stream once.

        Returns True on success (and sets :attr:`audio_connected`); on failure
        records the error in :attr:`audio_error` and returns False.
        """
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE / UPDATE_RATE),
                dtype='float32',
            )
            self.stream.start()
            self.audio_connected = True
            self.audio_error = None
            log(f"SUCCESS! Audio stream started at {SAMPLE_RATE} Hz")
            log(f"FFT size: {FFT_SIZE} samples ({FFT_SIZE/SAMPLE_RATE:.1f} seconds)")
            log(f"Frequency resolution: {FREQ_RESOLUTION} Hz")
            log(f"Update rate: {UPDATE_RATE} Hz")
            return True
        except Exception as e:
            # Close any partially-opened stream so retries don't leak PortAudio
            # resources (which could otherwise block a later successful connect).
            if self.stream is not None:
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            self.audio_connected = False
            self.audio_error = str(e)
            return False

    def start(self, play_audio=True):
        """Start the engine for non-mic modes.

        For 'test' mode this prefills the buffer with noise; for 'file' mode it
        loads the WAV and (when *play_audio* is True) starts the output stream so
        playback drives the analysis buffer in real time. With *play_audio* False
        the WAV is loaded but no audio is played — the caller advances the file
        itself via :meth:`advance_file` (used for faster-than-real-time scans).
        For 'mic' mode use :meth:`try_connect_mic` (driven by the splash UI).
        """
        if self.mode == 'test':
            log("TEST MODE: using synthetic elephant rumble signal")
            self.audio_connected = True
            # Pre-fill at least one full FFT window so the first spectra (and the
            # noise-floor seed) aren't computed from a mostly zero-padded buffer.
            block_size = int(SAMPLE_RATE / UPDATE_RATE)
            n_prefill = int(np.ceil(FFT_SIZE / block_size))
            for _ in range(n_prefill):
                self.generate_test_signal()
        elif self.mode == 'file':
            if not self.file_path:
                raise ValueError("file mode requires a file_path")
            log(f"FILE MODE: replaying '{self.file_path}'")
            (self.file_audio,
             self.file_audio_output,
             self.file_output_channels) = self._load_wav_for_replay(self.file_path)
            if play_audio:
                # Start audio output stream — callback keeps playback in sync with display buffer
                self.file_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=self.file_output_channels,
                    dtype='float32',
                    blocksize=int(SAMPLE_RATE / UPDATE_RATE),
                    callback=self._file_stream_callback,
                )
                self.file_stream.start()
                log(f"FILE MODE: audio output started ({self.file_output_channels}ch)")
            else:
                log("FILE MODE: silent fast-scan (no audio output)")
            self.audio_connected = True

    def stop(self):
        """Stop and close any open audio streams."""
        if self.file_stream is not None:
            try:
                self.file_stream.stop()
                self.file_stream.close()
            except Exception:
                pass
            self.file_stream = None
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    # -----------------------------
    # PER-FRAME ANALYSIS
    # -----------------------------
    def process_frame(self):
        """Run one analysis cycle and return a :class:`FrameResult`.

        Call once per display update. In test mode this also advances the
        synthetic signal generator; in file mode the buffer is filled by the
        output callback in sync with playback.
        """
        # Log any deferred status messages from audio callbacks
        if self._file_cb_status is not None:
            log(f"File stream status: {self._file_cb_status}")
            self._file_cb_status = None
        if self._mic_cb_status is not None:
            log(f"Audio status: {self._mic_cb_status}")
            self._mic_cb_status = None

        # In test mode, generate synthetic data each frame.
        # In file mode the buffer is filled by _file_stream_callback in sync
        # with audio output; in mic mode by _audio_callback.
        if self.mode == 'test':
            self.generate_test_signal()

        # Snapshot the ring buffer without a lock. audio_buffer_write_idx is a
        # Python int (GIL-atomic read); the buffer copy may overlap a callback
        # write at most at its boundary, producing one anomalous FFT bin —
        # imperceptible at the display update rate.
        _write_idx = self.audio_buffer_write_idx
        _buf_copy = self.audio_buffer.copy()
        audio_snapshot = np.roll(_buf_copy, -_write_idx)

        # Apply window function and compute FFT
        windowed = audio_snapshot * self.window
        spectrum = np.fft.rfft(windowed)
        # Normalize to get amplitude relative to full scale
        magnitude = np.abs(spectrum) * 2.0 / self.window_sum

        # Extract infrasound band
        infra_mag = magnitude[self.infrasound_mask]

        # Convert to dB SPL
        infra_db = 20 * np.log10(infra_mag + 1e-10) + MIC_CAL_OFFSET

        # Update noise floor (asymmetric exponential moving average)
        if self.noise_floor is None:
            self.noise_floor = infra_db.copy()
            self.noise_floor_warmup = 1
        elif self.noise_floor_warmup < NOISE_FLOOR_WARMUP_FRAMES:
            # Fast convergence during warmup (buffer may be partially empty)
            self.noise_floor = 0.3 * infra_db + 0.7 * self.noise_floor
            self.noise_floor_warmup += 1
        else:
            # Slow adaptation upward (signals don't raise floor)
            # Fast adaptation downward (floor drops when quiet)
            alpha = np.where(infra_db > self.noise_floor,
                             NOISE_FLOOR_ALPHA_UP, NOISE_FLOOR_ALPHA_DOWN)
            self.noise_floor = alpha * infra_db + (1 - alpha) * self.noise_floor

        # Compute dB above noise floor
        relative_db = infra_db - self.noise_floor

        total_samples = len(self.file_audio) if self.file_audio is not None else 0

        return FrameResult(
            infra_freqs=self.infra_freqs,
            infra_db=infra_db,
            relative_db=relative_db,
            noise_floor=self.noise_floor,
            magnitude=magnitude,
            file_position=self.file_position,
            total_samples=total_samples,
        )


if __name__ == '__main__':
    print(
        "infrasound_core.py is a shared backend module, not a runnable app.\n"
        "Run one of the display apps instead:\n"
        "  python infrasound.py [PATH|--file PATH|--test]              # bar display\n"
        "  python infrasound_detections.py [PATH|--file PATH|--test]   # detection display\n"
        "A bare PATH analyzes a WAV file; no path uses the live microphone.",
        file=sys.stderr,
    )
    sys.exit(2)
