"""Play a WAV file through the system's default audio output.

This script uses the `simpleaudio` package to stream audio data read via Python's
built-in `wave` module. It currently supports uncompressed PCM WAV files.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import wave

try:
    import simpleaudio as sa
except ImportError as exc:  # pragma: no cover - import error only hit at runtime
    raise SystemExit(
        "simpleaudio is required. Install it with 'pip install -r requirements.txt'"
    ) from exc

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import error only hit at runtime
    raise SystemExit(
        "numpy is required. Install it with 'pip install -r requirements.txt'"
    ) from exc

try:
    from scipy import signal as sp_signal
except ImportError as exc:  # pragma: no cover - import error only hit at runtime
    raise SystemExit(
        "scipy is required for filtering. Install it with 'pip install -r requirements.txt'"
    ) from exc


SUPPORTED_SAMPLE_WIDTHS = {1, 2, 3, 4}


def _pcm_bytes_to_float_array(
    data: bytes, n_channels: int, sampwidth: int
) -> tuple[np.ndarray, dict]:
    """Convert interleaved PCM bytes to float32 array in range [-1, 1].

    Returns (array, meta) where array has shape (n_frames, n_channels).
    meta carries details to convert back: dtype, scale, sampwidth.
    """
    if sampwidth == 1:
        # 8-bit PCM is unsigned
        arr_u8 = np.frombuffer(data, dtype=np.uint8)
        arr = arr_u8.astype(np.int16) - 128
        scale = 127.0
        dtype = np.uint8
    elif sampwidth == 2:
        arr = np.frombuffer(data, dtype=np.dtype('<i2'))
        scale = 32767.0
        dtype = np.dtype('<i2')
    elif sampwidth == 3:
        # 24-bit little-endian packed
        b = np.frombuffer(data, dtype=np.uint8)
        if b.size % 3 != 0:
            raise ValueError("Corrupt 24-bit PCM data length")
        b = b.reshape(-1, 3)
        val = (b[:, 0].astype(np.int32)) | (b[:, 1].astype(np.int32) << 8) | (
            b[:, 2].astype(np.int32) << 16
        )
        # Sign extend 24-bit
        neg = (val & 0x800000) != 0
        val[neg] -= 1 << 24
        arr = val
        scale = float((1 << 23) - 1)
        dtype = 3  # sentinel for 24-bit
    elif sampwidth == 4:
        arr = np.frombuffer(data, dtype=np.dtype('<i4'))
        scale = 2147483647.0
        dtype = np.dtype('<i4')
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}")

    if arr.size % n_channels != 0:
        raise ValueError("PCM data does not align with channel count")

    n_frames = arr.size // n_channels
    arr = arr.reshape(n_frames, n_channels)
    floats = arr.astype(np.float32) / scale
    if sampwidth == 1:
        # Already centered above
        floats = floats.astype(np.float32)
    return floats, {"dtype": dtype, "scale": scale, "sampwidth": sampwidth}


def _float_array_to_pcm_bytes(
    floats: np.ndarray, n_channels: int, meta: dict
) -> bytes:
    """Convert float32 array in [-1, 1] back to interleaved PCM bytes.
    Shape expected: (n_frames, n_channels).
    """
    dtype = meta["dtype"]
    scale = meta["scale"]
    sampwidth = meta["sampwidth"]

    # Clip to [-1, 1] to avoid overflow
    x = np.clip(floats, -1.0, 1.0)

    if sampwidth == 1:
        ints = (x * 127.0).round().astype(np.int16) + 128
        ints = np.clip(ints, 0, 255).astype(np.uint8)
        interleaved = ints.reshape(-1, n_channels)
        return interleaved.astype(np.uint8).ravel().tobytes()
    elif sampwidth == 2:
        ints = (x * 32767.0).round().astype(np.int32)
        ints = np.clip(ints, -32768, 32767).astype(np.int16)
        return ints.reshape(-1, n_channels).astype(np.dtype('<i2')).ravel().tobytes()
    elif sampwidth == 3:
        ints = (x * ((1 << 23) - 1)).round().astype(np.int32)
        ints = np.clip(ints, -(1 << 23), (1 << 23) - 1)
        vals = ints.reshape(-1, n_channels).astype(np.int32).ravel()
        # Convert to unsigned 24-bit representation for packing
        vals = np.where(vals < 0, vals + (1 << 24), vals)
        b0 = (vals & 0xFF).astype(np.uint8)
        b1 = ((vals >> 8) & 0xFF).astype(np.uint8)
        b2 = ((vals >> 16) & 0xFF).astype(np.uint8)
        packed = np.column_stack((b0, b1, b2)).ravel()
        return packed.tobytes()
    elif sampwidth == 4:
        ints = (x * 2147483647.0).round().astype(np.int64)
        ints = np.clip(ints, -2147483648, 2147483647).astype(np.int32)
        return ints.reshape(-1, n_channels).astype(np.dtype('<i4')).ravel().tobytes()
    else:
        raise ValueError("Unsupported sample width for conversion")


def _scipy_bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter using SciPy with zero-phase filtering.

    x shape: (n_frames, n_channels), float32 in [-1,1]. Returns filtered array with same shape.
    """
    if high_hz <= 0.0 or high_hz >= fs / 2:
        raise ValueError("high cutoff must be in (0, Nyquist)")
    if low_hz < 0.0 or low_hz >= high_hz:
        raise ValueError("low cutoff must be >= 0 and < high cutoff")
    wn = [low_hz / (fs / 2.0), high_hz / (fs / 2.0)]
    sos = sp_signal.butter(order, wn, btype='band', output='sos')
    # Apply per channel for clarity
    y = np.empty_like(x)
    for ch in range(x.shape[1]):
        y[:, ch] = sp_signal.sosfiltfilt(sos, x[:, ch].astype(np.float64), axis=0).astype(np.float32)
    return y


def _translate_infrasound_to_audible(
    x: np.ndarray,
    fs: float,
    carrier_hz: float,
    gain: float = 1.0,
) -> np.ndarray:
    """Translate very-low-frequency content to audible range via SSB modulation.

    Steps per channel:
      1) Build analytic signal via Hilbert transform.
      2) Multiply by complex exponential e^{j 2π f_c t} to shift spectrum upward.
      3) Take the real part and apply gain; clip later during PCM conversion.

    Inputs/Outputs are float32 in [-1, 1], shape (n_frames, n_channels).
    """
    if carrier_hz <= 0.0 or carrier_hz >= fs / 2:
        raise ValueError("carrier must be in (0, Nyquist)")

    n = x.shape[0]
    t = np.arange(n, dtype=np.float64) / float(fs)
    osc = np.exp(1j * (2.0 * math.pi * carrier_hz * t))

    y = np.empty_like(x)
    for ch in range(x.shape[1]):
        sig = x[:, ch].astype(np.float64)
        analytic = np.asarray(sp_signal.hilbert(sig))  # ensure ndarray[complex]
        shifted = np.asarray(analytic * osc)
        real_sig = np.real(shifted).astype(np.float32)
        y[:, ch] = (gain * real_sig).astype(np.float32)
    return y


def play_wav_file(
    path: pathlib.Path,
    *,
    block: bool = True,
    infra_only: bool = False,
    infra_low_hz: float = 0.1,
    infra_high_hz: float = 20.0,
    translate_infra: bool = False,
    translate_carrier_hz: float = 200.0,
    translate_gain: float = 1.0,
) -> None:
    """Play the WAV file located at *path*.

    Parameters
    ----------
    path:
        Path to the WAV file to play.
    block:
        If True, block the caller until playback finishes. When False, the
        function returns immediately after starting playback.

    Raises
    ------
    FileNotFoundError
        If the provided path does not exist.
    ValueError
        If the WAV file uses an unsupported sample width or cannot be read.
    """

    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")

    if path.suffix.lower() != ".wav":
        raise ValueError("Only .wav files are supported")

    try:
        with wave.open(str(path), "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(num_frames)
    except wave.Error as exc:
        raise ValueError(f"Failed to read WAV file: {exc}") from exc

    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise ValueError(
            f"Unsupported sample width {sample_width} bytes. "
            "Valid widths: 1, 2, 3, or 4."
        )

    if translate_infra:
        floats, meta = _pcm_bytes_to_float_array(audio_data, num_channels, sample_width)
        # Band-limit first, then translate
        floats = _scipy_bandpass(floats, float(frame_rate), infra_low_hz, infra_high_hz, order=4)
        floats = _translate_infrasound_to_audible(
            floats, float(frame_rate), translate_carrier_hz, gain=translate_gain
        )
        audio_data = _float_array_to_pcm_bytes(floats, num_channels, meta)
    elif infra_only:
        floats, meta = _pcm_bytes_to_float_array(audio_data, num_channels, sample_width)
        floats = _scipy_bandpass(floats, float(frame_rate), infra_low_hz, infra_high_hz, order=4)
        audio_data = _float_array_to_pcm_bytes(floats, num_channels, meta)

    play_obj = sa.play_buffer(audio_data, num_channels, sample_width, frame_rate)

    if block:
        play_obj.wait_done()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a WAV file through the speakers")
    parser.add_argument("path", type=pathlib.Path, help="Path to the WAV file to play")
    parser.add_argument(
        "--no-block",
        dest="block",
        action="store_false",
        help="Return immediately after starting playback",
    )
    parser.add_argument(
        "--infra-only",
        action="store_true",
        help="Filter to infrasound band (default 0.1–20 Hz) before playback",
    )
    parser.add_argument(
        "--infra-low",
        type=float,
        default=0.1,
        help="Infrasound lower cutoff in Hz (default: 0.1)",
    )
    parser.add_argument(
        "--infra-high",
        type=float,
        default=20.0,
        help="Infrasound upper cutoff in Hz (default: 20.0)",
    )
    parser.add_argument(
        "--translate-infra",
        action="store_true",
        help="Translate filtered infrasound band to audible range using SSB upconversion",
    )
    parser.add_argument(
        "--translate-carrier",
        type=float,
        default=200.0,
        help="Carrier frequency in Hz for translation (default: 200 Hz)",
    )
    parser.add_argument(
        "--translate-gain",
        type=float,
        default=1.0,
        help="Gain to apply after translation (default: 1.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        play_wav_file(
            args.path,
            block=args.block,
            infra_only=args.infra_only,
            infra_low_hz=args.infra_low,
            infra_high_hz=args.infra_high,
            translate_infra=args.translate_infra,
            translate_carrier_hz=args.translate_carrier,
            translate_gain=args.translate_gain,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 99

    return 0


if __name__ == "__main__":
    sys.exit(main())
