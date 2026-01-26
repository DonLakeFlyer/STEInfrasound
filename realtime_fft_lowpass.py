"""Live USB audio FFT display with low-pass filtering and decimation.

Processes 1-second chunks, applies 25 Hz low-pass filter, decimates to 100 Hz,
and displays the FFT spectrum.
"""

from __future__ import annotations

import argparse
import queue
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from scipy import signal


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture audio, low-pass filter, decimate, and show FFT."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available input devices and exit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Case-insensitive substring of the input device name.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=48_000.0,
        help="Sample rate in Hz (default: 48000).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of input channels (default: 1).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=1000,
        help="UI update interval in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=-120.0,
        help="Lower Y limit in dBFS (default: -120).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=10.0,
        help="Upper Y limit in dBFS (default: 10).",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def _list_input_devices() -> None:
    """Print all available input devices."""
    devices = sd.query_devices()
    print("Available input devices:")
    print("-" * 60)
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            name = dev.get("name", "Unknown")
            channels = dev.get("max_input_channels", 0)
            samplerate = dev.get("default_samplerate", 0)
            print(f"[{idx}] {name}")
            print(f"     Channels: {channels}, Sample Rate: {samplerate} Hz")
    print("-" * 60)


def _resolve_input_device(name: Optional[str]) -> Optional[int]:
    """Return a device index matching *name* or None for default."""
    candidates = sd.query_devices()
    input_devices = [
        idx for idx, dev in enumerate(candidates)
        if dev.get("max_input_channels", 0) > 0
    ]

    if name is None:
        if len(input_devices) == 1:
            return input_devices[0]
        return None

    needle = name.casefold()
    for idx in input_devices:
        if needle in candidates[idx].get("name", "").casefold():
            return idx

    raise ValueError(f"No input device containing '{name}' was found.")


def _prepare_axes(
    freqs: np.ndarray,
    y_min: float,
    y_max: float,
) -> tuple[plt.Figure, plt.Axes, plt.Line2D]:
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot(
        freqs,
        np.full_like(freqs, y_min, dtype=np.float32),
        color='cyan',
        linewidth=1.0,
    )
    ax.set_xlim(0, freqs[-1] if freqs.size else 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_title("Low-Pass Filtered FFT (0-25 Hz)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    fig.patch.set_facecolor('black')
    return fig, ax, line


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_devices:
        _list_input_devices()
        return 0

    if args.sample_rate <= 0:
        print("Sample rate must be positive.", file=sys.stderr)
        return 2
    if args.channels <= 0:
        print("Channels must be positive.", file=sys.stderr)
        return 2

    try:
        device_index = _resolve_input_device(args.device)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    # Process 1 second of data at a time
    chunk_size = int(args.sample_rate)
    decimation_factor = int(args.sample_rate / 100)  # Decimate to 100 Hz
    decimated_rate = args.sample_rate / decimation_factor
    decimated_size = chunk_size // decimation_factor

    # Design low-pass filter (25 Hz cutoff)
    nyquist = args.sample_rate / 2.0
    cutoff = 25.0
    filter_order = 8
    sos = signal.butter(filter_order, cutoff / nyquist, btype='low', output='sos')

    # Prepare FFT frequency bins for decimated data
    freqs = np.fft.rfftfreq(decimated_size, 1.0 / decimated_rate)
    window = np.hanning(decimated_size).astype(np.float32)

    fig, ax, line = _prepare_axes(freqs, args.y_min, args.y_max)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
    buffer: list[np.ndarray] = []

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(status, file=sys.stderr)
        buffer.append(indata.copy())

        # Accumulate until we have 1 second of data
        total_frames = sum(b.shape[0] for b in buffer)
        if total_frames >= chunk_size:
            chunk = np.concatenate(buffer, axis=0)[:chunk_size]
            buffer.clear()
            try:
                audio_queue.put_nowait(chunk)
            except queue.Full:
                pass

    latest_chunk: Optional[np.ndarray] = None

    def _update(_frame):
        nonlocal latest_chunk
        try:
            while True:
                latest_chunk = audio_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_chunk is None:
            return (line,)

        # Convert to mono
        mono = latest_chunk.mean(axis=1) if latest_chunk.ndim > 1 else latest_chunk

        # Apply low-pass filter
        filtered = signal.sosfilt(sos, mono)

        # Decimate to 100 Hz
        decimated = signal.decimate(filtered, decimation_factor, ftype='fir', zero_phase=True)

        # Compute FFT
        windowed = decimated * window
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)
        db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        line.set_ydata(db)
        return (line,)

    try:
        with sd.InputStream(
            samplerate=float(args.sample_rate),
            blocksize=4096,
            device=device_index,
            channels=args.channels,
            dtype="float32",
            callback=_callback,
        ):
            _ = FuncAnimation(
                fig,
                _update,
                interval=args.interval_ms,
                blit=False,
                cache_frame_data=False,
            )
            plt.show()
    except Exception as exc:
        print(f"Failed to start input stream: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
