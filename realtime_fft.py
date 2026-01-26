"""Live USB audio FFT display using sounddevice and matplotlib.

Start the script, choose a USB input device (by substring match), and a real-time
spectrum will be displayed. Close the window to stop.
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture audio from an input device and show a live FFT."
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
        help="Case-insensitive substring of the input device name (e.g. 'usb').",
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
        help="Number of input channels to capture (default: 1).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1024,
        help="Block size for the PortAudio stream (default: 1024 frames).",
    )
    parser.add_argument(
        "--fft-size",
        type=int,
        default=2048,
        help="FFT size (power of two recommended; default: 2048).",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=None,
        help="Max frequency to display in Hz (default: Nyquist).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=30,
        help="UI update interval in milliseconds (default: 30).",
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
    parser.add_argument(
        "--spectrum-type",
        type=str,
        choices=['audible', 'infrasound'],
        default='audible',
        help="Type of spectrum to display: 'audible' or 'infrasound'.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display the chart in fullscreen mode.",
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
            print(f"     Use: --device '{name}'")
    print("-" * 60)


def _resolve_input_device(name: Optional[str]) -> Optional[int]:
    """Return a device index matching *name*, the only device if one exists, or None for default."""
    candidates = sd.query_devices()
    input_devices = [idx for idx, dev in enumerate(candidates) if dev.get("max_input_channels", 0) > 0]

    if name is None:
        # If only one input device exists, use it automatically
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
    title: str,
    x_min: float,
    x_max: float,
) -> tuple[plt.Figure, plt.Axes, plt.Line2D, plt.Line2D]:
    fig, ax = plt.subplots(figsize=(12, 6))
    (line,) = ax.plot(freqs, np.full_like(freqs, y_min, dtype=np.float32), color='lime', linewidth=0.8, label='Spectrum')
    (peak_line,) = ax.plot(freqs, np.full_like(freqs, y_min, dtype=np.float32), color='yellow', linewidth=0.8, label='Peak Hold')
    line.set_visible(False)  # hide live spectrum; show only peak hold
    if x_max <= x_min:
        pad = max(1.0, x_min * 0.1)
        ax.set_xlim(max(x_min - pad, 0.0), x_min + pad)
    else:
        ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    # ax.legend(loc='upper right')  # legend removed

    # Set the background color of the plot to black and text colors to white
    ax.set_facecolor('black')  # Change axes background to black
    ax.tick_params(colors='white')  # Change tick colors to white
    ax.xaxis.label.set_color('white')  # Change x-axis label color to white
    ax.yaxis.label.set_color('white')  # Change y-axis label color to white
    ax.title.set_color('white')  # Change title color to white
    fig.patch.set_facecolor('black')  # Change figure background to black

    ax.margins(x=0, y=0)  # remove blank margins
    fig.tight_layout(pad=0.2)  # tighten layout

    return fig, ax, line, peak_line


def _compute_spectrum(
    chunk: np.ndarray,
    fft_size: int,
    window: np.ndarray,
    freq_mask: np.ndarray,
) -> np.ndarray:
    """Compute windowed magnitude spectrum in dBFS for a mono chunk."""
    mono = chunk.mean(axis=1)

    if mono.size < fft_size:
        padded = np.zeros(fft_size, dtype=np.float32)
        padded[-mono.size :] = mono
    else:
        padded = mono[-fft_size:]

    spectrum = np.fft.rfft(padded * window)
    mag = np.abs(spectrum)
    db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return db[freq_mask]


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
    if args.fft_size <= 0:
        print("FFT size must be positive.", file=sys.stderr)
        return 2

    try:
        device_index = _resolve_input_device(args.device)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    nyquist = args.sample_rate / 2.0
    if args.spectrum_type == 'audible':
        min_freq = 20.0
        max_freq = min(20_000.0, nyquist)
        spectrum_title = "Audible Spectrum"
    else:
        min_freq = 0.0
        max_freq = min(20.0, nyquist)
        spectrum_title = "Infrasound Spectrum"
    max_freq = max(min(max_freq, nyquist), min_freq)

    window = np.hanning(args.fft_size).astype(np.float32)
    freqs_full = np.fft.rfftfreq(args.fft_size, 1.0 / float(args.sample_rate))
    freq_mask = (freqs_full >= min_freq) & (freqs_full <= max_freq)
    if not np.any(freq_mask):
        freq_mask[0] = True  # ensure at least DC bin is kept
    freqs = freqs_full[freq_mask]
    # print freqs for debugging
    for f in freqs:
        print(f)
    if freqs.size == 1:
        freqs = np.concatenate([freqs, np.array([max_freq], dtype=np.float32)])  # pad for visibility

    fig, ax, line, peak_line = _prepare_axes(
        freqs,
        args.y_min,
        args.y_max,
        spectrum_title,
        min_freq,
        max_freq,
    )
    if args.fullscreen:
        try:
            fig.canvas.manager.full_screen_toggle()
        except Exception:
            pass

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

    def _callback(indata: np.ndarray, frames: int, time, status) -> None:  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop frame if UI cannot keep up

    latest_chunk: Optional[np.ndarray] = None
    peak_hold = np.full(len(freqs), args.y_min, dtype=np.float32)
    decay_rate = 2.0  # dB per frame

    def _update(_frame):
        nonlocal latest_chunk, peak_hold
        try:
            while True:
                latest_chunk = audio_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_chunk is None:
            return line, peak_line

        spectrum_db = _compute_spectrum(latest_chunk, args.fft_size, window, freq_mask)
        if spectrum_db.size < len(freqs):
            spectrum_db = np.pad(spectrum_db, (0, len(freqs) - spectrum_db.size), constant_values=args.y_min)
        line.set_ydata(spectrum_db)

        # Update peak hold: keep max of current spectrum or decayed peaks
        peak_hold = np.maximum(spectrum_db, peak_hold - decay_rate)
        peak_line.set_ydata(peak_hold)

        return line, peak_line

    try:
        with sd.InputStream(
            samplerate=float(args.sample_rate),
            blocksize=args.blocksize,
            device=device_index,
            channels=args.channels,
            dtype="float32",
            callback=_callback,
        ):
            _ = FuncAnimation(fig, _update, interval=args.interval_ms, blit=False, cache_frame_data=False)
            plt.show()
    except Exception as exc:
        print(f"Failed to start input stream: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
