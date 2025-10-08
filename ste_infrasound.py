"""Play a WAV file through the system's default audio output.

This script uses the `simpleaudio` package to stream audio data read via Python's
built-in `wave` module. It currently supports uncompressed PCM WAV files.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import wave

try:
    import simpleaudio as sa
except ImportError as exc:  # pragma: no cover - import error only hit at runtime
    raise SystemExit(
        "simpleaudio is required. Install it with 'pip install -r requirements.txt'"
    ) from exc


SUPPORTED_SAMPLE_WIDTHS = {1, 2, 3, 4}


def play_wav_file(path: pathlib.Path, *, block: bool = True) -> None:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        play_wav_file(args.path, block=args.block)
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
