# STEInfrasound WAV Player

A minimal Python utility that streams a WAV file to the system's default speakers using [`simpleaudio`](https://simpleaudio.readthedocs.io/).

## Prerequisites

- Python 3.9 or newer (tested on macOS)
- `pip` for installing dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```bash
python ste_infrasound.py path/to/file.wav
```

### Options

- `--no-block` — start playback and return immediately instead of waiting until the audio finishes.

### Example

```bash
python ste_infrasound.py audio/example.wav
```

If you see an error about missing `simpleaudio`, ensure the virtual environment is activated and the requirements were installed.

## Live FFT from USB input

Capture from an input device (e.g., USB mic) and display a real-time spectrum:

```bash
python realtime_fft.py --device usb --sample-rate 48000 --fft-size 4096
```

- Omit `--device` to use the default input. The value is a case-insensitive substring of the device name.
- Close the plot window to stop capture.
- On macOS, install PortAudio if `sounddevice` cannot find it: `brew install portaudio`.

## Troubleshooting

- **No sound:** Double-check your system volume and output device.
- **`ValueError: Only .wav files are supported`:** Convert your audio file to uncompressed PCM WAV format.
