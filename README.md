# STEInfrasound

A suite of Python tools for monitoring and analyzing infrasound (sound below 20 Hz) using USB audio input or WAV files. Includes real-time spectrum analyzers and audio playback utilities with infrasound filtering and frequency translation capabilities.

## rPi Setup

### Create SD Card Image

Create SD Card using Raspberry Pi Imager:

- Raspberry Pi 4 Model B — 4GB RAM
- Debian Trixie
- Set up initial WiFi
- Turn on SSH with password authentication
- Turn on Raspberry Pi Connect

### Install software onto rPi

```bash
cd ~/Downloads
wget https://raw.githubusercontent.com/DonLakeFlyer/STEInfrasound/main/rpi_setup.sh
chmod +x rpi_setup.sh
./rpi_setup.sh
```

### Setup SMB

Configure Samba file sharing for remote access to the Raspberry Pi home directory:

```bash
sudo apt install samba samba-common-bin

sudo nano /etc/samba/smb.conf
```

Add this configuration:

```ini
[home]
   path = /home/pi
   browseable = yes
   writeable = yes
   only guest = no
   create mask = 0775
   directory mask = 0775
   public = yes
```

Set Samba password and restart:

```bash
sudo smbpasswd -a pi
sudo systemctl restart smbd
```

### Setup Journaling

Configure persistent systemd journal logging:

```bash
sudo mkdir -p /var/log/journal
sudo mkdir -p /var/log/journal/$(cat /etc/machine-id)
sudo chown root:systemd-journal /var/log/journal
sudo chmod 2755 /var/log/journal
sudo setfacl -R -nm g:systemd-journal:rx /var/log/journal
sudo setfacl -R -nm g:adm:rx /var/log/journal
sudo setfacl -R -nm u:pi:rx /var/log/journal

sudo mkdir -p /etc/systemd/journald.conf.d
sudo nano /etc/systemd/journald.conf.d/override.conf
```

Add this configuration:

```ini
[Journal]
Storage=persistent
```

Finalize setup:

```bash
sudo rm -rf /var/log/journal
sudo systemd-tmpfiles --create
reboot
systemctl status systemd-journald
```

### Startup service

The Raspberry Pi is configured to automatically launch `infrasound.py` on boot using autostart for GUI applications:

```bash
# Make sure start script is executable
chmod +x /home/pi/repos/STEInfrasound/start_infrasound.sh

# Copy desktop file to autostart
mkdir -p ~/.config/autostart
cp /home/pi/repos/STEInfrasound/infrasound_display.desktop ~/.config/autostart/
chmod +x ~/.config/autostart/infrasound_display.desktop
```

To check if it will run on next boot:
```bash
ls -la ~/.config/autostart/
cat ~/.config/autostart/infrasound_display.desktop
```

Test the desktop file manually (before reboot):
```bash
gtk-launch infrasound_display.desktop
# Or
/home/pi/repos/STEInfrasound/start_infrasound.sh
```

Check the startup log after reboot:
```bash
cat /home/pi/infrasound_startup.log
```

To remove the old systemd user service (if you had it installed):
```bash
systemctl --user disable infrasound_display.service
systemctl --user stop infrasound_display.service
rm ~/.config/systemd/user/infrasound_display.service
```

View logs:
```bash
journalctl --user -xe
```

## Prerequisites

- Python 3.9 or newer
- `pip` for installing dependencies
- On macOS: `brew install portaudio` (required for sounddevice)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Applications

The live display apps share a common backend: audio capture, FFT, and the
adaptive noise floor live in `infrasound_core.py`, and the splash/error/menu
screens live in `infrasound_ui.py`. Each display app (`infrasound.py`,
`infrasound_detections.py`) is a thin front-end built on top of these.

### Real-Time Infrasound Display (`infrasound.py`)

Live infrasound spectrum analyzer designed for Raspberry Pi with 3.5" display. This is the main application that runs on the Raspberry Pi at startup.

```bash
python infrasound.py
```

**Test mode** (no microphone needed — generates a synthetic 18 Hz elephant rumble with harmonics, cycling 8s on / 8s off):

```bash
python infrasound.py --test
```

**File replay mode** (analyze a recorded WAV file instead of live input):

```bash
python infrasound.py --file path/to/recording.wav
# Or pass the path directly (bare path => file mode; no path => live mic):
python infrasound.py path/to/recording.wav
```

**Options:**
- `--test` — Run with a synthetic 18 Hz rumble signal; no microphone required
- `--file PATH` — Replay and analyze a recorded WAV file instead of live audio input
- A bare path argument is also accepted and treated as `--file PATH`

**Features:**
- Displays real-time FFT of 10-50 Hz band (infrasound + elephant rumble harmonics)
- Adaptive noise floor with dB-above-background display
- Color-coded signal strength (cyan/yellow/orange/red)
- 0.5 Hz frequency resolution
- Optimized for 320x480 touchscreen displays
- Auto-detects Raspberry Pi platform
- Peak decay visualization

### Detection Display (`infrasound_detections.py`)

Event-triggered frequency-vs-time view. Instead of scrolling continuously, it
draws one chart per detection: a chart starts when a clump of power at least 16
dB above the adaptive noise floor, more than ~2.5 Hz wide, persists for ~1.5 s.
The chart's X axis then grows for as long as the detection continues and freezes
the moment the clump drops out. The next qualifying clump clears the chart and
starts a fresh one. By default the raw bar spectrum (the same view as
`infrasound.py`) is shown above the detection heatmap so you can watch raw data
and detections at the same time.

False-positive filtering (on by default) rejects obvious wind/broadband noise:
frames where energy is smeared across a large fraction of the band, or where the
strongest clump is implausibly wide for a rumble harmonic, do not start a
detection. A broadband frame in the *middle* of an ongoing detection is treated
as a brief gap and does not end it (so a wind gust over a real rumble won't clip
the event).

```bash
# Live microphone (raw bars + detection heatmap)
python infrasound_detections.py

# Synthetic rumble (no microphone needed)
python infrasound_detections.py --test

# Replay and analyze a recorded WAV file
python infrasound_detections.py --file path/to/recording.wav
# Or pass the path directly (bare path => file mode; no path => live mic):
python infrasound_detections.py path/to/recording.wav

# Raise the detection threshold to 20 dB above the noise floor
python infrasound_detections.py --threshold 20

# Review a long recording faster than real time (silent 8x fast-scan)
python infrasound_detections.py --file path/to/recording.wav --speed 8

# Disable broadband/wind false-positive filtering
python infrasound_detections.py --no-fp-filter

# Hide the raw bar spectrum (detection heatmap only)
python infrasound_detections.py --no-bars
```

**Options:**
- `--test` — Run with a synthetic 18 Hz rumble signal; no microphone required
- `--file PATH` — Replay and analyze a recorded WAV file instead of live audio input
- `--threshold DB` — dB above noise floor required to trigger a detection (default: 16)
- `--speed N` — File mode only: scan faster than real time (whole number, N ≥ 1). Above 1× audio playback is skipped (silent fast-scan); every analysis window is still processed the same way as normal playback
- `--no-fp-filter` — Disable the wind/broadband false-positive filter (on by default)
- `--no-bars` — Hide the raw bar spectrum and show only the detection heatmap
- A bare path argument is also accepted and treated as `--file PATH`

**Axes / display:**
- Top chart (unless `--no-bars`) = the live bar spectrum from `infrasound.py`
- X axis = elapsed time within the current detection event (not real-time scrolling)
- Y axis = frequency (Hz), auto-zoomed to the band that contains data
- Greyscale intensity = dB above noise floor (threshold = light, louder = darker)
- A single event's chart is capped at 20 s wide

### WAV File Playback with Processing (`ste_infrasound.py`)

Play WAV files with optional infrasound filtering and frequency translation.

```bash
# Basic playback
python ste_infrasound.py audio/example.wav

# Filter to infrasound band only
python ste_infrasound.py audio/example.wav --infra-only --infra-low 0.1 --infra-high 20

# Translate infrasound to audible range (SSB upconversion)
python ste_infrasound.py audio/example.wav --translate-infra --translate-carrier 200 --translate-gain 2.0
```

**Options:**
- `--no-block` — Return immediately after starting playback
- `--infra-only` — Filter to infrasound band before playback
- `--infra-low FREQ` — Infrasound lower cutoff in Hz (default: 0.1)
- `--infra-high FREQ` — Infrasound upper cutoff in Hz (default: 20.0)
- `--translate-infra` — Translate infrasound to audible range using SSB
- `--translate-carrier FREQ` — Carrier frequency for translation (default: 200 Hz)
- `--translate-gain GAIN` — Gain multiplier after translation (default: 1.0)

## Troubleshooting

### Audio Issues
- **No sound (ste_infrasound.py):** Check system volume and output device
- **`ValueError: Only .wav files are supported`:** Convert audio to uncompressed PCM WAV format
- **PortAudio errors:** On macOS, run `brew install portaudio`

### Display Issues
- **GUI doesn't start on Raspberry Pi:** Check `~/infrasound_startup.log` for errors
- **Wrong screen size:** Verify DISPLAY environment variable is set correctly

### Find rPi over WiFi

```bash
ping -c 1 raspberrypi.local
```

### Raspberry Pi Connect

Allows remote access to the Raspberry Pi desktop from a web browser.

### Turn off removable media popup

Disable the popup that appears when you plug in the LS-P5:

```bash
nano ~/.config/pcmanfm/LXDE-pi/pcmanfm.conf
```

Add this configuration:

```ini
[volume]
mount_on_startup=0
mount_removable=0
autorun=0
```

Restart the panel:

```bash
lxpanelctl restart
```
