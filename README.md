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

### Real-Time Infrasound Display (`infrasound.py`)

Live infrasound spectrum analyzer designed for Raspberry Pi with 3.5" display. This is the main application that runs on the Raspberry Pi at startup.

```bash
python infrasound.py
```

**Test mode** (no microphone needed — generates a synthetic 18 Hz elephant rumble with harmonics, cycling 8s on / 8s off):

```bash
python infrasound.py --test
```

**Features:**
- Displays real-time FFT of 10-50 Hz band (infrasound + elephant rumble harmonics)
- Adaptive noise floor with dB-above-background display
- Color-coded signal strength (cyan/yellow/orange/red)
- 0.5 Hz frequency resolution
- Optimized for 320x480 touchscreen displays
- Auto-detects Raspberry Pi platform
- Peak decay visualization

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

### General-Purpose Live FFT (`realtime_fft.py`)

Flexible real-time spectrum analyzer with USB audio input support.

```bash
# List available audio devices
python realtime_fft.py --list-devices

# Audible spectrum from USB device
python realtime_fft.py --device usb --sample-rate 48000 --fft-size 4096

# Infrasound spectrum in fullscreen
python realtime_fft.py --spectrum-type infrasound --fullscreen
```

**Options:**
- `--device NAME` — Device name substring (e.g., 'usb')
- `--sample-rate RATE` — Sample rate in Hz (default: 48000)
- `--fft-size SIZE` — FFT size (default: 2048)
- `--spectrum-type TYPE` — 'audible' or 'infrasound' (default: audible)
- `--fullscreen` — Display in fullscreen mode
- `--y-min DB` — Lower Y limit in dBFS (default: -120)
- `--y-max DB` — Upper Y limit in dBFS (default: 10)

### Elephant Rumble Monitor (`realtime_infrasound.py`)

Specialized real-time display for elephant infrasound rumbles (5-25 Hz).

```bash
python realtime_infrasound.py
```

**Features:**
- Fixed 5-25 Hz display range
- Auto-selects best available sample rate (8-48 kHz)
- 16000-point FFT for high frequency resolution

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
