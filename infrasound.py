import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.rcParams['toolbar'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import platform
import os
import subprocess

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
    except:
        pass
    return False

IS_RPI = is_raspberry_pi()

if IS_RPI:
    print("Running on Raspberry Pi")
else:
    print("Running on desktop/other platform")

# -----------------------------
# CONFIGURATION
# -----------------------------
SAMPLE_RATE = 44100  # Hz
UPDATE_RATE = 2  # Updates per second
FREQ_RESOLUTION = 0.5  # Hz resolution

# Calculate FFT size needed for 0.5 Hz resolution
# Resolution = sample_rate / fft_size, so fft_size = sample_rate / resolution
FFT_SIZE = int(SAMPLE_RATE / FREQ_RESOLUTION)

# Buffer size should be enough for smooth updates
BUFFER_SIZE = FFT_SIZE  # Keep full FFT size for resolution

# Infrasound range (below human hearing, typically < 20 Hz)
LOW_HZ = 0.5
HIGH_HZ = 20

# Display settings
DB_MIN = 0
DB_MAX = 100

# Peak decay settings
DECAY_RATE = 0.90  # Multiplier per update (0.90 = 10% decay per update)

# Rolling buffer - always keep the last FFT_SIZE samples
audio_buffer = np.zeros(BUFFER_SIZE, dtype='float32')

# Peak hold buffer - will be initialized after infra_freqs is defined
peak_values = None

# Audio stream - will be set if successful
stream = None
audio_error = None
audio_connected = False

# -----------------------------
# AUDIO CALLBACK
# -----------------------------
def audio_callback(indata, frames, time_info, status):
    global audio_buffer

    if status:
        print(f"Audio status: {status}")

    # Roll the buffer and append new data
    data = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -len(data))
    audio_buffer[-len(data):] = data

# -----------------------------
# SPLASH SCREEN
# -----------------------------
def show_splash_screen():
    """Show a full-screen splash screen and attempt audio connection."""
    global stream, audio_error, audio_connected

    fig_splash = plt.figure(figsize=(12, 8))
    fig_splash.patch.set_facecolor('#1a1a1a')
    ax_splash = fig_splash.add_subplot(111)
    ax_splash.set_xlim(0, 1)
    ax_splash.set_ylim(0, 1)
    ax_splash.axis('off')

    # Main message
    ax_splash.text(0.5, 0.55, 'Infrasound Spectrum Monitor',
                   ha='center', va='center', fontsize=32, color='cyan', weight='bold')
    ax_splash.text(0.5, 0.45, 'Connecting to audio device...',
                   ha='center', va='center', fontsize=18, color='white')
    ax_splash.text(0.5, 0.38, 'Please wait',
                   ha='center', va='center', fontsize=14, color='gray')

    # Maximize window - platform specific
    mng = plt.get_current_fig_manager()
    if IS_RPI:
        # Raspberry Pi fullscreen
        try:
            mng.full_screen_toggle()
        except:
            try:
                mng.window.attributes('-fullscreen', True)
            except:
                pass
    else:
        # Desktop maximize
        try:
            mng.window.state('zoomed')  # Windows
        except:
            try:
                mng.full_screen_toggle()  # Some backends
            except:
                pass

    plt.tight_layout()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

    # Try to connect to audio device during countdown
    for i in range(10, 0, -1):
        if not audio_connected:
            ax_splash.texts[2].set_text(f'Connecting... ({i} second{"s" if i > 1 else ""} remaining)')
            plt.draw()

            # Attempt connection
            try:
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    callback=audio_callback,
                    blocksize=int(SAMPLE_RATE / UPDATE_RATE),
                    dtype='float32'
                )
                stream.start()
                audio_connected = True
                print(f"Audio stream started at {SAMPLE_RATE} Hz")
                print(f"FFT size: {FFT_SIZE} samples ({FFT_SIZE/SAMPLE_RATE:.1f} seconds)")
                print(f"Frequency resolution: {FREQ_RESOLUTION} Hz")
                print(f"Update rate: {UPDATE_RATE} Hz")

                # Show success message briefly
                ax_splash.texts[1].set_text('Audio device connected!')
                ax_splash.texts[1].set_color('lime')
                ax_splash.texts[2].set_text('Starting visualization...')
                plt.draw()
                plt.pause(1)
                break
            except Exception as e:
                audio_error = str(e)
                # Keep trying for remaining time
                plt.pause(1)
        else:
            break

    plt.close(fig_splash)

    # Return connection status
    return audio_connected

def show_error_screen(error_message):
    """Show a full-screen error window."""
    fig_error = plt.figure(figsize=(12, 8))
    fig_error.patch.set_facecolor('#1a1a1a')
    ax_error = fig_error.add_subplot(111)
    ax_error.set_xlim(0, 1)
    ax_error.set_ylim(0, 1)
    ax_error.axis('off')

    # Error message
    ax_error.text(0.5, 0.6, '⚠ Audio Device Error',
                  ha='center', va='center', fontsize=32, color='red', weight='bold')
    ax_error.text(0.5, 0.5, 'Failed to connect to audio device',
                  ha='center', va='center', fontsize=18, color='white')
    ax_error.text(0.5, 0.42, f'Error: {error_message}',
                  ha='center', va='center', fontsize=12, color='orange', style='italic')
    ax_error.text(0.5, 0.3, 'Please check:',
                  ha='center', va='center', fontsize=14, color='gray')
    ax_error.text(0.5, 0.25, '• Check that OM Systems LS-P5 is turned on',
                  ha='center', va='center', fontsize=12, color='lightgray')
    ax_error.text(0.5, 0.21, '• Check the LS-P5 is connected to rPi',
                  ha='center', va='center', fontsize=12, color='lightgray')
    ax_error.text(0.5, 0.17, '• Check that batteries are good',
                  ha='center', va='center', fontsize=12, color='lightgray')

    if IS_RPI:
        ax_error.text(0.5, 0.05, 'Use buttons below or close window to exit',
                      ha='center', va='center', fontsize=10, color='gray', style='italic')
    else:
        ax_error.text(0.5, 0.05, 'Close this window to exit',
                      ha='center', va='center', fontsize=10, color='gray', style='italic')

    # Add buttons for Raspberry Pi only
    if IS_RPI:
        from matplotlib.widgets import Button

        # Reboot button
        ax_reboot = plt.axes([0.35, 0.1, 0.12, 0.05])
        btn_reboot = Button(ax_reboot, 'Reboot', color='#ff6b6b', hovercolor='#ff5252')

        def reboot(event):
            print("Rebooting Raspberry Pi...")
            plt.close('all')
            subprocess.run(['sudo', 'reboot'], check=False)

        btn_reboot.on_clicked(reboot)

        # Shutdown button
        ax_shutdown = plt.axes([0.53, 0.1, 0.12, 0.05])
        btn_shutdown = Button(ax_shutdown, 'Shutdown', color='#6b6bff', hovercolor='#5252ff')

        def shutdown(event):
            print("Shutting down Raspberry Pi...")
            plt.close('all')
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)

        btn_shutdown.on_clicked(shutdown)

    # Maximize window - platform specific
    mng = plt.get_current_fig_manager()
    if IS_RPI:
        # Raspberry Pi fullscreen
        try:
            mng.full_screen_toggle()
        except:
            try:
                mng.window.attributes('-fullscreen', True)
            except:
                pass
    else:
        # Desktop maximize
        try:
            mng.window.state('zoomed')  # Windows
        except:
            try:
                mng.full_screen_toggle()  # Some backends
            except:
                pass

    plt.tight_layout()
    plt.show()

# Show splash screen and attempt audio connection
if not show_splash_screen():
    # Connection failed after timeout
    print(f"Error opening audio stream: {audio_error}")
    show_error_screen(audio_error if audio_error else "Connection timeout")
    exit(1)

# -----------------------------
# SETUP PLOT
# -----------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

# Precompute frequency axis
freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
infrasound_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
infra_freqs = freqs[infrasound_mask]

# Initialize peak values
peak_values = np.zeros(len(infra_freqs))

# Initialize bars
bars = ax.bar(infra_freqs, np.zeros(len(infra_freqs)), width=FREQ_RESOLUTION * 0.8, color='cyan')
ax.set_xlim(LOW_HZ, HIGH_HZ)
ax.set_ylim(DB_MIN, DB_MAX)
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("Magnitude (dB)", fontsize=12)
ax.set_title(f"Infrasound Spectrum Monitor ({LOW_HZ}-{HIGH_HZ} Hz) | Resolution: {FREQ_RESOLUTION} Hz", fontsize=14)
ax.grid(True, alpha=0.3)

# -----------------------------
# UPDATE FUNCTION
# -----------------------------
def update_plot(frame):
    global audio_buffer, peak_values

    # Get a snapshot of the buffer
    audio_snapshot = audio_buffer.copy()

    # Apply window function and compute FFT
    windowed = audio_snapshot * np.hanning(FFT_SIZE)
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)

    # Extract infrasound band
    infra_mag = magnitude[infrasound_mask]

    # Convert to dB
    infra_db = 20 * np.log10(infra_mag + 1e-10)

    # Apply decay to existing peaks
    peak_values *= DECAY_RATE

    # Update peaks if current value is higher
    peak_values = np.maximum(peak_values, infra_db)

    # Clip to display range
    display_values = np.clip(peak_values, DB_MIN, DB_MAX)

    # Update bars
    for bar, height in zip(bars, display_values):
        bar.set_height(height)

    return bars

# -----------------------------
# RUN ANIMATION
# -----------------------------
ani = FuncAnimation(
    fig,
    update_plot,
    interval=int(1000 / UPDATE_RATE),  # milliseconds (500ms for 2 Hz)
    blit=True,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()

# Clean up
if stream is not None:
    stream.stop()
    stream.close()
