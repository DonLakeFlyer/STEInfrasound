import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.rcParams['toolbar'] = 'none'
# Set backend explicitly for headless/service operation
if os.environ.get('DISPLAY'):
    try:
        matplotlib.use('TkAgg')
    except:
        try:
            matplotlib.use('Qt5Agg')
        except:
            pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import platform
import os
import subprocess
import sys
import traceback
import signal

# Add logging to help debug
def log(msg):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# Signal handler for clean shutdown
def signal_handler(sig, frame):
    log(f"Received signal {sig}, shutting down...")
    global stream
    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except:
            pass
    plt.close('all')
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

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

# Screen size settings for 3.5" 320x480 display
SCREEN_DPI = 100
SCREEN_WIDTH = 4.8  # inches (480 pixels / 100 dpi)
SCREEN_HEIGHT = 3.2  # inches (320 pixels / 100 dpi)

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

    fig_splash = None
    ax_splash = None

    # Give X server extra time to be ready when running as service
    if IS_RPI and os.environ.get('DISPLAY'):
        log("Waiting for display to be ready...")
        time.sleep(2)

    try:
        log("Creating splash screen...")
        fig_splash = plt.figure(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)
        fig_splash.patch.set_facecolor('#1a1a1a')
        ax_splash = fig_splash.add_subplot(111)
        ax_splash.set_xlim(0, 1)
        ax_splash.set_ylim(0, 1)
        ax_splash.axis('off')

        # Main message - smaller fonts for small screen
        ax_splash.text(0.5, 0.6, 'Infrasound Monitor',
                       ha='center', va='center', fontsize=14, color='cyan', weight='bold')
        ax_splash.text(0.5, 0.5, 'Connecting...',
                       ha='center', va='center', fontsize=10, color='white')
        ax_splash.text(0.5, 0.42, 'Please wait',
                       ha='center', va='center', fontsize=8, color='gray')

        # Maximize window - platform specific
        try:
            mng = plt.get_current_fig_manager()
            if IS_RPI:
                # Raspberry Pi fullscreen
                try:
                    mng.window.attributes('-fullscreen', True)
                    log("Set fullscreen mode")
                except Exception as e:
                    log(f"Fullscreen method 1 failed: {e}")
                    try:
                        mng.full_screen_toggle()
                        log("Set fullscreen via toggle")
                    except Exception as e2:
                        log(f"Fullscreen method 2 failed: {e2}")
            else:
                # Desktop maximize
                try:
                    mng.window.state('zoomed')  # Windows
                except:
                    try:
                        mng.full_screen_toggle()  # Some backends
                    except:
                        pass
        except Exception as e:
            log(f"Error setting window size: {e}")

        plt.tight_layout()
        log("Showing splash screen...")
        plt.show(block=False)
        plt.draw()
        plt.pause(0.5)
        log("Splash screen displayed")

    except Exception as e:
        log(f"ERROR creating splash screen: {e}")
        traceback.print_exc()
        # Continue anyway - audio connection is more important

    # Try to connect to audio device during countdown
    for i in range(10, 0, -1):
        log(f"Loop iteration: {i} seconds remaining, audio_connected={audio_connected}")

        if audio_connected:
            log("Already connected, breaking loop")
            break

        # Update display if we have a splash screen
        if ax_splash is not None:
            try:
                ax_splash.texts[2].set_text(f'{i}s...')
                plt.draw()
                plt.pause(0.1)
            except Exception as e:
                log(f"Error updating splash: {e}")

        # Attempt connection
        try:
            log(f"Attempting audio connection (attempt {11-i}/10)...")
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=audio_callback,
                blocksize=int(SAMPLE_RATE / UPDATE_RATE),
                dtype='float32'
            )
            stream.start()
            audio_connected = True
            log(f"SUCCESS! Audio stream started at {SAMPLE_RATE} Hz")
            log(f"FFT size: {FFT_SIZE} samples ({FFT_SIZE/SAMPLE_RATE:.1f} seconds)")
            log(f"Frequency resolution: {FREQ_RESOLUTION} Hz")
            log(f"Update rate: {UPDATE_RATE} Hz")

            # Show success message briefly
            if ax_splash is not None:
                try:
                    ax_splash.texts[1].set_text('Connected!')
                    ax_splash.texts[1].set_color('lime')
                    ax_splash.texts[2].set_text('Starting...')
                    plt.draw()
                    plt.pause(1)
                except Exception as e:
                    log(f"Error showing success message: {e}")
            break

        except Exception as e:
            audio_error = str(e)
            log(f"Audio connection attempt {11-i} failed: {e}")

            # Wait before next attempt (only if not on last iteration)
            if i > 1:
                try:
                    time.sleep(0.9)
                except Exception as sleep_err:
                    log(f"Sleep failed: {sleep_err}")

    log(f"Exiting splash screen loop. audio_connected={audio_connected}")

    # Close splash screen
    if fig_splash is not None:
        try:
            plt.close(fig_splash)
            log("Splash screen closed")
        except Exception as e:
            log(f"Error closing splash: {e}")

    # Return connection status
    log(f"Returning connection status: {audio_connected}")
    return audio_connected

def show_error_screen(error_message):
    """Show a full-screen error window."""
    log(f"Showing error screen: {error_message}")

    try:
        fig_error = plt.figure(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)
        fig_error.patch.set_facecolor('#1a1a1a')
        ax_error = fig_error.add_subplot(111)
        ax_error.set_xlim(0, 1)
        ax_error.set_ylim(0, 1)
        ax_error.axis('off')

        # Error message - smaller fonts for small screen
        ax_error.text(0.5, 0.75, '⚠ Audio Error',
                  ha='center', va='center', fontsize=14, color='red', weight='bold')
        ax_error.text(0.5, 0.68, 'Cannot connect',
                  ha='center', va='center', fontsize=9, color='white')

        # Truncate long error messages
        short_error = error_message[:40] + '...' if len(error_message) > 40 else error_message
        ax_error.text(0.5, 0.62, short_error,
                  ha='center', va='center', fontsize=6, color='orange', style='italic')

        ax_error.text(0.5, 0.53, 'Check:',
                  ha='center', va='center', fontsize=8, color='gray')
        ax_error.text(0.5, 0.48, '• LS-P5 is ON',
                  ha='center', va='center', fontsize=7, color='lightgray')
        ax_error.text(0.5, 0.43, '• LS-P5 connected',
                  ha='center', va='center', fontsize=7, color='lightgray')
        ax_error.text(0.5, 0.38, '• Batteries good',
                  ha='center', va='center', fontsize=7, color='lightgray')

        if IS_RPI:
            ax_error.text(0.5, 0.28, 'Use buttons or close',
                      ha='center', va='center', fontsize=6, color='gray', style='italic')
        else:
            ax_error.text(0.5, 0.28, 'Close to exit',
                      ha='center', va='center', fontsize=6, color='gray', style='italic')

        # Add buttons
        from matplotlib.widgets import Button

        if IS_RPI:
            # Reboot button - larger and positioned for small screen
            ax_reboot = plt.axes([0.1, 0.08, 0.35, 0.24])
            btn_reboot = Button(ax_reboot, 'Reboot', color='#ff6b6b', hovercolor='#ff5252')

            def reboot(event):
                print("Rebooting Raspberry Pi...")
                plt.close('all')
                subprocess.run(['sudo', 'reboot'], check=False)

            btn_reboot.on_clicked(reboot)
            btn_reboot.label.set_fontsize(9)

            # Shutdown button - larger and positioned for small screen
            ax_shutdown = plt.axes([0.55, 0.08, 0.35, 0.24])
            btn_shutdown = Button(ax_shutdown, 'Shutdown', color='#6b6bff', hovercolor='#5252ff')

            def shutdown(event):
                print("Shutting down Raspberry Pi...")
                plt.close('all')
                subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)

            btn_shutdown.on_clicked(shutdown)
            btn_shutdown.label.set_fontsize(9)
        else:
            # Exit button for non-RPi platforms
            ax_exit = plt.axes([0.325, 0.08, 0.35, 0.24])
            btn_exit = Button(ax_exit, 'Exit', color='#ff6b6b', hovercolor='#ff5252')

            def exit_program(event):
                print("Exiting program...")
                plt.close('all')
                sys.exit(1)

            btn_exit.on_clicked(exit_program)
            btn_exit.label.set_fontsize(9)

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
        log("Displaying error screen...")
        plt.show()

    except Exception as e:
        log(f"ERROR creating error screen: {e}")
        traceback.print_exc()

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
fig, ax = plt.subplots(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)

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
ax.set_xlabel("Frequency (Hz)", fontsize=7)
ax.set_ylabel("dB", fontsize=7)
ax.set_title(f"Infrasound {LOW_HZ}-{HIGH_HZ} Hz", fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=6)

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
