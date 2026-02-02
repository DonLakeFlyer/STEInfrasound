import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.rcParams['toolbar'] = 'none'
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
# SETUP AUDIO STREAM
# -----------------------------
try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE / UPDATE_RATE),  # Process enough samples for update rate
        dtype='float32'
    )
    stream.start()
    print(f"Audio stream started at {SAMPLE_RATE} Hz")
    print(f"FFT size: {FFT_SIZE} samples ({FFT_SIZE/SAMPLE_RATE:.1f} seconds)")
    print(f"Frequency resolution: {FREQ_RESOLUTION} Hz")
    print(f"Update rate: {UPDATE_RATE} Hz")
except Exception as e:
    print(f"Error opening audio stream: {e}")
    raise

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
stream.stop()
stream.close()
