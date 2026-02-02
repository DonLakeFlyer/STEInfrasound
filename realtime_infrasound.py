import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
SAMPLE_RATES = [48000, 44100, 22050, 16000, 8000]  # Try in order of preference
FFT_SIZE = 32768
LOW_HZ = 5  # Elephant rumbles start around 5 Hz
HIGH_HZ = 25  # Elephant rumbles typically up to 24 Hz
DB_MIN = 0  # Minimum dB for display
DB_MAX = 80  # Maximum dB for display

# -----------------------------
# SETUP AUDIO STREAM
# -----------------------------
stream = None
SAMPLE_RATE = None

for rate in SAMPLE_RATES:
    try:
        stream = sd.InputStream(
            samplerate=rate,
            channels=1,
            blocksize=FFT_SIZE,
            dtype='float32'
        )
        stream.start()
        SAMPLE_RATE = rate
        print(f"Successfully opened audio stream at {rate} Hz")
        break
    except sd.PortAudioError as e:
        print(f"Sample rate {rate} Hz failed: {e}")
        continue

if stream is None:
    raise RuntimeError("Could not open audio stream with any supported sample rate")


# -----------------------------
# SETUP PLOT
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
bars = None

# Precompute frequency axis
freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
infrasound_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
infra_freqs = freqs[infrasound_mask]

while True:
    # Read audio block
    audio, _ = stream.read(FFT_SIZE)
    audio = audio[:, 0]

    # FFT
    spectrum = np.fft.rfft(audio * np.hanning(len(audio)))
    magnitude = np.abs(spectrum)

    # Extract infrasound band
    infra_mag = magnitude[infrasound_mask]

    # Convert to dB
    infra_db = 20 * np.log10(infra_mag + 1e-10)  # Add small value to avoid log(0)

    # Update histogram
    if bars is None:
        bars = ax.bar(infra_freqs, infra_db, width=0.8)
        ax.set_xlim(LOW_HZ, HIGH_HZ)
        ax.set_ylim(DB_MIN, DB_MAX)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title("Elephant Rumble Infrasound Spectrum (5–25 Hz)")
    else:
        for bar, h in zip(bars, infra_db):
            bar.set_height(h)

    plt.pause(0.001)
