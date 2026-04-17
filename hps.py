import numpy as np
from collections import namedtuple

DetectionEvent = namedtuple('DetectionEvent', ['timestamp', 'fundamental_hz', 'fundamental_bin', 'hps_peak_db'])


def compute_hps(magnitude_db, num_harmonics=3):
    """Compute the Harmonic Product Spectrum in log domain.

    In log domain, the product becomes a sum:
    HPS_dB(f) = |X(f)|_dB + |X(2f)|_dB + |X(3f)|_dB + ...

    Peaks in the output correspond to frequencies whose integer harmonics
    are all present.

    Args:
        magnitude_db: 1-D array of magnitude spectrum in dB.
        num_harmonics: Number of harmonic factors (R). Default 3 means
            the sum uses |X(f)| + |X(2f)| + |X(3f)| in dB.

    Returns:
        1-D array (shorter than input) containing the HPS sum values in dB.
    """
    min_len = len(magnitude_db) // num_harmonics
    hps = magnitude_db[:min_len].copy()
    for h in range(2, num_harmonics + 1):
        downsampled = magnitude_db[::h][:min_len]
        hps += downsampled
    return hps


class HPSDetector:
    """Stateful HPS-based harmonic detector for elephant rumbles.

    Accepts a full magnitude spectrum each frame, performs HPS,
    temporal smoothing, and persistence gating.
    Returns a DetectionEvent when a harmonic fundamental is confirmed,
    or None.

    Args:
        freqs: Frequency array corresponding to the magnitude spectrum bins.
        fund_low: Lower bound of fundamental search range (Hz).
        fund_high: Upper bound of fundamental search range (Hz).
        num_harmonics: Number of HPS harmonic factors.
        threshold_db: HPS peak must exceed median by this many dB to fire.
        persistence_frames: Peak must persist this many consecutive frames.
        median_frames: Number of frames for temporal median smoothing.
    """

    def __init__(self, freqs, fund_low=10.0, fund_high=25.0,
                 num_harmonics=3, threshold_db=15.0,
                 persistence_frames=3, median_frames=5):
        self.freqs = freqs
        self.fund_low = fund_low
        self.fund_high = fund_high
        self.num_harmonics = num_harmonics
        self.threshold_db = threshold_db
        self.persistence_frames = persistence_frames
        self.median_frames = median_frames

        # HPS output length
        self.hps_len = len(freqs) // num_harmonics

        # Frequency mask for fundamental search (applied to HPS output)
        hps_freqs = freqs[:self.hps_len]
        self.search_mask = (hps_freqs >= fund_low) & (hps_freqs <= fund_high)

        # Ring buffer for temporal smoothing
        self.hps_history = []

        # Persistence counter for the detected bin
        self.persist_count = 0
        self.last_peak_bin = -1
        self._armed = True  # Must drop below threshold before re-firing

    def update(self, magnitude, timestamp):
        """Process one frame of magnitude spectrum.

        Args:
            magnitude: Full magnitude spectrum array (not just infrasound band).
            timestamp: Current time (e.g. time.time()).

        Returns:
            DetectionEvent or None.
        """
        # Convert to dB and compute HPS (sum in log domain)
        mag_db = 20 * np.log10(magnitude + 1e-10)
        hps = compute_hps(mag_db, self.num_harmonics)

        # Add to history for temporal median
        self.hps_history.append(hps)
        if len(self.hps_history) > self.median_frames:
            self.hps_history.pop(0)

        # Not enough history yet
        if len(self.hps_history) < self.median_frames:
            return None

        # Temporal median smoothing
        smoothed = np.median(self.hps_history, axis=0)

        # Search for peak in fundamental range
        search_values = smoothed[self.search_mask]
        if len(search_values) == 0:
            return None

        median_level = np.median(search_values)
        peak_idx_in_search = np.argmax(search_values)
        peak_value = search_values[peak_idx_in_search]

        # Check threshold
        if peak_value - median_level < self.threshold_db:
            self.persist_count = 0
            self.last_peak_bin = -1
            self._armed = True  # Signal gone — re-arm for next detection
            return None

        # Already fired for this sustained signal — wait for it to end
        if not self._armed:
            return None

        # Map back to global bin index
        search_indices = np.where(self.search_mask)[0]
        global_bin = search_indices[peak_idx_in_search]

        # Persistence check: same bin (±1) must fire consecutively
        if abs(global_bin - self.last_peak_bin) <= 1:
            self.persist_count += 1
        else:
            self.persist_count = 1
        self.last_peak_bin = global_bin

        if self.persist_count >= self.persistence_frames:
            self.persist_count = 0
            self._armed = False  # Disarm until signal drops below threshold
            freq_hz = self.freqs[global_bin]
            return DetectionEvent(
                timestamp=timestamp,
                fundamental_hz=freq_hz,
                fundamental_bin=global_bin,
                hps_peak_db=peak_value - median_level
            )

        return None
