# HPS-Based Elephant Rumble Detection — Implementation Plan

## Goal

Add a Harmonic Product Spectrum (HPS) detector to `STEInfrasound` that flags
candidate elephant rumbles by exploiting their harmonic stack (fundamental +
2f + 3f ...), rather than relying only on magnitude thresholds in the 10–30 Hz
band.

**Display approach: Option 1 — overlay on the existing bar display.** HPS
runs as a pure *detector*, not as a second visualization. The current bar
display is untouched; when HPS fires, we draw a transient marker above the
bar corresponding to the detected fundamental. No second panel, no extra
plot area — just a visual annotation on top of what's already there.

## Why HPS for this problem

Rumbles are tonal and harmonic — typical fundamentals sit in 14–24 Hz with
strong energy at 2f and 3f. HPS downsamples the magnitude spectrum by
integer factors and multiplies the results together:

```
HPS(f) = |X(f)| · |X(2f)| · |X(3f)| · ... · |X(Rf)|
```

Energy that lines up on a true harmonic series reinforces across the product;
broadband noise (wind, distant traffic) and single-tone interference
(machinery with no overtones) get suppressed.

The practical win for our hardware: the EM272's gentle rolloff below ~20 Hz
attenuates the fundamental at 14–16 Hz. But the 2nd and 3rd harmonics at
28–48 Hz are in the mic's flat region and come through cleanly. HPS
multiplies those cleaner harmonics together with the attenuated fundamental
and still picks the correct fundamental frequency — even when the
fundamental alone would be below our current color threshold.

## Signal chain (implemented)

```
USB audio in (EM272 → LS-P5 line out or direct USB) @ 44.1 kHz
    ↓
Hann-windowed FFT, 88200 samples @ 44.1 kHz → 0.5 Hz/bin, 2.0 s window
    ↓
Magnitude spectrum (store for display + feed HPS)
    ↓
Per-bin adaptive noise floor (asymmetric EMA in infrasound.py)
    ↓
HPS in log domain (sum of dB at f, 2f, 3f), R = 3
    ↓
Peak search restricted to 10–25 Hz
    ↓
Temporal smoothing: median filter over last 5 HPS frames
    ↓
Detection gate: peak > 15 dB above median AND persists ≥ 3 frames
    ↓
Armed/disarmed gating: must drop below threshold to re-fire
    ↓
Event marker on existing bar display (fades over 3 s)
```

No decimation or spectral whitening — the log-domain HPS with
median-based thresholding handles noise rejection without them.

## Parameters (as implemented — tune in the field)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample rate | 44100 Hz | Native USB audio rate, no decimation needed |
| FFT size | 88200 | 0.5 Hz/bin, 2.0 s window |
| Window | Hann | Standard, low side-lobes |
| Update rate | 2 Hz | Matches existing display rate |
| HPS harmonics R | 3 | 4 adds little after rolloff of real overtones |
| Fundamental search range | 10–25 Hz | Prior on elephant rumbles |
| HPS threshold | 15 dB above median HPS | Tuned to reject false positives in field |
| Temporal median filter | 5 frames | Smooth FM without killing short calls |
| Persistence to fire | 3 consecutive frames | Reject transient spikes |
| Re-arm | Signal must drop below threshold | One detection per sustained rumble |

## Integration with existing `infrasound.py` (Option 1: overlay on bars)

Three surgical changes, chosen so the current display continues working
unchanged when HPS is disabled:

1. **New module: `hps.py`** — pure function `compute_hps(mag, num_harmonics)`
   plus a small `HPSDetector` class that owns temporal smoothing
   and persistence state. No drawing code here — detection only.
2. **Hook in the main loop** — after each FFT (and the existing adaptive
   noise-floor update), call `HPSDetector.update(mag)` with the raw FFT
   magnitude spectrum. On a positive detection it returns an event record:
   `(timestamp, fundamental_hz, fundamental_bin, hps_peak_db)`.
3. **Overlay on the bar display** — the display code keeps a short list of
   *active markers*. When an event fires we append one; each frame we
   render all active markers and age them out.

### Marker rendering (the only display change)

Draw directly on top of the existing bar display — no new panels, no
layout changes. For each active marker:

- Position: horizontally centered on the bar for `fundamental_bin`;
  vertically at the top of the chart (`DB_MAX`), anchored so the
  triangle points down at the relevant bar.
- Shape: small downward-pointing triangle (▼), ~6–8 px tall. Downward
  triangle reads as "pointing at this bar" without obscuring the bar's
  color.
- Color: white with a thin black outline for contrast against any bar
  color (cyan/yellow/orange/red).
- Fade: linear alpha fade from 1.0 → 0.0 over 3.0 seconds (≈ 6 display
  frames at 2 Hz). After alpha hits 0, drop the marker from the list.
- Optional tiny label (future work): render the detected frequency next
  to the marker (e.g. "18.0"), same white-on-outline styling, same fade.
  Not yet implemented — would be toggled via a CLI flag or key press.

The marker list is the only new state on the display side. Pseudocode:

```python
class BarDisplay:
    def __init__(self, ...):
        self.active_markers = []  # list of (fund_bin, birth_time, freq_hz)

    def on_detection(self, event):
        self.active_markers.append(
            (event.fundamental_bin, event.timestamp, event.fundamental_hz)
        )

    def draw_frame(self, now, bar_heights, bar_colors):
        # ... existing bar drawing unchanged ...
        self._draw_markers(now, bar_heights)

    def _draw_markers(self, now, bar_heights):
        surviving = []
        for bin_idx, birth, freq in self.active_markers:
            age = now - birth
            if age >= 3.0:
                continue  # aged out
            alpha = 1.0 - (age / 3.0)
            x = self._bar_x_center(bin_idx)
            y = self._bar_top_y(bar_heights[bin_idx]) - 6
            self._draw_triangle(x, y, alpha=alpha)
            if self.show_labels:
                self._draw_label(x, y - 8, f"{freq:.1f}", alpha=alpha)
            surviving.append((bin_idx, birth, freq))
        self.active_markers = surviving
```

### What this preserves

- The color mapping (cyan/yellow/orange/red) is untouched.
- The adaptive noise floor and peak decay behavior is untouched.
- HPS detection is always enabled (no `--no-hps` flag currently). The
  overlay is purely additive and does not alter existing display logic.

### What this adds

- A purely *additive* visual layer: small triangle markers that appear for
  ~3 s above bars where HPS has confirmed a harmonic fundamental.
- An out-of-band detection signal that's independent of the color
  threshold — HPS can fire even when the fundamental bar is only yellow,
  because the 2nd and 3rd harmonics carried the detection.

## Gotchas specific to this setup

- **Pi 4 USB audio drops**: HPS is sensitive to frame gaps — a missed chunk
  realigns phase/harmonic structure. Log any xruns from ALSA and surface
  them on the display as a red status dot. `arecord -v` or
  `snd_pcm_status_get_overrange` equivalent.
- **Multiplicative noise blow-up**: HPS multiplies (sums in log domain),
  so one noisy downsampled spectrum dominates. The median-based
  thresholding in the search range handles this without explicit whitening.
- **50/60 Hz mains harmonics**: 50 Hz, 100 Hz, 150 Hz all land in the
  3×downsampled spectrum and can fake a fundamental at 16.67 Hz or 50/3 Hz.
  Narrow notch before FFT is cheap insurance if the site has power lines.
- **FM-induced bin walk**: Rumbles modulate 1–3 bins over 1–3 seconds. The
  3-frame temporal median handles this, but don't use a narrower peak
  picker than ±1 bin.
- **Too many harmonics (R > 4)**: Real overtones aren't perfectly
  integer-spaced. Beyond R=4 you start multiplying by noise and hurting
  SNR. 3 is the sweet spot.
- **Display range extended to 50 Hz**: The display now covers 10–50 Hz,
  so a human reviewer can confirm harmonics visually (e.g. 18 Hz
  fundamental + 36 Hz 2nd harmonic are both visible).

## Implementation steps (suggested order)

1. **Add `hps.py`** with `compute_hps(mag, R)` — pure, testable, no state.
2. **Unit tests**: feed synthetic harmonic stacks at known fundamentals,
   confirm HPS peak lands on the correct bin. Test edge cases: weak
   fundamental + strong 2f/3f (the EM272 rolloff scenario), single
   non-harmonic tone (should NOT produce a peak), pure noise.
3. **Add `HPSDetector` class**: owns temporal smoothing, persistence
   logic, and armed/disarmed gating. State is a small ring buffer of
   recent HPS frames.
4. **Wire into main loop**: call detector after existing FFT/noise-floor
   step. Emit events to a queue.
5. **Display overlay (Option 1)**: add the `active_markers` list to the
   existing bar display class and wire `on_detection(event)` to append
   to it. Render markers in `draw_frame` after the bars are drawn so
   they sit on top. Small downward triangle, white with outline, fades
   over 3 s. No other display changes.
6. **Field validation**: record 30 minutes of ambient noise on site + any
   known rumble clips; replay through the detector offline and tune
   threshold / persistence before going live.
7. **Optional: log detections to file** with timestamp, fundamental, and a
   short audio clip (e.g., 10 s centered on the event) for later review.

## Test corpus ideas

- Synthetic: 18 Hz fundamental + 36 Hz + 54 Hz, varying SNR, varying FM
  sweep rate, varying duration.
- Negative: pure 50 Hz hum, pure 60 Hz hum, pink noise, recorded wind.
- Real: any Cornell Elephant Listening Project / Macaulay Library clips you
  can grab that include documented rumbles. Decimate + filter to simulate
  EM272 rolloff and replay through the detector.

## Open questions to decide before coding

- Do we want HPS to replace the existing threshold-based event logic, or
  run alongside it as a second classifier? (Recommend: alongside — they
  catch slightly different things.)
- Should detection events be persisted to disk, or just displayed? (Recommend:
  both — a short log file with timestamps is cheap and makes field tuning
  much easier.)
- Do we keep display at 30 Hz, or extend to 50 Hz as part of this work?
  (Recommend: extend, since HPS will be finding rumbles whose harmonics
  you currently can't see.)
- Marker label default: on or off? (Recommend: off by default, toggleable
  with a CLI flag or keypress — keeps the display clean when detections
  are frequent.)

## References

- Schroeder, M. R. (1968). *Period Histogram and Product Spectrum: New
  Methods for Fundamental-Frequency Measurement*. JASA. — original HPS
  paper.
- Cornell Elephant Listening Project — rumble spectral characteristics.
- STEInfrasound repo: `infrasound.py` display logic (existing bar /
  noise-floor implementation we're hooking into).
