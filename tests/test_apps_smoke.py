"""End-to-end smoke tests: run each display app headless in ``--test`` mode.

These exercise the full wiring (argv parsing, engine construction, plot setup,
animation creation) under the non-interactive Agg backend, where ``plt.show()``
returns immediately so the scripts run to completion and exit cleanly.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parent.parent


def _headless_env():
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env.pop("DISPLAY", None)  # keep infrasound_ui from switching to TkAgg
    return env


def _run_app(script):
    return subprocess.run(
        [sys.executable, script, "--test"],
        cwd=str(REPO_ROOT),
        env=_headless_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("script", ["infrasound.py", "infrasound_detections.py"])
def test_app_runs_headless(script):
    result = _run_app(script)
    assert result.returncode == 0, result.stderr


def test_detections_no_bars_runs_headless():
    result = subprocess.run(
        [sys.executable, "infrasound_detections.py", "--test", "--no-bars"],
        cwd=str(REPO_ROOT),
        env=_headless_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_detections_silent_fast_scan_runs_headless(tmp_path):
    # A silent fast-scan opens no audio output stream, so it runs on a headless
    # CI box with no sound device.
    rate = 44100
    n = rate // 2
    t = np.arange(n) / rate
    data = (np.sin(2 * np.pi * 18 * t) * 20000).astype(np.int16)
    wav = tmp_path / "scan.wav"
    wavfile.write(str(wav), rate, data)

    result = subprocess.run(
        [sys.executable, "infrasound_detections.py", "--file", str(wav), "--speed", "20"],
        cwd=str(REPO_ROOT),
        env=_headless_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
