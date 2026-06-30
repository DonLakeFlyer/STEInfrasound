"""Shared pytest setup for the infrasound test suite.

Forces the non-interactive Agg matplotlib backend before any test imports
pyplot, so the bar-view tests can build figures without a display.
"""

import matplotlib

matplotlib.use("Agg")
