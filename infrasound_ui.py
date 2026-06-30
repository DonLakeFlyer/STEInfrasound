"""Shared display chrome for infrasound apps.

Provides the matplotlib backend setup, fullscreen handling, splash/connection
screen, error screen, and the power/exit menu dialog. These are reused by the
bar display (``infrasound.py``) and the detection display
(``infrasound_detections.py``) so the on-screen experience stays consistent.

All signal-processing concerns live in ``infrasound_core.py``; this module only
deals with windows and widgets.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.rcParams['toolbar'] = 'none'
# Set backend explicitly for headless/service operation. Must run before pyplot
# is imported anywhere, so apps should import this module before matplotlib.pyplot.
if os.environ.get('DISPLAY'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        try:
            matplotlib.use('Qt5Agg')
        except Exception:
            # log() isn't imported yet at module-import time; use stderr so this
            # fallback path can never crash the import.
            print("Could not select TkAgg or Qt5Agg backend; using matplotlib default",
                  file=sys.stderr)
import matplotlib.pyplot as plt

from infrasound_core import IS_RPI, log


# -----------------------------
# SCREEN SIZE (3.5" 320x480 display)
# -----------------------------
SCREEN_DPI = 100
SCREEN_WIDTH = 4.8   # inches (480 pixels / 100 dpi)
SCREEN_HEIGHT = 3.2  # inches (320 pixels / 100 dpi)


def maximize_window():
    """Maximize / fullscreen the current figure window (platform specific)."""
    try:
        mng = plt.get_current_fig_manager()
        if IS_RPI:
            try:
                mng.window.attributes('-fullscreen', True)
            except Exception:
                try:
                    mng.full_screen_toggle()
                except Exception:
                    pass
        else:
            try:
                mng.window.state('zoomed')  # Windows
            except Exception:
                try:
                    mng.full_screen_toggle()  # Some backends
                except Exception:
                    pass
    except Exception as e:
        log(f"Error maximizing window: {e}")


# -----------------------------
# SPLASH SCREEN + MIC CONNECTION
# -----------------------------
def show_splash_and_connect(engine):
    """Show a full-screen splash screen and attempt the mic connection.

    Drives ``engine.try_connect_mic()`` over a 10-second countdown, updating the
    splash text each second. Returns True if the engine connected.
    """
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

        maximize_window()

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
        log(f"Loop iteration: {i} seconds remaining, audio_connected={engine.audio_connected}")

        if engine.audio_connected:
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
        log(f"Attempting audio connection (attempt {11-i}/10)...")
        if engine.try_connect_mic():
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
        else:
            log(f"Audio connection attempt {11-i} failed: {engine.audio_error}")
            # Wait before next attempt (only if not on last iteration)
            if i > 1:
                try:
                    time.sleep(0.9)
                except Exception as sleep_err:
                    log(f"Sleep failed: {sleep_err}")

    log(f"Exiting splash screen loop. audio_connected={engine.audio_connected}")

    # Close splash screen
    if fig_splash is not None:
        try:
            plt.close(fig_splash)
            log("Splash screen closed")
        except Exception as e:
            log(f"Error closing splash: {e}")

    log(f"Returning connection status: {engine.audio_connected}")
    return engine.audio_connected


# -----------------------------
# ERROR SCREEN
# -----------------------------
def show_error_screen(error_message):
    """Show a full-screen error window with power/exit controls."""
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
                # Close figures and let the calling app's main flow exit; avoid
                # sys.exit() inside a widget callback (see power-menu Exit).
                plt.close('all')

            btn_exit.on_clicked(exit_program)
            btn_exit.label.set_fontsize(9)

        maximize_window()

        plt.tight_layout()
        log("Displaying error screen...")
        plt.show()

    except Exception as e:
        log(f"ERROR creating error screen: {e}")
        traceback.print_exc()


# -----------------------------
# POWER / EXIT MENU
# -----------------------------
# Module-level state for the singleton menu dialog.
_menu_dialog = None
_menu_buttons = []


def _show_menu_dialog():
    """Show menu dialog with power options."""
    global _menu_dialog, _menu_buttons
    from matplotlib.widgets import Button

    # Treat a figure the user closed via the window manager (X button) as not
    # open, otherwise a stale non-None handle would block reopening the menu.
    if _menu_dialog is not None and plt.fignum_exists(_menu_dialog.number):
        return  # Already showing
    _menu_dialog = None

    log("Showing menu dialog...")
    _menu_dialog = plt.figure(figsize=(SCREEN_WIDTH, SCREEN_HEIGHT), dpi=SCREEN_DPI)
    _menu_dialog.patch.set_facecolor('#1a1a1a')

    def _on_menu_close(event):
        # Reset state whenever the menu figure goes away (Cancel, a power
        # action, or the window manager close button) so it can reopen later.
        global _menu_dialog, _menu_buttons
        _menu_dialog = None
        _menu_buttons = []

    _menu_dialog.canvas.mpl_connect('close_event', _on_menu_close)

    _menu_buttons = []

    if IS_RPI:
        # Three buttons filling the screen: Reboot, Shutdown, Cancel
        ax_reboot = _menu_dialog.add_axes([0.05, 0.68, 0.9, 0.28])
        btn_reboot = Button(ax_reboot, 'Reboot', color='#ff6b6b', hovercolor='#ff5252')

        def reboot(event):
            log("User selected Reboot")
            plt.close('all')
            subprocess.run(['sudo', 'reboot'], check=False)

        btn_reboot.on_clicked(reboot)
        btn_reboot.label.set_fontsize(16)
        _menu_buttons.append(btn_reboot)

        ax_shutdown = _menu_dialog.add_axes([0.05, 0.36, 0.9, 0.28])
        btn_shutdown = Button(ax_shutdown, 'Shutdown', color='#6b6bff', hovercolor='#5252ff')

        def shutdown(event):
            log("User selected Shutdown")
            plt.close('all')
            subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)

        btn_shutdown.on_clicked(shutdown)
        btn_shutdown.label.set_fontsize(16)
        _menu_buttons.append(btn_shutdown)

        ax_cancel = _menu_dialog.add_axes([0.05, 0.04, 0.9, 0.28])
        btn_cancel = Button(ax_cancel, 'Cancel', color='#4a4a4a', hovercolor='#5a5a5a')
    else:
        # Two buttons filling the screen: Exit, Cancel
        ax_exit = _menu_dialog.add_axes([0.05, 0.52, 0.9, 0.44])
        btn_exit = Button(ax_exit, 'Exit', color='#ff6b6b', hovercolor='#ff5252')

        def exit_program(event):
            log("User selected Exit")
            # Close all figures so plt.show() returns and the app's main loop
            # unwinds normally (running engine.stop()); avoid sys.exit() here,
            # which would skip that cleanup and behaves inconsistently across
            # GUI backends when raised from a widget callback.
            plt.close('all')

        btn_exit.on_clicked(exit_program)
        btn_exit.label.set_fontsize(18)
        _menu_buttons.append(btn_exit)

        ax_cancel = _menu_dialog.add_axes([0.05, 0.04, 0.9, 0.44])
        btn_cancel = Button(ax_cancel, 'Cancel', color='#4a4a4a', hovercolor='#5a5a5a')

    def cancel(event):
        global _menu_dialog, _menu_buttons
        log("User cancelled menu")
        if _menu_dialog is not None:
            plt.close(_menu_dialog)
            _menu_dialog = None
            _menu_buttons = []

    btn_cancel.on_clicked(cancel)
    btn_cancel.label.set_fontsize(16 if IS_RPI else 18)
    _menu_buttons.append(btn_cancel)

    maximize_window()

    plt.tight_layout()
    plt.show(block=False)
    plt.draw()


def attach_power_menu(fig, ax):
    """Open the power/exit menu when the user clicks inside the chart(s).

    *ax* may be a single Axes or an iterable of Axes; a click in any of them
    opens the menu.
    """
    try:
        axes = list(ax)
    except TypeError:
        axes = [ax]

    def on_click(event):
        if event.inaxes in axes:
            _show_menu_dialog()

    fig.canvas.mpl_connect('button_press_event', on_click)


if __name__ == '__main__':
    print(
        "infrasound_ui.py is a shared library module, not a runnable app.\n"
        "Run one of the display apps instead:\n"
        "  python infrasound.py [PATH|--file PATH|--test]              # bar display\n"
        "  python infrasound_detections.py [PATH|--file PATH|--test]   # detection display\n"
        "A bare PATH analyzes a WAV file; no path uses the live microphone.",
        file=sys.stderr,
    )
    sys.exit(2)

