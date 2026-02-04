#!/bin/bash
# filepath: /Users/don/repos/STEInfrasound/start_infrasound.sh

# Log file for debugging
LOG_FILE="/home/pi/infrasound_startup.log"

echo "=== Infrasound startup $(date) ===" >> "$LOG_FILE"
echo "USER: $USER" >> "$LOG_FILE"
echo "HOME: $HOME" >> "$LOG_FILE"
echo "DISPLAY: $DISPLAY" >> "$LOG_FILE"
echo "PWD: $PWD" >> "$LOG_FILE"

# Change to working directory
cd /home/pi/repos/STEInfrasound || {
    echo "ERROR: Could not cd to /home/pi/repos/STEInfrasound" >> "$LOG_FILE"
    exit 1
}

# Activate your Python environment
echo "Activating virtual environment..." >> "$LOG_FILE"
source /home/pi/repos/STEInfrasound/venv/bin/activate 2>> "$LOG_FILE"

echo "Starting python app" | tee -a "$LOG_FILE"
echo "Running infrasound.py..." >> "$LOG_FILE"

# Run with timeout to prevent hanging on shutdown
timeout --signal=TERM --kill-after=5 3600 \
    env DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 \
    python infrasound.py >> "$LOG_FILE" 2>&1

echo "=== Script ended $(date) ===" >> "$LOG_FILE"
