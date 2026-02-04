#!/bin/bash
# filepath: /Users/don/repos/STEInfrasound/start_infrasound.sh
set -e

# Log file for debugging
LOG_FILE="/home/pi/infrasound_startup.log"

echo "=== Infrasound startup $(date) ===" >> "$LOG_FILE"
echo "USER: $USER" >> "$LOG_FILE"
echo "HOME: $HOME" >> "$LOG_FILE"
echo "DISPLAY: $DISPLAY" >> "$LOG_FILE"
echo "PWD: $PWD" >> "$LOG_FILE"

# Activate your Python environment
echo "Activating virtual environment..." >> "$LOG_FILE"
source /home/pi/repos/STEInfrasound/venv/bin/activate 2>> "$LOG_FILE"

echo "Starting python app" | tee -a "$LOG_FILE"
cd /home/pi/repos/STEInfrasound
source venv/bin/activate
echo "Running infrasound.py..." >> "$LOG_FILE"
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 python infrasound.py >> "$LOG_FILE" 2>&1

echo "=== Script ended $(date) ===" >> "$LOG_FILE"
