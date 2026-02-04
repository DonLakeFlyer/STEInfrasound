#!/bin/bash
set -e

# Activate your Python environment
source /home/pi/repos/STEInfrasound/venv/bin/activate

echo "Starting python app"
cd /home/pi/repos/STEInfrasound
source venv/bin/activate
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 python infrasound.py
