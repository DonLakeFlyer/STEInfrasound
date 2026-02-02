#!/bin/bash
set -e

# Wait for 10 seconds to ensure all services are up and usb devices are ready
echo "Sleeping for 10 seconds to allow services to start..."
sleep 10

# Activate your Python environment
source /home/pi/repos/STEInfrasound/venv/bin/activate

echo "Starting python app"
cd /home/pi/repos/STEInfrasound
source venv/bin/activate
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 python infrasound.py
