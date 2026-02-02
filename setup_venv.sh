#!/bin/bash
set -e

VENV_DIR="venv"

echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "✓ Setup complete! Activate with: source $VENV_DIR/bin/activate"
