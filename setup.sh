#!/bin/bash

echo "BTC Quantitative Analysis - Unix Setup"
echo "======================================"
echo

echo "Setting up virtual environment..."
python3 -m venv btc-quant-env

echo
echo "Activating virtual environment..."
source btc-quant-env/bin/activate

echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "Setup complete!"
echo
echo "To activate the environment in the future:"
echo "  source btc-quant-env/bin/activate"
echo
echo "To run the analysis:"
echo "  python enhanced_main.py"
echo
echo "To start Jupyter Lab:"
echo "  jupyter lab"
echo 