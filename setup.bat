@echo off
echo BTC Quantitative Analysis - Windows Setup
echo ========================================
echo.

echo Setting up virtual environment...
python -m venv btc-quant-env

echo.
echo Activating virtual environment...
call btc-quant-env\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the environment in the future:
echo   btc-quant-env\Scripts\activate
echo.
echo To run the analysis:
echo   python enhanced_main.py
echo.
echo To start Jupyter Lab:
echo   jupyter lab
echo.
pause 