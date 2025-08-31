@echo off
echo ============================================
echo Medical Image Processing Tool Setup
echo ============================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Testing installation...
python test_installation.py

echo.
echo Setup complete!
echo.
echo To run the demo:
echo   python demo.py
echo.
echo To process your own dataset:
echo   python main.py --dataset ./datasets/your_dataset --samples 5
echo.
pause
