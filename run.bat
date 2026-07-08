@echo off
set PYTHON_PATH="C:\Users\suraj\AppData\Local\Python\bin\python.exe"
if not exist %PYTHON_PATH% (
    set PYTHON_PATH=python
)

echo ===================================================
echo Starting Iris Flower Classification App...
echo Using Python: %PYTHON_PATH%
echo ===================================================

%PYTHON_PATH% -m streamlit run app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Failed to start the app. Make sure you ran install.bat first.
)
pause
