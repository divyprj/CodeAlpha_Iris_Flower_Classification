@echo off
set PYTHON_PATH="C:\Users\suraj\AppData\Local\Python\bin\python.exe"
if not exist %PYTHON_PATH% (
    set PYTHON_PATH=python
)

echo ===================================================
echo Installing dependencies for Iris Flower Classification...
echo Using Python: %PYTHON_PATH%
echo ===================================================

%PYTHON_PATH% -m pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo Installation failed! Please check if Python is installed and accessible.
) else (
    echo.
    echo Installation completed successfully.
)
pause
