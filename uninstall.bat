@echo off
set PYTHON_PATH="C:\Users\suraj\AppData\Local\Python\bin\python.exe"
if not exist %PYTHON_PATH% (
    set PYTHON_PATH=python
)

echo ===================================================
echo Uninstalling dependencies for Iris Flower Classification...
echo Using Python: %PYTHON_PATH%
echo ===================================================

%PYTHON_PATH% -m pip uninstall -y -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo Uninstallation encountered issues or dependencies were not found.
) else (
    echo.
    echo Dependencies successfully uninstalled.
)
pause
