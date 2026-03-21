@echo off
REM Launch Fixed Grid Image Remapping GUI

echo Starting Fixed Grid Image Remapping GUI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.13+ and try again
    pause
    exit /b 1
)

REM Run the GUI
python gui_app.py %*

REM Pause if there was an error
if errorlevel 1 (
    echo.
    echo GUI exited with an error
    pause
)
