@echo off
REM Installation script for Fixed Grid Image Remapping

echo ==========================================
echo  Fixed Grid Image Remapping - Installer
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.13+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Found Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please install pip first
    pause
    exit /b 1
)

echo Installing required packages...
echo This may take a few minutes...
echo.

REM Install requirements
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  Installation Complete!
echo ==========================================
echo.
echo You can now run the application using:
echo   launch_gui.bat
echo.
echo Or from command line:
echo   python gui_app.py
echo.
pause
