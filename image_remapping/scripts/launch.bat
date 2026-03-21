@echo off
chcp 65001 >nul
title Image Remapping Suite Launcher
setlocal enabledelayedexpansion
cls

echo ============================================
echo    Image Remapping Suite - Launcher
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo Python detected:
python --version
echo.

REM Navigate to the parent directory (image_remapping)
cd /d "%~dp0.."

if errorlevel 1 (
    echo [ERROR] Could not navigate to image_remapping directory.
    pause
    exit /b 1
)

echo Working directory: %CD%
echo.

REM Parse command line argument
set "INTERFACE=main"
if "%~1"=="main" set "INTERFACE=main"
if "%~1"=="gdc" set "INTERFACE=gdc"
if "%~1"=="integrated" set "INTERFACE=integrated"

REM Show menu if no argument provided
if "%~1"=="" (
    echo Select Interface:
    echo   [1] Main Lens Distortion Interface (default)
    echo   [2] GDC Grid Processing Interface
    echo   [3] Integrated Interface
    echo.
    set /p choice="Enter choice (1-3): "
    
    if "!choice!"=="1" set "INTERFACE=main"
    if "!choice!"=="2" set "INTERFACE=gdc"
    if "!choice!"=="3" set "INTERFACE=integrated"
    
    REM Handle empty choice - default to main
    if "!choice!"=="" set "INTERFACE=main"
)

echo.
echo ============================================
echo Launching with interface: %INTERFACE%
echo ============================================
echo.

echo Starting Image Remapping Suite...
python main.py --interface %INTERFACE%

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    echo.
    echo Troubleshooting:
    echo   1. Ensure dependencies are installed:
    echo      pip install -r requirements.txt
    echo   2. Check that all required files are present
    echo   3. Verify Python version is 3.8+
    echo.
    pause
)

echo.
echo Application closed.
pause
