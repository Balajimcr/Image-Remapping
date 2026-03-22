@echo off
REM Install dependencies and run the GUI

call install.bat
if errorlevel 1 exit /b 1

echo.
echo Starting GUI...
launch_gui.bat
