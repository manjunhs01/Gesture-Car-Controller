@echo off
REM VisionDrive controller - first run creates a virtual environment and installs dependencies.
REM Shows a console with status messages. Use run_silent.vbs for a fully invisible start.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3 was not found. Install Python 3.10 - 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo Could not create the virtual environment.
        pause
        exit /b 1
    )
)

if not exist ".venv\.deps_installed_v3" (
    echo Installing dependencies - this takes a few minutes the first time...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. See the messages above and the README troubleshooting section.
        pause
        exit /b 1
    )
    echo ok> ".venv\.deps_installed_v3"
)

".venv\Scripts\python.exe" main.py %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%
