@echo off
rem memorandum launcher for Windows.
rem Creates the venv and installs dependencies on first run.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv venv || goto :error
    echo [setup] Installing dependencies...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

"venv\Scripts\python.exe" memorandum.py %*
echo.
echo [memorandum] Finished. Press any key to close this window.
pause >nul
goto :eof

:error
echo [setup] Setup failed. Make sure Python 3.9+ is installed and on PATH.
pause
exit /b 1
