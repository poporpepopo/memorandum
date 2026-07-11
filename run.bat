@echo off
rem memorandum launcher for Windows.
rem Creates the venv, installs dependencies, and starts Ollama on first run.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv venv || goto :error
    echo [setup] Installing dependencies...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

rem Start the Ollama server if it is not already running.
curl -s -o nul --max-time 2 http://localhost:11434/api/version
if errorlevel 1 (
    if not exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" goto :ollama_error
    echo [setup] Starting Ollama...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    call :wait_ollama || goto :ollama_error
)

"venv\Scripts\python.exe" memorandum.py %*
echo.
echo [memorandum] Finished. Press any key to close this window.
pause >nul
goto :eof

:wait_ollama
for /l %%i in (1,1,30) do (
    curl -s -o nul --max-time 2 http://localhost:11434/api/version && exit /b 0
    rem "timeout" fails without an interactive console; use ping as a 1s sleep.
    ping -n 2 127.0.0.1 >nul
)
exit /b 1

:error
echo [setup] Setup failed. Make sure Python 3.9+ is installed and on PATH.
pause
exit /b 1

:ollama_error
echo [setup] Could not start Ollama. Install it from https://ollama.com/
echo         and pull the model: ollama pull gemma4:e4b
pause
exit /b 1
