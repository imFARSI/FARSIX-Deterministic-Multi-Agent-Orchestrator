@echo off
echo ╔══════════════════════════════════════════════════════════╗
echo ║  FARSIX — Framework for Agentic Reasoning and            ║
echo ║  Supervised Intelligence (X-tended)                      ║
echo ║  Built by Salman Farsi — Undergraduate AI Researcher     ║
echo ║  Brac University                                         ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo [1/3] Installing dependencies with Python 3.11...
py -3.11 -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python 3.11 is installed.
    pause
    exit /b 1
)
echo [2/3] Dependencies installed.
echo [3/3] Starting FARSIX dashboard...
echo.
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
py -3.11 -m streamlit run app.py
pause
