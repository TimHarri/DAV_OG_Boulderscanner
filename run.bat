@echo off
rem Boulderscanner starten (Windows): Doppelklick oder "run.bat" im Projektordner
cd /d "%~dp0"
if not exist .venv (
    echo Erstelle virtuelle Umgebung ...
    python -m venv .venv || py -m venv .venv || (echo Python nicht gefunden - von python.org installieren! & pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
if not exist test_wall.jpg .venv\Scripts\python.exe make_test_wall.py
echo.
echo  App:  http://localhost:8000/   (API-Docs: http://localhost:8000/docs)
echo  Beenden mit Strg+C
echo.
.venv\Scripts\python.exe -m uvicorn app:app --reload
pause
