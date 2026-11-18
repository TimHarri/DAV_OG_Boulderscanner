#!/usr/bin/env bash
# Boulderscanner starten (macOS/Linux): ./run.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Erstelle virtuelle Umgebung ..."
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
[ -f test_wall.jpg ] || .venv/bin/python make_test_wall.py
echo
echo " App:  http://localhost:8000/   (API-Docs: http://localhost:8000/docs)"
echo " Beenden mit Strg+C"
echo
exec .venv/bin/python -m uvicorn app:app --reload
