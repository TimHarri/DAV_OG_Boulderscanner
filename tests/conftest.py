"""Pytest-Fixture: isoliertes Datenverzeichnis + Testclient (kein Server noetig)."""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DATA = ROOT / ".pytest-data"
if TEST_DATA.exists():
    shutil.rmtree(TEST_DATA)
os.environ["BOULDER_DATA"] = str(TEST_DATA)
os.chdir(ROOT)
