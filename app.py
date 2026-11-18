"""MVP (Schritt 2): FastAPI-Backend.

Wand hochladen -> erkennen -> Holds korrigieren -> Boulder speichern/laden.
SQLite als Speicher (Austausch gegen PostgreSQL spaeter, Schema identisch).

Start:  uvicorn app:app --reload
Docs:   http://localhost:8000/docs
UI:     http://localhost:8000/
"""
import json
import os
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import OUT_SIZE, annotate, detect_baseline, detect_yolo, rectify

DATA = Path(os.environ.get("BOULDER_DATA", "data"))
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "boulder.db"

app = FastAPI(title="Boulderscanner MVP")


# ---------- DB ----------

SCHEMA = """
CREATE TABLE IF NOT EXISTS walls(
    id TEXT PRIMARY KEY, name TEXT, rect_path TEXT,
    holds_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS problems(
    id TEXT PRIMARY KEY, wall_id TEXT NOT NULL REFERENCES walls(id),
    name TEXT NOT NULL, moves_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def holds_list(row: sqlite3.Row) -> list[dict]:
    return json.loads(row["holds_json"])


def save_holds(conn: sqlite3.Connection, wall_id: str, holds: list[dict]) -> None:
    conn.execute("UPDATE walls SET holds_json=? WHERE id=?",
                 (json.dumps(holds), wall_id))


# ---------- Schemas ----------

class HoldPatch(BaseModel):
    role: str | None = None  # start|hand|foot|top|null (null = Rolle entfernen)
    color: str | None = None
    manual: bool | None = None
    polygon: list[list[int]] | None = None  # nur bei neuen/manuellen Holds
    center: list[int] | None = None


class ProblemIn(BaseModel):
    name: str
    moves: list[dict]  # [{order, hold_id, role, color}]


def wall_dict(row: sqlite3.Row, wall_id: str) -> dict:
    return {
        "id": wall_id, "name": row["name"],
        "image": row["rect_path"],  # unter /data/ erreichbar
        "holds": holds_list(row),
    }


# ---------- API ----------

@app.post("/api/walls")
async def create_wall(
    image: UploadFile = File(...),
    name: str = Form("Wand"),
    corners: str = Form(""),  # "x1,y1 x2,y2 x3,y3 x4,y4" (tl tr br bl), leer = ganzes Bild
    max_side: int = Form(1600),
) -> dict:
    raw = await image.read()
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "Bild konnte nicht gelesen werden")

    # Downsampling fuer schnelle Pipeline
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        arr = cv2.resize(arr, (int(w * s), int(h * s)))

    if corners.strip():
        try:
            pts = np.array([[float(v) for v in p.split(",")] for p in corners.split()])
            if pts.shape != (4, 2):
                raise ValueError
        except ValueError:
            raise HTTPException(400,
                "corners erwartet: 'x1,y1 x2,y2 x3,y3 x4,y4' (tl tr br bl)")
    else:
        hh, ww = arr.shape[:2]
        pts = np.array([[0, 0], [ww, 0], [ww, hh], [0, hh]], float)

    rect = rectify(arr, pts)

    import pathlib
    weights = pathlib.Path(__file__).parent / "holds_yolo11s-seg.pt"
    holds = detect_yolo(rect, weights)
    detector = "yolo11s-seg"
    if holds is None:
        holds = detect_baseline(rect)
        detector = "baseline-opencv"

    wall_id = uuid.uuid4().hex[:12]
    rect_path = f"wall_{wall_id}.jpg"
    cv2.imwrite(str(DATA / rect_path), annotate(rect, holds))
    (DATA / f"holds_{wall_id}.json").write_text(json.dumps(
        {"detector": detector, "holds": holds}))

    with db() as c:
        c.execute("INSERT INTO walls(id, name, rect_path, holds_json) VALUES (?,?,?,?)",
                  (wall_id, name, rect_path, json.dumps(holds)))
    return {**wall_dict(c.execute("SELECT * FROM walls WHERE id=?",
                                  (wall_id,)).fetchone(), wall_id),
            "detector": detector}


@app.get("/api/walls")
def list_walls() -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT id, name, rect_path FROM walls").fetchall()
    return [{"id": r["id"], "name": r["name"], "image": r["rect_path"]} for r in rows]


@app.get("/api/walls/{wall_id}")
def get_wall(wall_id: str) -> dict:
    with db() as c:
        row = c.execute("SELECT * FROM walls WHERE id=?", (wall_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Wand nicht gefunden")
    return wall_dict(row, wall_id)


@app.patch("/api/walls/{wall_id}/holds/{hold_id}")
def patch_hold(wall_id: str, hold_id: int, patch: HoldPatch) -> dict:
    with db() as c:
        row = c.execute("SELECT * FROM walls WHERE id=?", (wall_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Wand nicht gefunden")
        holds = holds_list(row)
        target = next((h for h in holds if h["id"] == hold_id), None)
        if target is None:  # unbekannte id + polygon => manueller Hold
            if patch.polygon is None:
                raise HTTPException(404, f"Hold {hold_id} nicht gefunden")
            target = {"id": hold_id, "polygon": patch.polygon,
                      "center": patch.center or [0, 0], "area_px": 0,
                      "confidence": 1.0, "color": patch.color or "unbekannt",
                      "role": None}
            holds.append(target)
        # "role" darf explizit null sein (Rolle loeschen) -> model_fields_set nutzen
        if "role" in patch.model_fields_set:
            if patch.role not in (None, "start", "hand", "foot", "top"):
                raise HTTPException(400, "role: start|hand|foot|top|null")
            target["role"] = patch.role
        for k in ("color", "manual"):
            if getattr(patch, k) is not None:
                target[k] = getattr(patch, k)
        if patch.polygon is not None:
            target["polygon"] = patch.polygon
        save_holds(c, wall_id, holds)
    return target


@app.post("/api/walls/{wall_id}/problems")
def create_problem(wall_id: str, body: ProblemIn) -> dict:
    with db() as c:
        if not c.execute("SELECT 1 FROM walls WHERE id=?", (wall_id,)).fetchone():
            raise HTTPException(404, "Wand nicht gefunden")
        pid = uuid.uuid4().hex[:12]
        c.execute("INSERT INTO problems(id, wall_id, name, moves_json) VALUES (?,?,?,?)",
                  (pid, wall_id, body.name, json.dumps(body.moves)))
    return {"id": pid, "wall_id": wall_id, "name": body.name, "moves": body.moves}


@app.get("/api/problems")
def list_problems(wall_id: str | None = None) -> list[dict]:
    q, args = "SELECT id, wall_id, name, moves_json FROM problems", ()
    if wall_id:
        q += " WHERE wall_id=?"
        args = (wall_id,)
    with db() as c:
        rows = c.execute(q, args).fetchall()
    return [{"id": r["id"], "wall_id": r["wall_id"], "name": r["name"],
             "moves": json.loads(r["moves_json"])} for r in rows]


# ---------- Statisch ----------

app.mount("/data", StaticFiles(directory=DATA), name="data")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("web/index.html")


app.mount("/web", StaticFiles(directory="web"), name="web")
