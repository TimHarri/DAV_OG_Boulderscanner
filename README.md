# Boulderscanner

Foto einer Boulderwand -> automatische Grifferkennung -> klickbare Wand im Browser ->
eigene Boulder setzen, speichern und teilen.

## Schritt 2: MVP starten (Web-App)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python make_test_wall.py                       # optional: Testwand erzeugen
uvicorn app:app --reload                       # Server starten
# Browser: http://localhost:8000/  (API-Docs: /docs)
```

Im Web-UI:
1. Wand hochladen (Foto waehlen, Name, optional 4 Eckpunkte)
2. Griffe anklicken -> Rollen zuweisen (Start/Hand/Fuss/Top), wird automatisch gespeichert
3. "Boulder speichern" -> Problem landet in der Liste; Klick laedt es wieder

API: `POST /api/walls` (Upload+Erkennung), `GET/PATCH .../holds`,
`POST /api/walls/{id}/problems`, `GET /api/problems`. Datengrundlage: SQLite
(`data/boulder.db`), spaeter gegen PostgreSQL tauschbar.

## YOLO-Feintuning (echte Erkennungsqualitaet)

```bash
pip install ultralytics
python train/train_yolo.py --data dataset/data.yaml --epochs 80
# -> holds_yolo11s-seg.pt wird automatisch von app.py/pipeline.py genutzt
```

Datensatz: Roboflow "Climbing Holds and Volumes" (CC BY 4.0);
eigene Wandfotos + Nutzerkorrekturen ergaenzen.

## Erkennung verbessern – Stufenplan

Von der Baseline zu verlässlicher Erkennung, sortiert nach Wirkung:

1. **YOLO11-seg feintunen** (größter Sprung): Roboflow-Datensatz „Climbing Holds and Volumes“ (CC BY 4.0) laden, mit `train/train_yolo.py` trainieren → `holds_yolo11s-seg.pt` wird automatisch genutzt.
2. **Eigenes Daten sammeln:** Jede Korrektur im Web-UI wird über `python export_training_data.py --out dataset_own` zu YOLO-Trainingsdaten exportiert. 200–500 eigene Hallenfotos (verschiedene Wände, Licht, Winkel) heben die Qualität auf Hallen-Typik-Niveau.
3. **Auflösung/Details:** `imgsz` hoch (960→1280) oder Tiling für große Wände (SAHI), da Griffe klein im Bild sind.
4. **SAM2-Nacharbeit:** Detektions-Boxes → präzise Masken verfeinern lassen.
5. **Farb-Klassifikation lernen:** Crop-Classifier (timm MobileNetV3) statt HSV-Heuristik, pro Halle feintunen.
6. **Robustheit:** Personen maskieren (Pose-Skeleton), Blenden/Glanz begegnen (CLAHE), Volumen als eigene Klasse.
7. **Messen statt fühlen:** kleines gelabeltes Validierungsset pro Halle; Erkennungsqualität (Precision/Recall) vor jedem Modellwechsel prüfen.

## Testing

- **API-Tests ohne Server:** `pytest -q` (nutzt synthetische Testwand, isoliertes Datenverzeichnis)
- **CI:** GitHub Actions läuft bei jedem Push/PR (`.github/workflows/test.yml`)
- **Lokal schnell starten:** `run.bat` (Windows, Doppelklick) bzw. `./run.sh` (macOS/Linux) – venv, Abhängigkeiten, Testwand und Server in einem Befehl

## P0-Spike (ohne Server, Datei-basiert)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Test ohne echte Wand (sofort lauffaehig)

```bash
python make_test_wall.py                     # synthetische Wand erzeugen
python pipeline.py test_wall.jpg             # Pipeline laufen lassen
cd out && python3 -m http.server 8000        # Browser: http://localhost:8000/viewer.html
```

## Mit eigenem Wandfoto

Foto moeglichst frontal, gute Ausleuchtung. Die 4 Wand-Ecken als `X,Y`-Paare
(oben-links, oben-rechts, unten-rechts, unten-links) in Bildpixeln angeben:

```bash
python pipeline.py mein_foto.jpg --corners 120,80 1080,60 1100,1520 100,1550
cd out && python3 -m http.server 8000
```

Im Viewer Griffe anklicken, um Rollen zu setzen
(Start / Hand / Fuss / Top), dann "Problem exportieren" -> `problem.json`.

## Detector

Standard ist eine OpenCV-Baseline (Kontrast zum Wand-Hintergrund, Konturen).
Sie ist bewusst simpel und dient als Referenz.

Fuer echte Qualitaet: YOLO11-seg auf dem Roboflow-Datensatz
"Climbing Holds and Volumes" (CC BY 4.0) feintunen und die Gewichte als
`holds_yolo11s-seg.pt` neben `pipeline.py` legen. Die Pipeline erkennt die
Datei automatisch und schaltet auf YOLO um (`pip install ultralytics` noetig).

Grober Trainingspfad:
1. Datensatz von https://universe.roboflow.com/ laden (YOLO-Format, Klassen: hold, volume).
2. `yolo segment train data=<dataset>.yaml model=yolo11s-seg.pt epochs=80 imgsz=960`
3. `runs/segment/train/weights/best.pt` -> `holds_yolo11s-seg.pt`

## Struktur

- `app.py` – FastAPI-Backend (SQLite, statische Auslieferung des Web-UI)
- `web/index.html` – Single-Page-Frontend (Wand scannen, Griffe klicken, Boulder speichern)
- `pipeline.py` – Entzerrung + Detektion (Baseline / YOLO)
- `tests/` – API-Tests über den FastAPI TestClient (`pytest -q`)
- `export_training_data.py` – DB-Korrekturen → YOLO-Trainingsdatensatz
- `run.bat` / `run.sh` – Ein-Klick-Start
- `train/train_yolo.py` – YOLO11-seg Fine-Tuning
- `make_test_wall.py` – synthetische Testwand

## Lizenz-Hinweis

Ultralytics (YOLO) steht unter AGPL-3.0. Fuer ein closed-source-Produkt
spaeter entweder Enterprise-Lizenz kaufen oder auf einen Apache-2.0-Detektor
(z. B. RT-DETR) ausweichen. Die OpenCV-Baseline ist unproblematisch.
