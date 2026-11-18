# Boulder P0-Spike

Foto einer Boulderwand -> automatische Grifferkennung -> klickbare Wand im Browser ->
eigenen Boulder setzen und als JSON exportieren.

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

## Lizenz-Hinweis

Ultralytics (YOLO) steht unter AGPL-3.0. Fuer ein closed-source-Produkt
spaeter entweder Enterprise-Lizenz kaufen oder auf einen Apache-2.0-Detektor
(z. B. RT-DETR) ausweichen. Die OpenCV-Baseline ist unproblematisch.
