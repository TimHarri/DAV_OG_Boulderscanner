"""Exportiert Wände + Holds aus der MVP-Datenbank als YOLO-Segmentierungs-Datensatz.

Damit wird jede Korrektur im Web-UI (Griffe löschen/zeichnen, Rollen setzen)
zu Trainingsmaterial für das Feintuning (siehe train/train_yolo.py):

    python export_training_data.py --out dataset_own
    # -> dataset_own/images/*.jpg + dataset_own/labels/*.txt (YOLO-Format)
    # Danach mit den Roboflow-Daten zusammenlegen und trainieren.
"""
import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/boulder.db")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="dataset_own")
    args = p.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"DB nicht gefunden: {args.db} (erst Wände im Web-UI anlegen)")

    images, labels = Path(args.out) / "images", Path(args.out) / "labels"
    for d in (images, labels):
        d.mkdir(parents=True, exist_ok=True)

    n_walls, n_holds = 0, 0
    with sqlite3.connect(args.db) as conn:
        for wall_id, rect_path, holds_json in conn.execute(
                "SELECT id, rect_path, holds_json FROM walls"):
            img_path = Path(args.data_dir) / rect_path
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  übersprungen (Bild fehlt): {rect_path}")
                continue
            h, w = img.shape[:2]
            holds = json.loads(holds_json)

            stem = f"wall_{wall_id}"
            shutil.copy2(img_path, images / f"{stem}.jpg")
            lines = []
            for hold in holds:
                poly = hold.get("polygon") or []
                if len(poly) < 3:
                    continue
                pts = []
                for x, y in poly:
                    pts.append(min(max(x, 0) / w, 1.0))
                    pts.append(min(max(y, 0) / h, 1.0))
                lines.append("0 " + " ".join(f"{v:.6f}" for v in pts))
                n_holds += 1
            (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_walls += 1

    print(f"OK: {n_walls} Wände, {n_holds} Holds -> {args.out}/")
    print("Nächster Schritt: dataset_own mit Roboflow-Datensatz vereinen")
    print("und train/train_yolo.py --data <vereinigte data.yaml> starten.")


if __name__ == "__main__":
    main()
