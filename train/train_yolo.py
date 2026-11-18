"""YOLO11-seg Fine-Tuning fuer die Grifferkennung (optional, CPU/GPU).

1. Datensatz laden (Roboflow 'Climbing Holds and Volumes', CC BY 4.0,
   YOLO-Segmentierungsformat, Klassen: hold, volume) und entpacken:
       unzip climbing-holds-and-volumes.zip -d dataset
2. dataset/data.yaml pruefen (train/val/test-Pfade).
3. Training starten:
       python train/train_yolo.py --data dataset/data.yaml --epochs 80
4. Ergebnis liegt als holds_yolo11s-seg.pt im Projektordner;
   app.py und pipeline.py nutzen es dann automatisch.
"""
import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Pfad zu data.yaml")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--model", default="yolo11s-seg.pt")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        degrees=8, shear=4, hsv_h=0.02, mosaic=0.4,  # Hallen-typische Augmentation
        patience=20,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    target = Path("holds_yolo11s-seg.pt")
    shutil.copy2(best, target)
    print(f"Fertig: {target} ({best})")


if __name__ == "__main__":
    main()
