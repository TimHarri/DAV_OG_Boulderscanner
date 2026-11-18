"""YOLO-Feintuning fuer die Grifferkennung (CPU/GPU).

Zwei Modi, passend zur Annotation der Trainingsdaten:
  --task segment (Standard): Polygon-Labels (z. B. eigener Export aus dem
      Web-UI via export_training_data.py) -> yolo11s-seg Grundmodell
  --task detect: Bounding-Box-Labels (Roboflow-Datensatz "Climbing Holds
      and Volumes" ist box-annotiert) -> yolo11s Grundmodell

Beide Ergebnisdateien werden von app.py/pipeline.py automatisch erkannt:
  holds_yolo11s-seg.pt (Segmentierung) bzw. holds_yolo11s.pt (Detektion)

Beispiele:
  python train/train_yolo.py --data dataset_merged/data.yaml --epochs 80
  python train/train_yolo.py --data dataset_roboflow/data.yaml --task detect --epochs 80
"""
import argparse
import shutil
from pathlib import Path

OUT_NAMES = {"segment": "holds_yolo11s-seg.pt", "detect": "holds_yolo11s.pt"}
DEFAULT_MODELS = {"segment": "yolo11s-seg.pt", "detect": "yolo11s.pt"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Pfad zu data.yaml")
    p.add_argument("--task", choices=OUT_NAMES, default="segment",
                   help="segment = Polygon-Labels, detect = Box-Labels (Roboflow-Datensatz)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--model", default=None, help="default: passend zu --task")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model or DEFAULT_MODELS[args.task])
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        degrees=8, shear=4, hsv_h=0.02, mosaic=0.4,  # Hallen-typische Augmentation
        patience=20,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(OUT_NAMES[args.task])
    shutil.copy2(best, target)
    print(f"Fertig: {target} ({best})")
    print("App neu starten - die Erkennung nutzt die neuen Gewichte automatisch.")


if __name__ == "__main__":
    main()
