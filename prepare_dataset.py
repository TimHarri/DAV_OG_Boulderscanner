"""Vereint mehrere YOLO-Segmentierungs-Datensätze zu einem Trainings-Datensatz.

Typischer Aufruf nach dem Download des Roboflow-Datensatzes ("Climbing Holds
and Volumes", CC BY 4.0, Klassen hold=0/volume=1) und dem Export der eigenen
Korrekturen (export_training_data.py --out dataset_own):

    python prepare_dataset.py dataset_own dataset_synthetic --out dataset_merged
    python train/train_yolo.py --data dataset_merged/data.yaml --epochs 80

Roboflow-Download (optional, kostenloser Account noetig):
    export ROBOFLOW_API_KEY=...        # Windows: set ROBOFLOW_API_KEY=...
    python prepare_dataset.py --from-roboflow blackcreed/climbing-holds-and-volumes --out dataset_roboflow
"""
import argparse
import random
import shutil
from pathlib import Path

DEFAULT_NAMES = {0: "hold", 1: "volume"}
VAL_FRACTION = 0.15
SEED = 42


def merge(sources: list[str], out: str, val_fraction: float = VAL_FRACTION) -> dict:
    """Kopiert images+labels aller Quellen in train/val-Splits von 'out'."""
    images = Path(out) / "images"
    labels = Path(out) / "labels"
    for sub in ("train", "val"):
        (images / sub).mkdir(parents=True, exist_ok=True)
        (labels / sub).mkdir(parents=True, exist_ok=True)

    pairs, names = [], dict(DEFAULT_NAMES)
    for src in sources:
        src = Path(src)
        yaml_path = src / "data.yaml"
        if yaml_path.exists():  # Klassennamen uebernehmen (erste Quelle gewinnt)
            for line in yaml_path.read_text().splitlines():
                line = line.strip().removeprefix("- ").strip()
                if ":" in line and line.split(":")[0].isdigit():
                    idx, name = line.split(":", 1)
                    names.setdefault(int(idx), name.strip().strip("'\""))

        img_dir = src / "images"
        if not img_dir.exists():
            raise SystemExit(f"Kein images/-Ordner in {src}")
        for img in sorted(img_dir.rglob("*.jpg")) + sorted(img_dir.rglob("*.png")):
            lbl = (src / "labels" / f"{img.stem}.txt")
            if not lbl.exists():
                lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))

    if not pairs:
        raise SystemExit("Keine Bild/Label-Paare gefunden")

    random.Random(SEED).shuffle(pairs)
    counts = {"train": 0, "val": 0}
    for i, (img, lbl) in enumerate(pairs):
        split = "val" if i < int(len(pairs) * val_fraction) else "train"
        stem = f"{img.parent.name}_{img.stem}" if (img.parent.name != "images") else img.stem
        shutil.copy2(img, images / split / f"{stem}{img.suffix}")
        shutil.copy2(lbl, labels / split / f"{stem}.txt")
        counts[split] += 1

    nc = max(names) + 1
    names_lines = "\n".join(f"  {k}: {names[k]}" for k in sorted(names))
    (Path(out) / "data.yaml").write_text(
        f"path: {Path(out).resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {nc}\nnames:\n{names_lines}\n")
    return {"total": len(pairs), **counts, "classes": names}


def download_roboflow(slug: str, out: str, version: int | None = None,
                      fmt: str = "yolov11") -> None:
    """Laedt einen Universe-Datensatz im YOLO-Exportformat.

    Achtung: Roboflow kennt keine 'seg'-Formatsuffixe - bei
    Instance-Segmentation-Projekten enthaelt der 'yolov11'-Export
    automatisch Polygon-Labels, bei Box-Projekten Bounding-Boxes.
    Der Trainingsmodus (--task) wird darum erst nach dem Download passend
    zum Label-Inhalt gewaehlt."""
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("pip install roboflow  (und ROBOFLOW_API_KEY setzen, "
                         "kostenloser Account auf roboflow.com)")
    import os
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY ist nicht gesetzt "
                         "(Roboflow-Account -> Settings -> API Key)")
    project = Roboflow(api_key=key).workspace().project(slug.split("/")[-1])
    v = project.version(version) if version else project.versions()[0]["version"]
    v.download(fmt, location=out)
    print(f"OK: {slug} -> {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs="*", help="Datensatz-Ordner (je images/ + labels/)")
    p.add_argument("--out", default="dataset_merged")
    p.add_argument("--val", type=float, default=VAL_FRACTION)
    p.add_argument("--from-roboflow", metavar="WORKSPACE/DATASET",
                   help="Datensatz von Roboflow Universe laden (kostet nur einen "
                        "kostenlosen Account, dazu ROBOFLOW_API_KEY)")
    p.add_argument("--version", type=int, default=None,
                   help="Roboflow-Datensatz-Version (ohne: neueste)")
    p.add_argument("--format", default="yolov11", choices=["yolov11"],
                   help="YOLO-Exportformat (Instance-Segmentation-Projekte "
                        "liefern automatisch Polygon-Labels)")
    args = p.parse_args()

    if args.from_roboflow:
        download_roboflow(args.from_roboflow, args.out, args.version, args.format)
    if not args.sources and args.from_roboflow:
        print("Nur Download - zum Zusammenfuehren weitere Aufrufe mit Quellen starten.")
    else:
        stats = merge(args.sources, args.out, args.val)
        print(f"OK: {stats['total']} Paare "
              f"(train={stats['train']}, val={stats['val']}, "
              f"Klassen: {stats['classes']}) -> {args.out}/data.yaml")
