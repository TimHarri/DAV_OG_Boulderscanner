"""Erzeugt einen synthetischen YOLO-Segmentierungs-Datensatz (Klassen: hold=0, volume=1).

Zweck:
1. Trainings-Pipeline offline smoke-testen (ohne Roboflow-Account).
2. Referenzformat für eigene Trainingsdaten (gleiches Format wie Roboflow-Datensatz).

Aufruf:
    python make_synthetic_dataset.py --n 40 --out dataset_synthetic
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline import OUT_SIZE

COLORS = {
    "rot": (40, 40, 200), "gelb": (30, 200, 220), "gruen": (60, 160, 60),
    "blau": (200, 120, 40), "orange": (30, 120, 240), "pink": (150, 80, 220),
}


def generate(n: int, out: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    images, labels = Path(out) / "images", Path(out) / "labels"
    for d in (images, labels):
        d.mkdir(parents=True, exist_ok=True)
    W, H = OUT_SIZE

    for i in range(n):
        img = np.full((H, W, 3), int(rng.integers(95, 125)), np.uint8)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 8, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
        lines = []

        for _ in range(int(rng.integers(25, 45))):  # Holds (Klasse 0)
            cx, cy = int(rng.integers(60, W - 60)), int(rng.integers(60, H - 60))
            ax, ay = int(rng.integers(20, 55)), int(rng.integers(16, 48))
            angle = int(rng.integers(0, 180))
            color = COLORS[list(COLORS)[rng.integers(0, len(COLORS))]]
            cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, color, -1)
            cv2.ellipse(img, (cx, cy), (max(ax - 8, 4), max(ay - 8, 4)), angle,
                        0, 360, tuple(int(c * 0.7) for c in color), -1)
            pts = cv2.ellipse2Poly((cx, cy), (ax, ay), angle, 0, 360, 12)
            lines.append("0 " + " ".join(f"{min(max(x, 0) / W, 1):.6f} "
                                         f"{min(max(y, 0) / H, 1):.6f}" for x, y in pts))

        for _ in range(int(rng.integers(1, 4))):  # Volumen (Klasse 1)
            x, y = int(rng.integers(60, W - 320)), int(rng.integers(60, H - 320))
            poly = np.array([[x, y], [x + 190, y + 25], [x + 160, y + 170], [x - 15, y + 140]])
            shade = int(rng.integers(70, 100))
            cv2.fillPoly(img, [poly], (shade, shade, shade))
            lines.append("1 " + " ".join(f"{min(max(px, 0) / W, 1):.6f} "
                                         f"{min(max(py, 0) / H, 1):.6f}" for px, py in poly))

        cv2.imwrite(str(images / f"syn{i:04d}.jpg"), img)
        (labels / f"syn{i:04d}.txt").write_text("\n".join(lines) + "\n")

    (Path(out) / "data.yaml").write_text(
        f"path: {Path(out).resolve()}\ntrain: images\nval: images\nnc: 2\n"
        "names:\n  0: hold\n  1: volume\n")
    (Path(out) / "meta.json").write_text(json.dumps(
        {"n_images": n, "classes": {"0": "hold", "1": "volume"}, "source": "synthetic"}))
    print(f"OK: {n} Bilder -> {out}/ (data.yaml vorhanden)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", default="dataset_synthetic")
    args = p.parse_args()
    generate(args.n, args.out)
