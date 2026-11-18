"""P0-Spike: Foto -> Entzerrung -> Grifferkennung -> JSON + annotiertes Bild.

Detector: Baseline (OpenCV-Konturen) laeuft immer out-of-the-box.
Wenn ein trainiertes YOLO11-seg-Modell vorliegt (siehe README),
wird es automatisch bevorzugt (Ultralytics installiert sein muss).
"""
import argparse
import colorsys
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

OUT_SIZE = (1200, 1600)  # entzerrte Wand in Pixeln (B x H)

HUE_NAMES = [(0, "rot"), (12, "orange"), (28, "gelb"), (65, "gruen"),
             (110, "cyan"), (135, "blau"), (160, "pink"), (172, "rot")]

ROLE_COLORS = {"start": (255, 200, 0), "hand": (0, 200, 0),
               "foot": (0, 150, 255), "top": (0, 0, 255), None: (255, 255, 255)}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sortiert 4 Eckpunkte zu tl, tr, br, bl."""
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(angles)
    rot = pts[idx]
    sums = rot.sum(axis=1)
    if rot[0].sum() > rot[2].sum():  # Startpunkt soll oben links liegen
        rot = np.roll(rot, 2, axis=0)
    return rot


def rectify(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    src = order_corners(corners.astype(np.float32))
    dst = np.array([[0, 0], [OUT_SIZE[0] - 1, 0],
                    [OUT_SIZE[0] - 1, OUT_SIZE[1] - 1], [0, OUT_SIZE[1] - 1]],
                   dtype=np.float32)
    h, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img, h, OUT_SIZE)


def color_name(bgr: tuple) -> str:
    h, _, _ = colorsys.rgb_to_hsv(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
    hue = h * 180
    for limit, name in HUE_NAMES:
        if hue < limit:
            return name
    return "rot"


def detect_baseline(rect: np.ndarray) -> list[dict]:
    """Griffe heben sich farblich/hell vom Wand-Hintergrund ab."""
    blur = cv2.GaussianBlur(rect, (7, 7), 0)
    bg = np.median(blur.reshape(-1, 3), axis=0)
    diff = np.linalg.norm(blur.astype(np.float32) - bg, axis=2)
    diff_u8 = np.clip(diff * 2, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = rect.shape[0] * rect.shape[1]
    holds = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not img_area * 0.0004 < area < img_area * 0.02:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        mean_bgr = rect[max(cy - 5, 0):cy + 5, max(cx - 5, 0):cx + 5].reshape(-1, 3).mean(axis=0)
        poly = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True).reshape(-1, 2)
        holds.append({
            "id": len(holds),
            "polygon": poly.tolist(),
            "center": [cx, cy],
            "area_px": int(area),
            "confidence": round(float(diff[cy, cx] / 128.0), 2),
            "color": color_name(tuple(mean_bgr)),
            "role": None,
        })
    return holds


def detect_yolo(rect: np.ndarray, weights: Path) -> list[dict] | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    if not weights.exists():
        return None
    result = YOLO(str(weights))(rect, verbose=False)[0]
    holds = []
    for i, (box, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.conf)):
        x1, y1, x2, y2 = map(int, box)
        holds.append({
            "id": i, "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "area_px": (x2 - x1) * (y2 - y1),
            "confidence": round(float(conf), 2),
            "color": "unbekannt", "role": None,
        })
    return holds


def annotate(rect: np.ndarray, holds: list[dict]) -> np.ndarray:
    out = rect.copy()
    for h in holds:
        pts = np.array(h["polygon"], np.int32)
        color = ROLE_COLORS[h["role"]]
        cv2.polylines(out, [pts], True, color, 2)
        cv2.circle(out, tuple(h["center"]), 4, color, -1)
        cv2.putText(out, str(h["id"]), (h["center"][0] + 8, h["center"][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Boulder P0 Spike Pipeline")
    p.add_argument("image", help="Wandfoto ( frontal moeglichst )")
    p.add_argument("--corners", nargs=4, metavar="X,Y",
                   help="4 Wand-Ecken tl tr br bl; fehlt: gesamtes Bild")
    p.add_argument("--out", default="out")
    p.add_argument("--yolo-weights", default="holds_yolo11s-seg.pt")
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Bild nicht gefunden: {args.image}")

    if args.corners:
        corners = np.array([list(map(float, c.split(","))) for c in args.corners])
    else:
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)

    rect = rectify(img, corners)

    holds = detect_yolo(rect, Path(args.yolo_weights))
    detector = "yolo11s-seg" if holds is not None else "baseline-opencv"
    if holds is None:
        holds = detect_baseline(rect)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "wall_rect.jpg"), rect)
    cv2.imwrite(str(out_dir / "wall_annotated.jpg"), annotate(rect, holds))
    (out_dir / "holds.json").write_text(
        json.dumps({"image": "wall_rect.jpg", "detector": detector,
                    "wall_px": OUT_SIZE, "holds": holds}, indent=1))
    shutil.copy2(Path(__file__).parent / "viewer.html", out_dir / "viewer.html")

    print(f"Detector: {detector} | {len(holds)} Griffe erkannt")
    print(f"Ergebnis: {out_dir}/  ->  Anzeige: cd {out_dir} && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
