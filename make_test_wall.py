"""Erzeugt ein synthetisches Testwand-Bild (ohne echte Kletterhalle testbar)."""
import argparse
import cv2
import numpy as np

COLORS = {
    "rot": (40, 40, 200), "gelb": (30, 200, 220), "gruen": (60, 160, 60),
    "blau": (200, 120, 40), "orange": (30, 120, 240), "pink": (150, 80, 220),
    "weiss": (235, 235, 235), "schwarz": (35, 35, 35),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="test_wall.jpg")
    p.add_argument("--holds", type=int, default=35)
    args = p.parse_args()

    rng = np.random.default_rng(7)
    img = np.full((1600, 1200, 3), 110, np.uint8)
    noise = rng.normal(0, 9, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for _ in range(14):  # Holzstruktur-Streifen
        y = rng.integers(0, 1600)
        cv2.line(img, (0, y), (1200, y + rng.integers(-60, 60)), (100, 100, 100), 2)

    names = list(COLORS)
    for _ in range(args.holds):
        cx, cy = int(rng.integers(70, 1130)), int(rng.integers(70, 1530))
        ax, ay = int(rng.integers(22, 55)), int(rng.integers(18, 48))
        angle = int(rng.integers(0, 180))
        color = COLORS[names[rng.integers(0, len(names))]]
        cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, color, -1)
        cv2.ellipse(img, (cx, cy), (max(ax - 8, 5), max(ay - 8, 5)), angle,
                    0, 360, tuple(int(c * 0.7) for c in color), -1)

    for _ in range(3):  # Volumen
        x, y = int(rng.integers(100, 1000)), int(rng.integers(100, 1400))
        pts = np.array([[x, y], [x + 160, y + 20], [x + 140, y + 140], [x - 10, y + 120]])
        cv2.fillPoly(img, [pts], (95, 95, 95))

    cv2.imwrite(args.out, img)
    print(f"OK: {args.out} (Ecken fuer Homographie: 0,0 1200,0 1200,1600 0,1600)")


if __name__ == "__main__":
    main()
