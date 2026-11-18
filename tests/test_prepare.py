"""Tests fuer prepare_dataset.merge (reine Dateioperationen, kein Training)."""
from pathlib import Path

from prepare_dataset import merge


def make_source(root: Path, stem: str, files: int, classes=(0,)) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(files):
        (root / "images" / f"{stem}{i}.jpg").write_bytes(b"\xff\xd8fake")
        (root / "labels" / f"{stem}{i}.txt").write_text(
            "\n".join(f"{c} 0.5 0.5 0.6 0.5 0.6 0.6" for c in classes) + "\n")


def test_merge_creates_splits_and_yaml(tmp_path):
    src_a = tmp_path / "a"
    src_b = tmp_path / "b"
    make_source(src_a, "a", 10, classes=(0,))
    make_source(src_b, "b", 10, classes=(0, 1))
    (src_a / "data.yaml").write_text("nc: 2\nnames:\n  0: hold\n  1: volume\n")

    out = tmp_path / "merged"
    stats = merge([str(src_a), str(src_b)], str(out))

    assert stats["total"] == 20
    assert stats["train"] + stats["val"] == 20
    assert stats["val"] >= 1  # val_fraction 0.15 x 20 -> 3
    yml = (out / "data.yaml").read_text()
    assert "nc: 2" in yml and "hold" in yml and "volume" in yml

    # Labels landen zusammen mit den Bildern im selben Split
    for split in ("train", "val"):
        n_img = len(list((out / "images" / split).glob("*")))
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        assert n_img == n_lbl and n_img == stats[split]


def test_merge_without_labels_fails(tmp_path):
    src = tmp_path / "x"
    (src / "images").mkdir(parents=True)
    (src / "images" / "only.jpg").write_bytes(b"\xff\xd8fake")
    try:
        merge([str(src)], str(tmp_path / "out"))
        assert False, "erwartete SystemExit"
    except SystemExit:
        pass
