"""API-Tests: alles ueber den FastAPI TestClient, ohne laufenden Server.

Basis ist die synthetische Testwand (make_test_wall.build_wall), damit
der Test reproduzierbar und ohne echte Kletterhalle laeuft. Ausfuehrung:

    pytest -q
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from make_test_wall import build_wall
from pipeline import OUT_SIZE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(scope="module")
def wall(client):
    img = cv2.imread(build_wall(".tmp-wall-src.jpg"))
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    r = client.post("/api/walls", files={"image": ("test.jpg", enc.tobytes(), "image/jpeg")},
                    data={"name": "CI-Wand", "corners": ""})
    assert r.status_code == 200, r.text
    return r.json()


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Boulderscanner" in r.text


def test_wall_created_with_holds(client, wall):
    assert wall["name"] == "CI-Wand"
    assert wall["detector"] == "baseline-opencv" or wall["detector"].startswith("yolo/")
    assert len(wall["holds"]) > 5  # Baseline: ~27; mit liegenden YOLO-Gewichten beliebig


def test_hold_centers_in_bounds(client, wall):
    for h in wall["holds"][:5]:
        x, y = h["center"]
        assert 0 <= x < OUT_SIZE[0] and 0 <= y < OUT_SIZE[1]
        assert len(h["polygon"]) >= 3


def test_wall_image_is_served(client, wall):
    r = client.get(f"/data/{wall['image']}")
    assert r.status_code == 200
    assert len(r.content) > 10_000


def test_assign_and_clear_role(client, wall):
    hid = wall["holds"][0]["id"]
    r = client.patch(f"/api/walls/{wall['id']}/holds/{hid}", json={"role": "start"})
    assert r.json()["role"] == "start"
    r = client.patch(f"/api/walls/{wall['id']}/holds/{hid}", json={"role": None})
    assert r.json()["role"] is None


def test_invalid_role_rejected(client, wall):
    hid = wall["holds"][0]["id"]
    r = client.patch(f"/api/walls/{wall['id']}/holds/{hid}", json={"role": "kaputt"})
    assert r.status_code == 400


def test_manual_hold_append(client, wall):
    r = client.patch(f"/api/walls/{wall['id']}/holds/9999",
                     json={"polygon": [[10, 10], [40, 10], [40, 40], [10, 40]],
                           "center": [25, 25], "role": "hand"})
    assert r.status_code == 200
    body = client.get(f"/api/walls/{wall['id']}").json()
    manual = [h for h in body["holds"] if h["id"] == 9999]
    assert len(manual) == 1 and manual[0]["role"] == "hand"


def test_problem_roundtrip(client, wall):
    hid = wall["holds"][0]["id"]
    client.patch(f"/api/walls/{wall['id']}/holds/{hid}", json={"role": "start"})
    r = client.post(f"/api/walls/{wall['id']}/problems",
                    json={"name": "Erster", "moves": [{"order": 1, "hold_id": hid,
                                                       "role": "start"}]})
    assert r.status_code == 200
    problems = client.get(f"/api/problems?wall_id={wall['id']}").json()
    assert any(p["name"] == "Erster" for p in problems)


def test_unknown_wall_404(client):
    assert client.get("/api/walls/nope").status_code == 404
    r = client.post("/api/walls/nope/problems", json={"name": "x", "moves": []})
    assert r.status_code == 404


def test_homography_upload(client):
    img = cv2.imread(build_wall(".tmp-skew-src.jpg"))
    h, w = img.shape[:2]
    dst = np.float32([[60, 120], [1050, 40], [1150, 1500], [0, 1380]])
    skewed = cv2.warpPerspective(img, cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst), (w, h))
    ok, enc = cv2.imencode(".jpg", skewed)
    assert ok
    r = client.post("/api/walls",
                    files={"image": ("skew.jpg", enc.tobytes(), "image/jpeg")},
                    data={"corners": "60,120 1050,40 1150,1500 0,1380"})
    assert r.status_code == 200
    assert len(r.json()["holds"]) > 15
