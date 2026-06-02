from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from obb_geo_api_server import app


class _StubService:
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def detect(self, image_bytes, filename, return_image, return_boxes, obj_thresh, nms_thresh, geo_mode="required"):
        self.calls.append(
            {
                "filename": filename,
                "return_image": return_image,
                "return_boxes": return_boxes,
                "obj_thresh": obj_thresh,
                "nms_thresh": nms_thresh,
                "geo_mode": geo_mode,
            }
        )
        return {
            "image": {
                "file_name": filename,
                "width": 1000,
                "height": 1000,
                "center_geo": [118.121057, 24.536394],
            },
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "ship",
                    "confidence": 0.91,
                    "pixel_center": [20.0, 30.0],
                    "geo_center": [118.12, 24.53],
                }
            ],
            "perf": {"total_ms": 12.5},
            "image_jpeg": b"\xff\xd8\xff\xd9",
        }


def test_detect_returns_json_payload_for_none_mode(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "none", "return_image": "true", "return_boxes": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image"]["file_name"] == sample_upload_name
    assert payload["image"]["center_geo"] == [118.121057, 24.536394]
    assert payload["detections"][0]["geo_center"] == [118.12, 24.53]
    assert payload["image_result"]["mode"] == "none"
    assert stub.calls == [
        {
            "filename": sample_upload_name,
            "return_image": True,
            "return_boxes": True,
            "obj_thresh": None,
            "nms_thresh": None,
            "geo_mode": "required",
        }
    ]


def test_detect_passes_geo_mode(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"geo_mode": "none"},
    )

    assert response.status_code == 200
    assert stub.calls[0]["geo_mode"] == "none"


def test_detect_rejects_unknown_geo_mode(sample_png_bytes, sample_upload_name):
    client = TestClient(app)
    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"geo_mode": "bad-mode"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_GEO_MODE"


def test_detect_returns_base64_image(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "base64", "return_image": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_result"]["mode"] == "base64"
    assert base64.b64decode(payload["image_result"]["value"]) == b"\xff\xd8\xff\xd9"


def test_detect_returns_binary_image_for_binary_mode(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "binary", "return_image": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-detections-count"] == "1"
    assert response.content == b"\xff\xd8\xff\xd9"


def test_detect_rejects_unknown_image_mode(sample_png_bytes, sample_upload_name):
    client = TestClient(app)
    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "bad-mode"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE_MODE"


def test_get_config_returns_current_config():
    client = TestClient(app)
    response = client.get("/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "img_size" in data
    assert "obj_thresh" in data
    assert "nms_thresh" in data


def test_patch_config_updates_thresh():
    client = TestClient(app)
    try:
        response = client.patch("/v1/config", json={"obj_thresh": 0.5, "nms_thresh": 0.6})
        assert response.status_code == 200
        data = response.json()
        assert data["updated"] is True
        assert data["config"]["obj_thresh"] == 0.5
        assert data["config"]["nms_thresh"] == 0.6
    finally:
        # restore even if assertions above fail
        client.patch("/v1/config", json={"obj_thresh": 0.25, "nms_thresh": 0.45})


def test_api_key_rejects_when_set(monkeypatch):
    monkeypatch.setattr("obb_geo_api_server.API_KEY", "test-secret")
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 401

    response = client.get("/v1/health", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200

    monkeypatch.setattr("obb_geo_api_server.API_KEY", "")
