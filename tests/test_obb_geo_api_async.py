from __future__ import annotations

import time

from fastapi.testclient import TestClient

from obb_geo_api_server import app


class _AsyncStubService:
    def detect(self, image_bytes, filename, return_image, return_boxes, obj_thresh, nms_thresh):
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


def test_async_job_lifecycle(monkeypatch, sample_png_bytes, sample_upload_name):
    monkeypatch.setattr("obb_geo_api_server.service", _AsyncStubService())
    client = TestClient(app)

    create_response = client.post(
        "/v1/detect/jobs",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["job_id"]

    # Poll until succeeded
    for _ in range(10):
        status_response = client.get(f"/v1/detect/jobs/{job_id}")
        if status_response.json()["status"] == "succeeded":
            break
        time.sleep(0.05)

    final = status_response.json()
    assert final["status"] == "succeeded"
    # job result 不含 image_jpeg（JSON 安全）
    assert "image_jpeg" not in final["result"]
    assert final["result"]["image"]["center_geo"] == [118.121057, 24.536394]
    assert final["result"]["detections"][0]["geo_center"] == [118.12, 24.53]

    # 图片走独立端点
    image_response = client.get(f"/v1/detect/jobs/{job_id}/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.content == b"\xff\xd8\xff\xd9"

    delete_response = client.delete(f"/v1/detect/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_async_job_not_found():
    client = TestClient(app)
    response = client.get("/v1/detect/jobs/nonexistent-id")
    assert response.status_code == 404


def test_metrics_endpoint_returns_prometheus_text():
    client = TestClient(app)
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert "obb_geo_api_requests_total" in response.text
