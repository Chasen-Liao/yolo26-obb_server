from __future__ import annotations

import numpy as np
import pytest

from obb_geo_service import OBBGeoDetectionConfig, OBBGeoDetectionError, OBBGeoService


class _FakeResult:
    def plot(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeModel:
    def __init__(self):
        self.names = {0: "ship"}
        self.calls = []

    def predict(self, source, imgsz, conf, iou, verbose):
        self.calls.append(
            {
                "size": source.size,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "verbose": verbose,
            }
        )
        return [_FakeResult()]


def test_load_raises_when_model_path_missing(tmp_path):
    service = OBBGeoService(OBBGeoDetectionConfig(model_path=str(tmp_path / "missing.pt")))
    with pytest.raises(OBBGeoDetectionError, match="Model file not found"):
        service.load()


def test_detect_returns_geo_enriched_payload(monkeypatch, sample_png_bytes, sample_upload_name):
    service = OBBGeoService(OBBGeoDetectionConfig(img_size=960, obj_thresh=0.25, nms_thresh=0.45))
    service._loaded = True
    service._model = _FakeModel()

    monkeypatch.setattr(
        "obb_geo_service.build_image_summary",
        lambda filename: {
            "image_name": filename,
            "width": 1000,
            "height": 1000,
            "center_lon": 118.121057,
            "center_lat": 24.536394,
            "bounds": {},
            "affine": [1.0, 0.0, 118.0, 0.0, -1.0, 24.0],
        },
    )
    monkeypatch.setattr(
        "obb_geo_service.extract_obb_detections",
        lambda result, class_names: [
            {
                "index": 0,
                "class_id": 0,
                "class_name": "ship",
                "confidence": 0.92,
                "polygon": [[1.0, 2.0], [5.0, 2.0], [5.0, 6.0], [1.0, 6.0]],
                "pixel_center": [3.0, 4.0],
            }
        ],
    )
    monkeypatch.setattr(
        "obb_geo_service.attach_geo_centers",
        lambda filename, detections: [
            {
                **detections[0],
                "geo_center": [118.123456, 24.534321],
            }
        ],
    )

    result = service.detect(
        sample_png_bytes,
        filename=sample_upload_name,
        return_image=True,
        return_boxes=True,
        obj_thresh=0.3,
        nms_thresh=0.4,
    )

    assert result["image"] == {
        "file_name": sample_upload_name,
        "width": 1000,
        "height": 1000,
        "center_geo": [118.121057, 24.536394],
    }
    assert result["detections"][0]["pixel_center"] == [3.0, 4.0]
    assert result["detections"][0]["geo_center"] == [118.123456, 24.534321]
    assert result["image_jpeg"][:2] == b"\xff\xd8"
    assert result["perf"]["total_ms"] >= 0
    assert service._model.calls == [
        {"size": (16, 16), "imgsz": 960, "conf": 0.3, "iou": 0.4, "verbose": False}
    ]


def test_detect_rejects_invalid_image_bytes(sample_upload_name):
    service = OBBGeoService(OBBGeoDetectionConfig())
    service._loaded = True
    service._model = _FakeModel()

    with pytest.raises(OBBGeoDetectionError, match="Failed to decode image"):
        service.detect(b"not-an-image", filename=sample_upload_name)


def test_detect_rejects_unknown_geo_filename(monkeypatch, sample_png_bytes):
    service = OBBGeoService(OBBGeoDetectionConfig())
    service._loaded = True
    service._model = _FakeModel()
    monkeypatch.setattr("obb_geo_service.build_image_summary", lambda filename: (_ for _ in ()).throw(KeyError(filename)))

    with pytest.raises(OBBGeoDetectionError) as exc_info:
        service.detect(sample_png_bytes, filename="missing.jpg")

    assert exc_info.value.code == "GEO_RECORD_NOT_FOUND"
    assert exc_info.value.status_code == 404
