import numpy as np
import pytest
from PIL import Image

from inference import build_detection_records, extract_obb_detections, run_inference


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _FakeObb:
    def __init__(self, polygons, class_ids, confidences):
        self.xyxyxyxy = _FakeTensor(polygons)
        self.cls = _FakeTensor(class_ids)
        self.conf = _FakeTensor(confidences)

    def __len__(self):
        return len(self.cls.tolist())


class _FakeResult:
    def __init__(self, obb):
        self.obb = obb
        self.plot_called = False

    def plot(self):
        self.plot_called = True
        return np.array(
            [
                [[0, 10, 20], [30, 40, 50]],
                [[60, 70, 80], [90, 100, 110]],
            ],
            dtype=np.uint8,
        )


def test_build_detection_records_calculates_polygon_centers():
    polygons = [
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
        [[0.0, 0.0], [4.0, 0.0], [4.0, 8.0], [0.0, 8.0]],
    ]
    records = build_detection_records(
        polygons=polygons,
        class_ids=[1, 0],
        confidences=[0.91, 0.42],
        class_names={0: "ship", 1: "plane"},
    )
    assert records == [
        {
            "index": 0,
            "class_id": 1,
            "class_name": "plane",
            "confidence": 0.91,
            "polygon": polygons[0],
            "pixel_center": [20.0, 30.0],
        },
        {
            "index": 1,
            "class_id": 0,
            "class_name": "ship",
            "confidence": 0.42,
            "polygon": polygons[1],
            "pixel_center": [2.0, 4.0],
        },
    ]


def test_build_detection_records_returns_empty_list_for_empty_input():
    assert build_detection_records([], [], [], {}) == []


def test_build_detection_records_raises_for_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        build_detection_records(
            polygons=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            class_ids=[1, 2],
            confidences=[0.9],
            class_names={1: "plane"},
        )


def test_extract_obb_detections_builds_records_from_result_obb():
    result = _FakeResult(
        _FakeObb(
            polygons=[[[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]],
            class_ids=[1.0],
            confidences=[0.91],
        )
    )

    detections = extract_obb_detections(result, {1: "plane"})

    assert detections == [
        {
            "index": 0,
            "class_id": 1,
            "class_name": "plane",
            "confidence": 0.91,
            "polygon": [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
            "pixel_center": [20.0, 30.0],
        }
    ]


def test_run_inference_returns_pil_image_and_detection_records(monkeypatch, tmp_path):
    result = _FakeResult(
        _FakeObb(
            polygons=[[[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]]],
            class_ids=[0.0],
            confidences=[0.75],
        )
    )

    class _FakeModel:
        def __init__(self):
            self.names = {0: "ship"}
            self.predict_calls = []

        def predict(self, source, verbose):
            self.predict_calls.append({"source": source, "verbose": verbose})
            return [result]

    fake_model = _FakeModel()
    monkeypatch.setattr("inference.load_model", lambda: fake_model)
    image_path = tmp_path / "demo.png"
    image_path.write_bytes(b"not-an-image")

    plotted_image, detections = run_inference(image_path)

    assert isinstance(plotted_image, Image.Image)
    assert plotted_image.mode == "RGB"
    assert plotted_image.size == (2, 2)
    assert detections == [
        {
            "index": 0,
            "class_id": 0,
            "class_name": "ship",
            "confidence": 0.75,
            "polygon": [[2.0, 4.0], [6.0, 4.0], [6.0, 8.0], [2.0, 8.0]],
            "pixel_center": [4.0, 6.0],
        }
    ]
    assert fake_model.predict_calls == [{"source": str(image_path), "verbose": False}]
    assert result.plot_called is True