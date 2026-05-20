from inference import build_detection_records


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
