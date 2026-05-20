from app import render_detection_details, render_image_summary


def test_render_image_summary_includes_center_coordinates():
    html = render_image_summary(
        {
            "image_name": "train__t_10144.jpg",
            "width": 1000,
            "height": 1000,
            "center_lon": 118.1210570438674,
            "center_lat": 24.536394476542437,
            "bounds": {
                "min_x": 118.11702682529427,
                "min_y": 24.53270759346783,
                "max_x": 118.12508749403375,
                "max_y": 24.540081261873343,
            },
            "affine": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    )
    assert "train__t_10144.jpg" in html
    assert "118.121057" in html
    assert "24.536394" in html


def test_render_detection_details_limits_to_five_items():
    detections = [
        {
            "index": index,
            "class_id": index,
            "class_name": f"class-{index}",
            "confidence": 0.99 - (index * 0.01),
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "pixel_center": [float(index), float(index + 1)],
            "geo_center": [118.0 + index, 24.0 + index],
        }
        for index in range(6)
    ]
    html = render_detection_details(detections)
    assert html.count("<details>") == 5
    assert "class-5" not in html
