from app import render_detection_details, render_image_summary
from app import EMPTY_DETECTIONS_HTML, EMPTY_SUMMARY_HTML, get_preview_image_path, run_demo


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


def test_run_demo_returns_placeholder_when_no_image_selected():
    assert run_demo("") == (
        "请选择一张样例图。",
        EMPTY_SUMMARY_HTML,
        None,
        EMPTY_DETECTIONS_HTML,
    )


def test_run_demo_runs_full_success_flow(monkeypatch):
    plotted_image = object()
    detections = [{"class_name": "plane"}]
    summary = {
        "image_name": "demo.jpg",
        "width": 100,
        "height": 80,
        "center_lon": 118.1,
        "center_lat": 24.5,
        "bounds": {},
    }
    enriched_detections = [
        {
            "class_id": 0,
            "class_name": "plane",
            "confidence": 0.95,
            "pixel_center": [10.0, 12.0],
            "geo_center": [118.2, 24.6],
        }
    ]
    calls = []

    def fake_get_image_path(selected_image):
        calls.append(("get_image_path", selected_image))
        return "/tmp/demo.jpg"

    def fake_run_inference(image_path):
        calls.append(("run_inference", image_path))
        return plotted_image, detections

    def fake_build_image_summary(selected_image):
        calls.append(("build_image_summary", selected_image))
        return summary

    def fake_attach_geo_centers(selected_image, raw_detections):
        calls.append(("attach_geo_centers", selected_image, raw_detections))
        return enriched_detections

    monkeypatch.setattr("app.get_image_path", fake_get_image_path)
    monkeypatch.setattr("app.run_inference", fake_run_inference)
    monkeypatch.setattr("app.build_image_summary", fake_build_image_summary)
    monkeypatch.setattr("app.attach_geo_centers", fake_attach_geo_centers)

    status, summary_html, result_image, details_html = run_demo("demo.jpg")

    assert status == "检测完成。"
    assert summary_html == render_image_summary(summary)
    assert result_image is plotted_image
    assert details_html == render_detection_details(enriched_detections)
    assert calls == [
        ("get_image_path", "demo.jpg"),
        ("run_inference", "/tmp/demo.jpg"),
        ("build_image_summary", "demo.jpg"),
        ("attach_geo_centers", "demo.jpg", detections),
    ]


def test_run_demo_returns_placeholder_on_known_errors(monkeypatch):
    monkeypatch.setattr("app.get_image_path", lambda _: (_ for _ in ()).throw(FileNotFoundError("missing file")))

    assert run_demo("demo.jpg") == (
        "图片文件不存在或无法读取。",
        EMPTY_SUMMARY_HTML,
        None,
        EMPTY_DETECTIONS_HTML,
    )

    monkeypatch.setattr("app.get_image_path", lambda _: "/tmp/demo.jpg")
    monkeypatch.setattr("app.run_inference", lambda _: (_ for _ in ()).throw(KeyError("missing geo")))

    status, summary_html, result_image, details_html = run_demo("demo.jpg")
    assert status == "地理信息缺失，无法完成映射。"
    assert summary_html == EMPTY_SUMMARY_HTML
    assert result_image is None
    assert details_html == EMPTY_DETECTIONS_HTML


def test_get_preview_image_path_returns_selected_image_path(monkeypatch):
    monkeypatch.setattr("app.get_image_path", lambda filename: f"/tmp/{filename}")
    assert get_preview_image_path("demo.jpg") == "/tmp/demo.jpg"
    assert get_preview_image_path("") is None