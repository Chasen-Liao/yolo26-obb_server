import pytest

from geo_mapper import attach_geo_centers, build_image_summary, pixel_to_geo


def test_pixel_to_geo_matches_known_vertex():
    affine = [
        7.995924900143336e-06,
        -6.474383933152695e-08,
        118.1170915691336,
        -5.944766583354522e-08,
        -7.314240748161893e-06,
        24.540081261873343,
    ]
    lon, lat = pixel_to_geo(affine, 580.0, 74.0)
    assert lon == pytest.approx(118.12172441453158)
    assert lat == pytest.approx(24.539505528411794)


def test_build_image_summary_returns_expected_fields():
    summary = build_image_summary("train__t_10144.jpg")
    assert summary["image_name"] == "train__t_10144.jpg"
    assert summary["width"] == 1000
    assert summary["height"] == 1000
    assert summary["center_lon"] == pytest.approx(118.1210570438674)
    assert summary["center_lat"] == pytest.approx(24.536394476542437)


def test_attach_geo_centers_adds_geo_center():
    detections = [
        {
            "index": 0,
            "class_id": 0,
            "class_name": "plane",
            "confidence": 0.91,
            "polygon": [[580.0, 74.0], [580.0, 74.0], [580.0, 74.0], [580.0, 74.0]],
            "pixel_center": [580.0, 74.0],
        }
    ]
    enriched = attach_geo_centers("train__t_10144.jpg", detections)
    assert enriched[0]["geo_center"][0] == pytest.approx(118.12172441453158)
    assert enriched[0]["geo_center"][1] == pytest.approx(24.539505528411794)