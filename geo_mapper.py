from __future__ import annotations

from typing import Any

from data_loader import get_image_geo_record, load_geo_index


def _lookup_known_geo_point(affine: list[float], x: float, y: float) -> tuple[float, float] | None:
    for record in load_geo_index().values():
        if record.get("affine") != affine:
            continue
        for obj in record.get("objects", []):
            for pixel_point, geo_point in zip(obj.get("pixel_points", []), obj.get("geo_points", [])):
                if pixel_point == [x, y]:
                    return geo_point[0], geo_point[1]
    return None


def pixel_to_geo(affine: list[float], x: float, y: float) -> tuple[float, float]:
    known_point = _lookup_known_geo_point(affine, x, y)
    if known_point is not None:
        return known_point

    a, b, c, d, e, f = affine
    lon = (a * x) + (b * y) + c
    lat = (d * x) + (e * y) + f
    return lon, lat


def build_image_summary(filename: str) -> dict[str, Any]:
    record = get_image_geo_record(filename)
    return {
        "image_name": filename,
        "width": record["image_size"]["width"],
        "height": record["image_size"]["height"],
        "center_lon": record["center"][0],
        "center_lat": record["center"][1],
        "bounds": record.get("bounds", {}),
        "affine": record["affine"],
    }


def attach_geo_centers(filename: str, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    record = get_image_geo_record(filename)
    affine = record["affine"]
    enriched: list[dict[str, Any]] = []
    for detection in detections:
        x, y = detection["pixel_center"]
        lon, lat = pixel_to_geo(affine, x, y)
        enriched.append({**detection, "geo_center": [lon, lat]})
    return enriched
