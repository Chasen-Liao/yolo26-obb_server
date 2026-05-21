from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

try:
    from .data_loader import get_model_path
except ImportError:
    from data_loader import get_model_path

if TYPE_CHECKING:
    from ultralytics import YOLO


def patch_torchvision_fake_registration() -> None:
    original_register_fake = torch.library.register_fake

    if getattr(original_register_fake, "_yolo26_safe_patch", False):
        return

    def safe_register_fake(op_name: str):
        decorator = original_register_fake(op_name)

        def wrapped(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if op_name == "torchvision::nms" and "does not exist" in str(exc):
                    return fn
                raise

        return wrapped

    safe_register_fake._yolo26_safe_patch = True  # type: ignore[attr-defined]
    torch.library.register_fake = safe_register_fake


def build_detection_records(
    polygons: list[list[list[float]]],
    class_ids: list[float],
    confidences: list[float],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    if not (len(polygons) == len(class_ids) == len(confidences)):
        raise ValueError("polygons, class_ids, and confidences must have the same length")

    records: list[dict[str, Any]] = []
    for index, (polygon, class_id, confidence) in enumerate(zip(polygons, class_ids, confidences)):
        normalized_polygon = [[float(x), float(y)] for x, y in polygon]
        # pixel_center uses the average center of the OBB polygon vertices.
        center_x = sum(point[0] for point in normalized_polygon) / len(normalized_polygon)
        center_y = sum(point[1] for point in normalized_polygon) / len(normalized_polygon)
        numeric_class_id = int(class_id)
        records.append(
            {
                "index": index,
                "class_id": numeric_class_id,
                "class_name": class_names.get(numeric_class_id, str(numeric_class_id)),
                "confidence": float(confidence),
                "polygon": normalized_polygon,
                "pixel_center": [center_x, center_y],
            }
        )
    return records


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    patch_torchvision_fake_registration()
    from ultralytics import YOLO

    model_path = get_model_path()
    return YOLO(str(model_path))


def extract_obb_detections(result: Any, class_names: dict[int, str]) -> list[dict[str, Any]]:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return []
    polygons = obb.xyxyxyxy.cpu().tolist()
    class_ids = obb.cls.cpu().tolist()
    confidences = obb.conf.cpu().tolist()
    return build_detection_records(polygons, class_ids, confidences, class_names)


def run_inference(image_path: str | Path) -> tuple[Image.Image, list[dict[str, Any]]]:
    model = load_model()
    results = model.predict(source=str(image_path), verbose=False)
    result = results[0]
    plotted = result.plot()
    plotted_image = Image.fromarray(plotted[:, :, ::-1])
    detections = extract_obb_detections(result, model.names)
    return plotted_image, detections