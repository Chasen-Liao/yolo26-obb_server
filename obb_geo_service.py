from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError

from demo.geo_mapper import attach_geo_centers, build_image_summary
from demo.inference import extract_obb_detections, patch_torchvision_fake_registration


@dataclass
class OBBGeoDetectionConfig:
    model_path: str = os.getenv("MODEL_PATH", "yolo26n_obb_fair1m.pt")
    img_size: int = int(os.getenv("IMG_SIZE", "1024"))
    obj_thresh: float = float(os.getenv("OBJ_THRESH", "0.25"))
    nms_thresh: float = float(os.getenv("NMS_THRESH", "0.45"))
    request_timeout_sec: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(100 * 1024 * 1024)))
    max_pixels: int = int(os.getenv("MAX_PIXELS", str(10000 * 10000)))


class OBBGeoDetectionError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class OBBGeoService:
    def __init__(self, config: OBBGeoDetectionConfig):
        self.config = config
        self._lock = threading.Lock()
        self._loaded = False
        self._model = None

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise OBBGeoDetectionError(
                    "MODEL_NOT_FOUND",
                    f"Model file not found: {model_path}",
                    500,
                    {"model_path": str(model_path)},
                )
            patch_torchvision_fake_registration()
            from ultralytics import YOLO

            self._model = YOLO(str(model_path))
            self._loaded = True

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self._loaded,
            "model_path": self.config.model_path,
            "img_size": self.config.img_size,
            "obj_thresh": self.config.obj_thresh,
            "nms_thresh": self.config.nms_thresh,
        }

    def close(self) -> None:
        self._model = None
        self._loaded = False

    def detect(
        self,
        image_bytes: bytes,
        filename: str,
        return_image: bool = True,
        return_boxes: bool = True,
        obj_thresh: float | None = None,
        nms_thresh: float | None = None,
    ) -> dict[str, Any]:
        if not filename:
            raise OBBGeoDetectionError("INVALID_FILENAME", "Original filename is required.", 400)
        if len(image_bytes) > self.config.max_image_bytes:
            raise OBBGeoDetectionError("IMAGE_TOO_LARGE", "Image bytes exceed max_image_bytes.", 400)

        self.load()

        started_at = time.perf_counter()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise OBBGeoDetectionError("INVALID_IMAGE", f"Failed to decode image: {exc}", 400) from exc

        if image.width * image.height > self.config.max_pixels:
            raise OBBGeoDetectionError("IMAGE_TOO_LARGE", "Image pixels exceed max_pixels.", 400)

        try:
            summary = build_image_summary(filename)
        except KeyError as exc:
            raise OBBGeoDetectionError(
                "GEO_RECORD_NOT_FOUND",
                f"Geo record not found for filename: {filename}",
                404,
                {"filename": filename},
            ) from exc

        conf = self.config.obj_thresh if obj_thresh is None else obj_thresh
        iou = self.config.nms_thresh if nms_thresh is None else nms_thresh
        results = self._model.predict(source=image, imgsz=self.config.img_size, conf=conf, iou=iou, verbose=False)
        result = results[0]
        detections = extract_obb_detections(result, self._model.names)
        detections = attach_geo_centers(filename, detections) if return_boxes else []

        plotted_bytes = None
        if return_image:
            plotted_image = Image.fromarray(result.plot()[:, :, ::-1])
            buffer = io.BytesIO()
            plotted_image.save(buffer, format="JPEG")
            plotted_bytes = buffer.getvalue()

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 3)
        return {
            "image": {
                "file_name": filename,
                "width": summary["width"],
                "height": summary["height"],
                "center_geo": [summary["center_lon"], summary["center_lat"]],
            },
            "detections": detections,
            "perf": {"total_ms": elapsed_ms},
            "image_jpeg": plotted_bytes,
        }
