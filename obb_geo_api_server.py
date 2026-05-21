from __future__ import annotations

import base64
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from contextlib import asynccontextmanager

from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）；已设置的环境变量不会被覆盖
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from obb_geo_service import OBBGeoDetectionConfig, OBBGeoDetectionError, OBBGeoService


APP_VERSION = "1.0.0"
START_TS = time.time()
API_KEY = os.getenv("API_KEY", "")
DEFAULT_RETURN_IMAGE_MODE = os.getenv("RETURN_IMAGE_MODE", "none")
ASYNC_WORKERS = int(os.getenv("ASYNC_WORKERS", "1"))

config = OBBGeoDetectionConfig()
service = OBBGeoService(config)
executor = ThreadPoolExecutor(max_workers=max(1, ASYNC_WORKERS))
job_lock = Lock()
jobs: dict[str, dict[str, Any]] = {}
metrics_lock = Lock()
metrics = {
    "requests_total": 0,
    "requests_success_total": 0,
    "requests_failed_total": 0,
    "detect_sync_total": 0,
    "detect_async_total": 0,
    "detect_timeout_total": 0,
}


@asynccontextmanager
async def _lifespan(application: FastAPI):
    yield
    service.close()
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="YOLO OBB Geo API", version=APP_VERSION, lifespan=_lifespan)


def _inc(metric_key: str) -> None:
    with metrics_lock:
        metrics[metric_key] = metrics.get(metric_key, 0) + 1


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "Invalid API key."})


def _build_error(err: OBBGeoDetectionError, request_id: str) -> dict[str, Any]:
    return {
        "code": err.code,
        "message": err.message,
        "request_id": request_id,
        "details": err.details,
    }


def _config_dict() -> dict[str, Any]:
    return {
        "model_path": config.model_path,
        "img_size": config.img_size,
        "obj_thresh": config.obj_thresh,
        "nms_thresh": config.nms_thresh,
        "request_timeout_sec": config.request_timeout_sec,
        "max_image_bytes": config.max_image_bytes,
        "max_pixels": config.max_pixels,
    }


def _save_job(job_id: str, data: dict[str, Any]) -> None:
    with job_lock:
        jobs[job_id] = data


def _get_job(job_id: str) -> dict[str, Any] | None:
    with job_lock:
        return jobs.get(job_id)


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    """剥离 image_jpeg 等 bytes 字段，确保 JSON 可序列化。"""
    return {k: v for k, v in result.items() if k != "image_jpeg"}


@app.get("/v1/health")
def health(_: None = Depends(_check_api_key)) -> dict[str, Any]:
    return {"status": "ok", "version": APP_VERSION, "uptime_sec": int(time.time() - START_TS)}


@app.get("/v1/model/status")
def model_status(_: None = Depends(_check_api_key)) -> dict[str, Any]:
    return service.status()


@app.get("/v1/config")
def get_config(_: None = Depends(_check_api_key)) -> dict[str, Any]:
    return _config_dict()


@app.patch("/v1/config")
def patch_config(payload: dict[str, Any], _: None = Depends(_check_api_key)) -> dict[str, Any]:
    for key in ("obj_thresh", "nms_thresh", "img_size", "request_timeout_sec", "max_image_bytes", "max_pixels"):
        if key in payload and payload[key] is not None:
            setattr(config, key, payload[key])
    return {"updated": True, "config": _config_dict()}


@app.post("/v1/detect")
async def detect(
    image: UploadFile = File(...),
    return_image: bool = Form(True),
    return_boxes: bool = Form(True),
    image_mode: str = Form(DEFAULT_RETURN_IMAGE_MODE),
    obj_thresh: float | None = Form(default=None),
    nms_thresh: float | None = Form(default=None),
    _: None = Depends(_check_api_key),
):
    _inc("requests_total")
    _inc("detect_sync_total")
    request_id = str(uuid.uuid4())
    try:
        if image_mode not in {"binary", "base64", "none"}:
            raise OBBGeoDetectionError("INVALID_IMAGE_MODE", "image_mode must be one of binary/base64/none.", 400)
        if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
            raise OBBGeoDetectionError("UNSUPPORTED_MEDIA_TYPE", "Only jpg/png are supported.", 415)
        if obj_thresh is not None and not (0 <= obj_thresh <= 1):
            raise OBBGeoDetectionError("INVALID_OBJ_THRESH", "obj_thresh must be between 0 and 1.", 400)
        if nms_thresh is not None and not (0 <= nms_thresh <= 1):
            raise OBBGeoDetectionError("INVALID_NMS_THRESH", "nms_thresh must be between 0 and 1.", 400)

        image_bytes = await image.read()
        fut: Future = executor.submit(
            service.detect,
            image_bytes,
            image.filename or "",
            return_image,
            return_boxes,
            obj_thresh,
            nms_thresh,
        )
        result = fut.result(timeout=config.request_timeout_sec)
        _inc("requests_success_total")

        if return_image and image_mode == "binary":
            headers = {
                "X-Request-ID": request_id,
                "X-Detections-Count": str(len(result.get("detections", []))),
                "X-Perf-Total-Ms": str(result["perf"]["total_ms"]),
            }
            return Response(content=result.get("image_jpeg") or b"", media_type="image/jpeg", headers=headers)

        payload = {
            "request_id": request_id,
            "image": result["image"],
            "detections": result.get("detections", []),
            "perf": result.get("perf", {}),
            "image_result": {"mode": "none", "value": None},
        }
        if return_image and image_mode == "base64":
            payload["image_result"] = {
                "mode": "base64",
                "value": base64.b64encode(result.get("image_jpeg") or b"").decode("ascii"),
            }
        return JSONResponse(payload)
    except TimeoutError:
        _inc("requests_failed_total")
        _inc("detect_timeout_total")
        raise HTTPException(
            status_code=504,
            detail={
                "code": "REQUEST_TIMEOUT",
                "message": "Detection request timed out.",
                "request_id": request_id,
                "details": {"timeout_sec": config.request_timeout_sec},
            },
        )
    except OBBGeoDetectionError as err:
        _inc("requests_failed_total")
        raise HTTPException(status_code=err.status_code, detail=_build_error(err, request_id))


@app.post("/v1/detect/jobs")
async def create_detect_job(
    image: UploadFile = File(...),
    return_image: bool = Form(True),
    return_boxes: bool = Form(True),
    obj_thresh: float | None = Form(default=None),
    nms_thresh: float | None = Form(default=None),
    _: None = Depends(_check_api_key),
) -> dict[str, Any]:
    _inc("requests_total")
    _inc("detect_async_total")
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only jpg/png are supported."})

    job_id = str(uuid.uuid4())
    image_bytes = await image.read()
    fut = executor.submit(
        service.detect,
        image_bytes,
        image.filename or "",
        return_image,
        return_boxes,
        obj_thresh,
        nms_thresh,
    )
    _save_job(job_id, {"status": "queued", "future": fut, "created_at": int(time.time()), "counted": False})
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/detect/jobs/{job_id}")
def get_detect_job(job_id: str, _: None = Depends(_check_api_key)) -> dict[str, Any]:
    item = _get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})

    fut: Future = item["future"]
    if fut.running():
        item["status"] = "running"
        return {"job_id": job_id, "status": "running"}
    if not fut.done():
        return {"job_id": job_id, "status": "queued"}

    try:
        result = fut.result()
        item["status"] = "succeeded"
        item["result"] = result
        # 返回 JSON 安全的 result（剥离 image_jpeg）
        return {"job_id": job_id, "status": "succeeded", "result": _json_safe_result(result)}
    except OBBGeoDetectionError as err:
        item["status"] = "failed"
        item["error"] = {"code": err.code, "message": err.message, "details": err.details}
        return {"job_id": job_id, "status": "failed", "error": item["error"]}


@app.get("/v1/detect/jobs/{job_id}/image")
def get_detect_job_image(job_id: str, _: None = Depends(_check_api_key)) -> Response:
    item = _get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})
    if "result" not in item:
        raise HTTPException(status_code=409, detail={"code": "JOB_NOT_READY", "message": "Job result not ready."})
    image_bytes = item["result"].get("image_jpeg")
    if image_bytes is None:
        raise HTTPException(status_code=404, detail={"code": "IMAGE_NOT_FOUND", "message": "No image in result."})
    return Response(content=image_bytes, media_type="image/jpeg")


@app.delete("/v1/detect/jobs/{job_id}")
def delete_detect_job(job_id: str, _: None = Depends(_check_api_key)) -> dict[str, Any]:
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})
        jobs.pop(job_id, None)
    return {"deleted": True, "job_id": job_id}


@app.get("/v1/metrics")
def get_metrics() -> Response:
    rows = []
    with metrics_lock:
        for key, value in metrics.items():
            rows.append(f"obb_geo_api_{key} {value}")
    rows.append(f"obb_geo_api_uptime_seconds {int(time.time() - START_TS)}")
    return Response(content="\n".join(rows) + "\n", media_type="text/plain; version=0.0.4")
