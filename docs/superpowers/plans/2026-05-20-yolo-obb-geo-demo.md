# YOLO OBB Geo API Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个整体风格对齐 `/data/RK/yolov8_server/` 的 FastAPI 服务，上传 `sample_100_mix/` 中对应图片文件后返回 OBB 检测结果、图片中心经纬度、检测框中心像素点和检测框中心经纬度，并支持同步、异步、配置和指标接口。

**Architecture:** 保留 `demo/` 目录中的现有推理与地理映射能力，在根目录新增 `obb_geo_service.py` 负责模型加载、图片解码、文件名匹配、地理结果组装；新增 `obb_geo_api_server.py` 负责 FastAPI 路由、鉴权、配置、同步检测、异步任务和指标输出。测试分为服务层单测、同步接口测试、异步接口测试和文档/依赖测试。

**Tech Stack:** Python 3、FastAPI、Uvicorn、python-multipart、Ultralytics YOLO、Pillow、Pytest、httpx

---

## File Structure

- `requirements.txt`：补充 API 运行与测试依赖
- `obb_geo_service.py`：服务层，处理上传图片、地理匹配、OBB 推理和结果组装
- `obb_geo_api_server.py`：API 层，提供 `/v1/*` 路由、异步任务和 metrics
- `tests/conftest.py`：测试通用 fixture，如上传文件 bytes 和样例图片名
- `tests/test_api_project_files.py`：依赖与基础测试资源检查
- `tests/test_obb_geo_service.py`：服务层单元测试
- `tests/test_obb_geo_api_sync.py`：同步接口测试
- `tests/test_obb_geo_api_async.py`：异步、metrics 与任务管理测试
- `README.md`：API 服务启动和调用说明

### Task 1: 补齐 API 依赖与测试夹具

**Files:**
- Modify: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_api_project_files.py`

- [ ] **Step 1: 写依赖与测试夹具的失败测试**

`tests/test_api_project_files.py`

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_requirements_include_api_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package_name in ["fastapi", "uvicorn", "python-multipart", "httpx", "pytest"]:
        assert package_name in requirements


def test_test_fixtures_module_exists():
    fixture_file = ROOT / "tests" / "conftest.py"
    assert fixture_file.exists(), "missing tests/conftest.py"
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_api_project_files.py -q`
Expected: FAIL，提示 `requirements.txt` 缺少 `fastapi`、`uvicorn`、`python-multipart` 或缺少 `tests/conftest.py`

- [ ] **Step 3: 写最小实现**

`requirements.txt`

```text
streamlit>=1.45,<2.0
ultralytics>=8.3,<9.0
pillow>=10.0,<11.0
fastapi>=0.115,<1.0
uvicorn>=0.30,<1.0
python-multipart>=0.0.9,<1.0
httpx>=0.27,<1.0
pytest>=8.0,<9.0
```

`tests/conftest.py`

```python
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_upload_name() -> str:
    return "train__t_10144.jpg"


@pytest.fixture
def sample_png_bytes() -> bytes:
    image = Image.new("RGB", (16, 16), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_dataset_image_path(sample_upload_name: str) -> Path:
    return ROOT / "sample_100_mix" / sample_upload_name
```

- [ ] **Step 4: 重新运行测试**

Run: `python -m pytest tests/test_api_project_files.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add requirements.txt tests/conftest.py tests/test_api_project_files.py
git commit -m "chore: add api service dependencies and fixtures"
```

### Task 2: 实现服务层的图片解码、地理匹配与检测结果组装

**Files:**
- Create: `obb_geo_service.py`
- Create: `tests/test_obb_geo_service.py`

- [ ] **Step 1: 写服务层失败测试**

`tests/test_obb_geo_service.py`

```python
from __future__ import annotations

import numpy as np
import pytest

from obb_geo_service import OBBGeoDetectionConfig, OBBGeoDetectionError, OBBGeoService


class _FakeResult:
    def __init__(self):
        self.names = {0: "ship"}

    def plot(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeModel:
    def __init__(self):
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


def test_detect_returns_geo_not_found_when_filename_missing(monkeypatch, sample_png_bytes):
    service = OBBGeoService(OBBGeoDetectionConfig())
    service._loaded = True
    service._model = _FakeModel()
    monkeypatch.setattr("obb_geo_service.build_image_summary", lambda filename: (_ for _ in ()).throw(KeyError(filename)))

    with pytest.raises(OBBGeoDetectionError) as exc_info:
        service.detect(sample_png_bytes, filename="missing.jpg")

    assert exc_info.value.code == "GEO_RECORD_NOT_FOUND"
    assert exc_info.value.status_code == 404
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_obb_geo_service.py -q`
Expected: FAIL，提示缺少 `obb_geo_service.py` 或缺少 `OBBGeoService`

- [ ] **Step 3: 写最小实现**

`obb_geo_service.py`

```python
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, UnidentifiedImageError

from demo.geo_mapper import attach_geo_centers, build_image_summary
from demo.inference import extract_obb_detections, patch_torchvision_fake_registration


@dataclass
class OBBGeoDetectionConfig:
    model_path: str = os.getenv("MODEL_PATH", "yolo26n_obb_fair1m.pt")
    img_size: int = int(os.getenv("IMG_SIZE", "1024"))
    obj_thresh: float = float(os.getenv("OBJ_THRESH", "0.25"))
    nms_thresh: float = float(os.getenv("NMS_THRESH", "0.45"))
    request_timeout_sec: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "8"))
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(100 * 1024 * 1024)))
    max_pixels: int = int(os.getenv("MAX_PIXELS", str(10000 * 10000)))


class OBBGeoDetectionError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
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

    def close(self) -> None:
        with self._lock:
            self._model = None
            self._loaded = False

    def status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._loaded,
            "model_name": self.config.model_path,
            "img_size": self.config.img_size,
            "obj_thresh": self.config.obj_thresh,
            "nms_thresh": self.config.nms_thresh,
            "device": "cpu",
        }

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        if not image_bytes:
            raise OBBGeoDetectionError("INVALID_IMAGE", "Empty image body.", 400)
        if len(image_bytes) > self.config.max_image_bytes:
            raise OBBGeoDetectionError(
                "IMAGE_TOO_LARGE",
                "Image exceeds size limit.",
                400,
                {"max_image_bytes": self.config.max_image_bytes, "actual": len(image_bytes)},
            )
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise OBBGeoDetectionError("INVALID_IMAGE", "Failed to decode image.", 415) from exc
        width, height = image.size
        if width * height > self.config.max_pixels:
            raise OBBGeoDetectionError(
                "IMAGE_RESOLUTION_TOO_LARGE",
                "Image resolution exceeds limit.",
                400,
                {"max_pixels": self.config.max_pixels, "actual_pixels": width * height},
            )
        return image

    def detect(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        return_image: bool = True,
        return_boxes: bool = True,
        obj_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._loaded or self._model is None:
            raise OBBGeoDetectionError("MODEL_NOT_READY", "Model is not loaded.", 503)
        if not filename:
            raise OBBGeoDetectionError("INVALID_FILENAME", "Original filename is required.", 400)

        effective_obj = self.config.obj_thresh if obj_thresh is None else obj_thresh
        effective_nms = self.config.nms_thresh if nms_thresh is None else nms_thresh

        begin_total = time.perf_counter()
        begin_pre = time.perf_counter()
        image = self._decode_image(image_bytes)
        try:
            summary = build_image_summary(filename)
        except KeyError as exc:
            raise OBBGeoDetectionError(
                "GEO_RECORD_NOT_FOUND",
                f"Geo record not found for image: {filename}",
                404,
                {"file_name": filename},
            ) from exc
        preprocess_ms = (time.perf_counter() - begin_pre) * 1000

        with self._lock:
            begin_infer = time.perf_counter()
            results = self._model.predict(
                source=image,
                imgsz=self.config.img_size,
                conf=effective_obj,
                iou=effective_nms,
                verbose=False,
            )
            infer_ms = (time.perf_counter() - begin_infer) * 1000

        begin_post = time.perf_counter()
        result = results[0]
        raw_detections = extract_obb_detections(result, result.names or {}) if return_boxes else []
        try:
            detections = attach_geo_centers(filename, raw_detections) if return_boxes else []
        except KeyError as exc:
            raise OBBGeoDetectionError(
                "GEO_MAPPING_INCOMPLETE",
                f"Geo mapping data is incomplete for image: {filename}",
                500,
                {"file_name": filename},
            ) from exc
        plotted_image = Image.fromarray(result.plot()[:, :, ::-1]) if return_image else None
        image_jpeg = None
        if plotted_image is not None:
            buffer = io.BytesIO()
            plotted_image.convert("RGB").save(buffer, format="JPEG")
            image_jpeg = buffer.getvalue()
        postprocess_ms = (time.perf_counter() - begin_post) * 1000
        total_ms = (time.perf_counter() - begin_total) * 1000

        return {
            "request_id": str(uuid.uuid4()),
            "image": {
                "file_name": filename,
                "width": summary["width"],
                "height": summary["height"],
                "center_geo": [summary["center_lon"], summary["center_lat"]],
            },
            "detections": detections,
            "perf": {
                "preprocess_ms": round(preprocess_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "postprocess_ms": round(postprocess_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "image_jpeg": image_jpeg,
        }
```

- [ ] **Step 4: 运行测试，确认它通过**

Run: `python -m pytest tests/test_obb_geo_service.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_service.py tests/test_obb_geo_service.py
git commit -m "feat: add obb geo detection service"
```

### Task 3: 实现同步 API、配置接口与主检测接口

**Files:**
- Create: `obb_geo_api_server.py`
- Create: `tests/test_obb_geo_api_sync.py`

- [ ] **Step 1: 写同步接口失败测试**

`tests/test_obb_geo_api_sync.py`

```python
from __future__ import annotations

import base64

from fastapi.testclient import TestClient

import obb_geo_api_server as api


class _FakeService:
    def __init__(self):
        self.config = api.config
        self.detect_calls = []

    def load(self):
        return None

    def close(self):
        return None

    def status(self):
        return {
            "model_loaded": True,
            "model_name": "yolo26n_obb_fair1m.pt",
            "img_size": 1024,
            "obj_thresh": 0.25,
            "nms_thresh": 0.45,
            "device": "cpu",
        }

    def detect(self, image_bytes, *, filename, return_image, return_boxes, obj_thresh, nms_thresh):
        self.detect_calls.append(
            {
                "filename": filename,
                "return_image": return_image,
                "return_boxes": return_boxes,
                "obj_thresh": obj_thresh,
                "nms_thresh": nms_thresh,
            }
        )
        return {
            "request_id": "req-1",
            "image": {
                "file_name": filename,
                "width": 1000,
                "height": 1000,
                "center_geo": [118.121057, 24.536394],
            },
            "detections": [
                {
                    "index": 0,
                    "class_id": 1,
                    "class_name": "plane",
                    "confidence": 0.91,
                    "pixel_center": [100.0, 200.0],
                    "geo_center": [118.12, 24.53],
                }
            ],
            "perf": {"total_ms": 12.3},
            "image_jpeg": b"\xff\xd8fake-jpeg",
        }


def test_health_and_model_status(monkeypatch):
    monkeypatch.setattr(api, "service", _FakeService())
    with TestClient(api.app) as client:
        health = client.get("/v1/health")
        status = client.get("/v1/model/status")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert status.status_code == 200
    assert status.json()["model_loaded"] is True


def test_detect_base64_returns_json(monkeypatch, sample_png_bytes, sample_upload_name):
    fake_service = _FakeService()
    monkeypatch.setattr(api, "service", fake_service)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/detect",
            files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
            data={"image_mode": "base64", "return_image": "true", "return_boxes": "true"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "req-1"
    assert payload["image"]["center_geo"] == [118.121057, 24.536394]
    assert payload["detections"][0]["geo_center"] == [118.12, 24.53]
    assert payload["image_result"]["mode"] == "base64"
    assert base64.b64decode(payload["image_result"]["value"]).startswith(b"\xff\xd8")
    assert fake_service.detect_calls[0]["filename"] == sample_upload_name


def test_detect_binary_returns_jpeg_response(monkeypatch, sample_png_bytes, sample_upload_name):
    monkeypatch.setattr(api, "service", _FakeService())
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/detect",
            files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
            data={"image_mode": "binary", "return_image": "true"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-request-id"] == "req-1"
    assert response.content.startswith(b"\xff\xd8")


def test_patch_config_updates_thresholds(monkeypatch):
    monkeypatch.setattr(api, "service", _FakeService())
    with TestClient(api.app) as client:
        response = client.patch("/v1/config", json={"obj_thresh": 0.5, "nms_thresh": 0.3})

    assert response.status_code == 200
    assert response.json()["config"]["obj_thresh"] == 0.5
    assert response.json()["config"]["nms_thresh"] == 0.3
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_obb_geo_api_sync.py -q`
Expected: FAIL，提示缺少 `obb_geo_api_server.py` 或缺少 `/v1/detect`

- [ ] **Step 3: 写最小实现**

`obb_geo_api_server.py`

```python
from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from obb_geo_service import OBBGeoDetectionConfig, OBBGeoDetectionError, OBBGeoService


APP_VERSION = "1.0.0"
START_TS = time.time()
API_KEY = os.getenv("API_KEY", "")
DEFAULT_RETURN_IMAGE_MODE = os.getenv("RETURN_IMAGE_MODE", "binary")

app = FastAPI(title="YOLO OBB Geo API", version=APP_VERSION)
config = OBBGeoDetectionConfig()
service = OBBGeoService(config)


def _config_dict() -> Dict[str, Any]:
    return {
        "model_path": config.model_path,
        "img_size": config.img_size,
        "obj_thresh": config.obj_thresh,
        "nms_thresh": config.nms_thresh,
        "request_timeout_sec": config.request_timeout_sec,
        "max_image_bytes": config.max_image_bytes,
        "max_pixels": config.max_pixels,
    }


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED", "message": "Invalid API key."})


def _build_error(err: OBBGeoDetectionError, request_id: str) -> Dict[str, Any]:
    return {
        "code": err.code,
        "message": err.message,
        "request_id": request_id,
        "details": err.details,
    }


@app.on_event("startup")
def startup_event() -> None:
    service.load()


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.close()


@app.get("/v1/health")
def health(_: None = Depends(_check_api_key)) -> Dict[str, Any]:
    return {"status": "ok", "version": APP_VERSION, "uptime_sec": int(time.time() - START_TS)}


@app.get("/v1/model/status")
def model_status(_: None = Depends(_check_api_key)) -> Dict[str, Any]:
    return service.status()


@app.get("/v1/config")
def get_config(_: None = Depends(_check_api_key)) -> Dict[str, Any]:
    return _config_dict()


@app.patch("/v1/config")
def patch_config(payload: Dict[str, Any], _: None = Depends(_check_api_key)) -> Dict[str, Any]:
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
    obj_thresh: Optional[float] = Form(default=None),
    nms_thresh: Optional[float] = Form(default=None),
    _: None = Depends(_check_api_key),
):
    request_id = f"req-{int(time.time() * 1000)}"
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
        result = service.detect(
            image_bytes,
            filename=image.filename or "",
            return_image=return_image,
            return_boxes=return_boxes,
            obj_thresh=obj_thresh,
            nms_thresh=nms_thresh,
        )
        request_id = result["request_id"]
        image_jpeg = result.pop("image_jpeg", None)

        if return_image and image_mode == "binary":
            headers = {
                "X-Request-ID": request_id,
                "X-Detections-Count": str(len(result.get("detections", []))),
                "X-Perf-Total-Ms": str(result["perf"]["total_ms"]),
            }
            return Response(content=image_jpeg or b"", media_type="image/jpeg", headers=headers)

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
                "value": base64.b64encode(image_jpeg or b"").decode("ascii"),
            }
        return JSONResponse(payload)
    except OBBGeoDetectionError as err:
        raise HTTPException(status_code=err.status_code, detail=_build_error(err, request_id))
```

- [ ] **Step 4: 运行测试，确认它通过**

Run: `python -m pytest tests/test_obb_geo_api_sync.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_api_server.py tests/test_obb_geo_api_sync.py
git commit -m "feat: add sync obb geo api endpoints"
```

### Task 4: 扩展异步任务接口与 metrics

**Files:**
- Modify: `obb_geo_api_server.py`
- Create: `tests/test_obb_geo_api_async.py`

- [ ] **Step 1: 写异步接口失败测试**

`tests/test_obb_geo_api_async.py`

```python
from __future__ import annotations

from concurrent.futures import Future

from fastapi.testclient import TestClient

import obb_geo_api_server as api


class _FakeService:
    def __init__(self):
        self.config = api.config

    def load(self):
        return None

    def close(self):
        return None

    def status(self):
        return {"model_loaded": True}

    def detect(self, image_bytes, *, filename, return_image, return_boxes, obj_thresh, nms_thresh):
        return {
            "request_id": "job-request",
            "image": {
                "file_name": filename,
                "width": 1000,
                "height": 1000,
                "center_geo": [118.121057, 24.536394],
            },
            "detections": [
                {
                    "index": 0,
                    "class_id": 1,
                    "class_name": "plane",
                    "confidence": 0.91,
                    "pixel_center": [100.0, 200.0],
                    "geo_center": [118.12, 24.53],
                }
            ],
            "perf": {"total_ms": 12.3},
            "image_jpeg": b"\xff\xd8fake-jpeg",
        }


def test_async_job_lifecycle(monkeypatch, sample_png_bytes, sample_upload_name):
    monkeypatch.setattr(api, "service", _FakeService())

    future = Future()
    future.set_result(api.service.detect(sample_png_bytes, filename=sample_upload_name, return_image=True, return_boxes=True, obj_thresh=None, nms_thresh=None))
    monkeypatch.setattr(api.executor, "submit", lambda fn, *args, **kwargs: future)
    api.jobs.clear()

    with TestClient(api.app) as client:
        created = client.post(
            "/v1/detect/jobs",
            files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        )
        job_id = created.json()["job_id"]
        queried = client.get(f"/v1/detect/jobs/{job_id}")
        image_resp = client.get(f"/v1/detect/jobs/{job_id}/image")
        deleted = client.delete(f"/v1/detect/jobs/{job_id}")

    assert created.status_code == 200
    assert queried.status_code == 200
    assert queried.json()["status"] == "succeeded"
    assert queried.json()["result"]["image"]["center_geo"] == [118.121057, 24.536394]
    assert image_resp.status_code == 200
    assert image_resp.content.startswith(b"\xff\xd8")
    assert deleted.json() == {"deleted": True, "job_id": job_id}


def test_metrics_endpoint_returns_prometheus_lines(monkeypatch):
    monkeypatch.setattr(api, "service", _FakeService())
    api.metrics["requests_total"] = 3
    with TestClient(api.app) as client:
        response = client.get("/v1/metrics")

    assert response.status_code == 200
    assert "obb_geo_api_requests_total 3" in response.text
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_obb_geo_api_async.py -q`
Expected: FAIL，提示缺少 `/v1/detect/jobs`、`/v1/metrics` 或模块内没有 `jobs` / `metrics`

- [ ] **Step 3: 在现有 API 文件上做最小增量实现**

在 `obb_geo_api_server.py` 顶部补充这些导入和全局变量：

```python
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock


ASYNC_WORKERS = int(os.getenv("ASYNC_WORKERS", "1"))

executor = ThreadPoolExecutor(max_workers=max(1, ASYNC_WORKERS))
job_lock = Lock()
jobs: Dict[str, Dict[str, Any]] = {}

metrics_lock = Lock()
metrics = {
    "requests_total": 0,
    "requests_success_total": 0,
    "requests_failed_total": 0,
    "detect_sync_total": 0,
    "detect_async_total": 0,
}


def _inc(metric_key: str) -> None:
    with metrics_lock:
        metrics[metric_key] = metrics.get(metric_key, 0) + 1


def _save_job(job_id: str, data: Dict[str, Any]) -> None:
    with job_lock:
        jobs[job_id] = data


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with job_lock:
        return jobs.get(job_id)
```

把 `shutdown_event()` 改成：

```python
@app.on_event("shutdown")
def shutdown_event() -> None:
    service.close()
    executor.shutdown(wait=False, cancel_futures=True)
```

把 `detect()` 的开头与成功/失败分支补上 metrics：

```python
    _inc("requests_total")
    _inc("detect_sync_total")
```

在 `image_jpeg = result.pop("image_jpeg", None)` 后补上：

```python
        _inc("requests_success_total")
```

在 `except OBBGeoDetectionError as err:` 前后补上：

```python
    except OBBGeoDetectionError as err:
        _inc("requests_failed_total")
        raise HTTPException(status_code=err.status_code, detail=_build_error(err, request_id))
    except Exception as err:
        _inc("requests_failed_total")
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": str(err), "request_id": request_id, "details": {}},
        )
```

在文件末尾追加这些路由：

```python
@app.post("/v1/detect/jobs")
async def create_detect_job(
    image: UploadFile = File(...),
    return_image: bool = Form(True),
    return_boxes: bool = Form(True),
    obj_thresh: Optional[float] = Form(default=None),
    nms_thresh: Optional[float] = Form(default=None),
    _: None = Depends(_check_api_key),
) -> Dict[str, Any]:
    _inc("requests_total")
    _inc("detect_async_total")
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=415, detail={"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only jpg/png are supported."})
    job_id = str(uuid.uuid4())
    image_bytes = await image.read()
    future = executor.submit(
        service.detect,
        image_bytes,
        filename=image.filename or "",
        return_image=return_image,
        return_boxes=return_boxes,
        obj_thresh=obj_thresh,
        nms_thresh=nms_thresh,
    )
    _save_job(job_id, {"status": "queued", "future": future})
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/detect/jobs/{job_id}")
def get_detect_job(job_id: str, _: None = Depends(_check_api_key)) -> Dict[str, Any]:
    item = _get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})
    future: Future = item["future"]
    if not future.done():
        return {"job_id": job_id, "status": item.get("status", "queued")}
    result = future.result()
    item["status"] = "succeeded"
    item["result"] = result
    return {
        "job_id": job_id,
        "status": "succeeded",
        "result": {
            "request_id": result["request_id"],
            "image": result["image"],
            "detections": result["detections"],
            "perf": result["perf"],
        },
    }


@app.get("/v1/detect/jobs/{job_id}/image")
def get_detect_job_image(job_id: str, _: None = Depends(_check_api_key)) -> Response:
    item = _get_job(job_id)
    if not item:
        raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})
    if "result" not in item:
        raise HTTPException(status_code=409, detail={"code": "JOB_NOT_READY", "message": "Job result not ready."})
    return Response(content=item["result"].get("image_jpeg") or b"", media_type="image/jpeg")


@app.delete("/v1/detect/jobs/{job_id}")
def delete_detect_job(job_id: str, _: None = Depends(_check_api_key)) -> Dict[str, Any]:
    with job_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail={"code": "JOB_NOT_FOUND", "message": "Job not found."})
        jobs.pop(job_id, None)
    return {"deleted": True, "job_id": job_id}


@app.get("/v1/metrics")
def get_metrics(_: None = Depends(_check_api_key)) -> Response:
    rows = []
    with metrics_lock:
        for key, value in metrics.items():
            rows.append(f"obb_geo_api_{key} {value}")
    rows.append(f"obb_geo_api_uptime_seconds {int(time.time() - START_TS)}")
    return Response(content="\n".join(rows) + "\n", media_type="text/plain; version=0.0.4")
```

- [ ] **Step 4: 运行测试，确认它通过**

Run: `python -m pytest tests/test_obb_geo_api_async.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_api_server.py tests/test_obb_geo_api_async.py
git commit -m "feat: add async obb geo api endpoints"
```

### Task 5: 编写 API 使用文档并做完整验证

**Files:**
- Create: `README.md`
- Create: `tests/test_readme_api_usage.py`

- [ ] **Step 1: 写文档失败测试**

`tests/test_readme_api_usage.py`

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_describes_api_usage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "uvicorn obb_geo_api_server:app" in readme
    assert "/v1/detect" in readme
    assert "sample_100_mix" in readme
    assert "image_mode" in readme
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_readme_api_usage.py -q`
Expected: FAIL，提示缺少 `README.md` 或内容不包含 API 用法

- [ ] **Step 3: 写最小实现**

`README.md`

```markdown
# YOLO OBB Geo API Service

基于 `yolo26n_obb_fair1m.pt` 和 `sample_100_mix/geo.json` 的 OBB 地理检测 API 服务。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 启动服务

```bash
uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8001
```

启动后访问 `http://127.0.0.1:8001/docs` 查看 Swagger 文档。

## 检测接口

```bash
curl -X POST "http://127.0.0.1:8001/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "image_mode=base64" \
  -F "return_image=true"
```

## 业务约束

- 上传文件必须来自 `sample_100_mix`
- 服务保留上传文件原始文件名
- 通过原始文件名匹配 `geo.json`
- 成功结果返回图片中心经纬度和检测框中心经纬度

## 图片返回模式

- `image_mode=binary`
- `image_mode=base64`
- `image_mode=none`
```

- [ ] **Step 4: 跑文档测试和完整测试集**

Run: `python -m pytest tests/test_api_project_files.py tests/test_obb_geo_service.py tests/test_obb_geo_api_sync.py tests/test_obb_geo_api_async.py tests/test_readme_api_usage.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add README.md tests/test_readme_api_usage.py
git commit -m "docs: add obb geo api usage guide"
```

## Self-Review Checklist

- Spec coverage: 已覆盖依赖、服务层、同步接口、异步接口、配置、metrics、README 和验证命令
- Placeholder scan: 无 `TBD`、`TODO`、`implement later` 等占位语句
- Type consistency: `OBBGeoDetectionConfig`、`OBBGeoDetectionError`、`OBBGeoService.detect()`、`image_result`、`center_geo`、`geo_center` 等命名在各任务中保持一致


def get_model_path() -> Path:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH}")
    return MODEL_PATH
```

`geo_mapper.py`

```python
from __future__ import annotations

from typing import Any

from data_loader import get_image_geo_record


def pixel_to_geo(affine: list[float], x: float, y: float) -> tuple[float, float]:
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
```

- [ ] **Step 4: 运行测试，确认实现通过**

Run: `python -m pytest tests/test_data_loader.py tests/test_geo_mapper.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add data_loader.py geo_mapper.py tests/test_data_loader.py tests/test_geo_mapper.py
git commit -m "feat: add geo metadata loading utilities"
```

### Task 3: 实现 OBB 推理包装与检测结果整理

**Files:**
- Create: `inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: 写推理结果整理的失败测试**

`tests/test_inference.py`

```python
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
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_inference.py -q`
Expected: FAIL，提示 `ModuleNotFoundError` 或缺少 `build_detection_records`

- [ ] **Step 3: 写最小实现**

`inference.py`

```python
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

from data_loader import get_model_path


def build_detection_records(
    polygons: list[list[list[float]]],
    class_ids: list[float],
    confidences: list[float],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (polygon, class_id, confidence) in enumerate(zip(polygons, class_ids, confidences)):
        normalized_polygon = [[float(x), float(y)] for x, y in polygon]
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
```

- [ ] **Step 4: 运行测试，确认实现通过**

Run: `python -m pytest tests/test_inference.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add inference.py tests/test_inference.py
git commit -m "feat: add obb inference wrapper"
```

### Task 4: 实现 Gradio 页面与结果渲染

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: 写页面渲染辅助函数的失败测试**

`tests/test_app.py`

```python
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
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_app.py -q`
Expected: FAIL，提示 `ModuleNotFoundError` 或缺少待实现函数

- [ ] **Step 3: 写最小实现**

`app.py`

```python
from __future__ import annotations

from html import escape
from typing import Any

import gradio as gr

from data_loader import get_image_path, list_sample_images
from geo_mapper import attach_geo_centers, build_image_summary
from inference import run_inference


EMPTY_SUMMARY_HTML = "<p>暂无图片信息。</p>"
EMPTY_DETECTIONS_HTML = "<p>暂无检测结果。</p>"


def render_image_summary(summary: dict[str, Any]) -> str:
    bounds = summary.get("bounds", {})
    bounds_html = ""
    if bounds:
        bounds_html = (
            "<li><strong>边界范围:</strong> "
            f"({bounds['min_x']:.6f}, {bounds['min_y']:.6f}) / "
            f"({bounds['max_x']:.6f}, {bounds['max_y']:.6f})"
            "</li>"
        )
    return (
        "<div>"
        "<h3>图片地理信息</h3>"
        "<ul>"
        f"<li><strong>图片名:</strong> {escape(summary['image_name'])}</li>"
        f"<li><strong>图片尺寸:</strong> {summary['width']} × {summary['height']}</li>"
        f"<li><strong>图片中心点经纬度:</strong> ({summary['center_lon']:.6f}, {summary['center_lat']:.6f})</li>"
        f"{bounds_html}"
        "</ul>"
        "</div>"
    )


def render_detection_details(detections: list[dict[str, Any]], limit: int = 5) -> str:
    if not detections:
        return "<p>未检测到目标。</p>"
    blocks: list[str] = []
    ranked = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    for rank, detection in enumerate(ranked[:limit], start=1):
        px, py = detection["pixel_center"]
        lon, lat = detection["geo_center"]
        blocks.append(
            "<details>"
            f"<summary>检测框 {rank} - {escape(detection['class_name'])} | 置信度 {detection['confidence']:.3f}</summary>"
            "<ul>"
            f"<li><strong>类别 ID:</strong> {detection['class_id']}</li>"
            f"<li><strong>像素中心点:</strong> ({px:.2f}, {py:.2f})</li>"
            f"<li><strong>经纬度中心点:</strong> ({lon:.6f}, {lat:.6f})</li>"
            "</ul>"
            "</details>"
        )
    return "\n".join(blocks)


def run_demo(selected_image: str):
    if not selected_image:
        return "请选择一张样例图。", EMPTY_SUMMARY_HTML, None, EMPTY_DETECTIONS_HTML
    try:
        image_path = get_image_path(selected_image)
        plotted_image, detections = run_inference(image_path)
        summary = build_image_summary(selected_image)
        enriched_detections = attach_geo_centers(selected_image, detections)
    except (FileNotFoundError, KeyError) as exc:
        return str(exc), EMPTY_SUMMARY_HTML, None, EMPTY_DETECTIONS_HTML

    status_text = "检测完成。" if enriched_detections else "未检测到目标。"
    return (
        status_text,
        render_image_summary(summary),
        plotted_image,
        render_detection_details(enriched_detections),
    )


def build_app() -> gr.Blocks:
    sample_images = list_sample_images()
    default_value = sample_images[0] if sample_images else None

    with gr.Blocks(title="YOLO OBB 样例图地理检测 Demo") as demo:
        gr.Markdown("# YOLO OBB 样例图地理检测 Demo")
        gr.Markdown("选择 `sample_100_mix/` 中的一张图片，执行一次 OBB 检测并查看地理结果。")
        with gr.Row():
            image_selector = gr.Dropdown(
                choices=sample_images,
                value=default_value,
                label="样例图片",
            )
            run_button = gr.Button("开始检测", variant="primary")

        status = gr.Markdown("请选择一张样例图后开始检测。")
        image_summary = gr.HTML(EMPTY_SUMMARY_HTML)
        result_image = gr.Image(label="检测结果图", type="pil")
        detection_details = gr.HTML(EMPTY_DETECTIONS_HTML)

        run_button.click(
            fn=run_demo,
            inputs=image_selector,
            outputs=[status, image_summary, result_image, detection_details],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0")
```

- [ ] **Step 4: 运行测试，确认实现通过**

Run: `python -m pytest tests/test_app.py tests/test_data_loader.py tests/test_geo_mapper.py tests/test_inference.py -q`
Expected: PASS

- [ ] **Step 5: 手工启动页面做冒烟验证**

Run: `python app.py`
Expected: 终端输出 Gradio 启动地址；浏览器中可选择样例图、点击检测、看到结果图、图片中心点经纬度，以及最多 5 个折叠检测项

- [ ] **Step 6: 提交本任务改动**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add gradio geo detection demo"
```

## Self-Review Checklist

- [ ] 覆盖 spec 的全部要求：样例图选择、OBB 检测、结果图、图片中心点经纬度、最多 5 个折叠检测项
- [ ] 计划中没有 `TODO`、`TBD`、占位词或“稍后补充”
- [ ] 各任务中的函数名、文件名、命令和测试目标保持一致
- [ ] 依赖安装在 Task 1 完成，后续测试命令不会因缺少基础依赖而失败
