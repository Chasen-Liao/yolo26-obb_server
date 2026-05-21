# YOLO OBB Geo API Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个整体风格对齐 `/data/RK/yolov8_server/` 的 FastAPI 服务，接收上传图片文件，保留原始文件名并用其匹配 `sample_100_mix/geo.json`，返回 OBB 检测结果、图片中心经纬度、检测框中心像素点和检测框中心经纬度。

**Architecture:** 保留 `demo/` 中已验证的样例推理与地理映射逻辑，在项目根目录新增 `obb_geo_service.py` 作为服务层，负责模型加载、图片解码、YOLO OBB 推理、`geo.json` 匹配和统一结果组装；新增 `obb_geo_api_server.py` 作为 API 层，负责 FastAPI 路由、可选鉴权、同步检测、异步任务、配置和指标接口。测试按服务层、同步接口、异步接口和依赖文档分层覆盖。项目使用 uv 管理。

**Tech Stack:** Python 3、uv、FastAPI、Uvicorn、python-multipart、Ultralytics YOLO、Pillow、Pytest、httpx

---

**Review 修项对照：**

| # | 原问题 | 修法 |
|---|--------|------|
| 1 | 异步任务状态接口会把 `image_jpeg` 二进制直接塞进 JSON，导致序列化失败 | `get_detect_job` 返回时剥离 `image_jpeg`，图片走独立 `/v1/detect/jobs/{job_id}/image` |
| 2 | `TestClient(app)` 触发 startup 事件调 `service.load()`，测试 stub 没有完整生命周期 | 移除 startup 事件中的 eager load，改为 lazy load（`detect()` 首次调用时自动加载）|
| 3 | 写了 config 接口但没测试 | Task 3 补齐 GET/PATCH `/v1/config` 测试 |
| 4 | 默认 `image_mode=binary` 不符主需求（JSON 检测+经纬度） | 默认改为 `none`；需要图片时显式指定 `base64` 或 `binary` |
| 5 | API Key 鉴权范围膨胀 | 保留可选鉴权（`API_KEY` 为空则跳过），补测试和文档 |
| 6 | 项目迁移到 uv | 用 `uv` 管理 `pyproject.toml` 和依赖，`requirements.txt` 由 `uv export` 生成 |

---

## File Structure

- `pyproject.toml`：uv 管理的项目元数据和依赖
- `requirements.txt`：由 `uv export` 生成，供兼容场景使用
- `obb_geo_service.py`：服务层，处理上传图片、地理匹配、OBB 推理和结果组装
- `obb_geo_api_server.py`：API 层，提供 `/v1/*` 路由、异步任务和 metrics
- `tests/conftest.py`：通用 fixture，提供样例图片名、上传文件 bytes
- `tests/test_api_project_files.py`：检查新增 API 相关文件和依赖
- `tests/test_obb_geo_service.py`：服务层单元测试
- `tests/test_obb_geo_api_sync.py`：同步接口 + config 接口测试
- `tests/test_obb_geo_api_async.py`：异步、metrics 和任务管理测试
- `README_API.md`：新 API 服务的启动与调用说明，避免覆盖现有 demo 文档

## Task 1: uv 初始化 + 依赖 + 测试夹具

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_api_project_files.py`

- [ ] **Step 1: 初始化 uv 项目**

```bash
cd /data/RK/yolo26-obb_server
uv init --no-readme
```

这会生成 `pyproject.toml`。然后添加依赖：

```bash
uv add "streamlit>=1.45,<2.0" "ultralytics>=8.3,<9.0" "pillow>=10.0,<11.0" "fastapi>=0.115,<1.0" "uvicorn>=0.30,<1.0" "python-multipart>=0.0.9,<1.0"
uv add --dev "httpx>=0.27,<1.0" "pytest>=8.0,<9.0"
```

导出 `requirements.txt` 供兼容：

```bash
uv export --no-hashes -o requirements.txt
```

删掉 uv init 自动创建的 `hello.py`（如果存在）。

- [ ] **Step 2: 写依赖与测试夹具的失败测试**

`tests/test_api_project_files.py`

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_includes_api_dependencies():
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package_name in ["fastapi", "uvicorn", "python-multipart", "httpx", "pytest"]:
        assert package_name in content


def test_api_fixture_module_exists():
    fixture_file = ROOT / "tests" / "conftest.py"
    assert fixture_file.exists(), "missing tests/conftest.py"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_api_project_files.py -q`
Expected: FAIL，提示缺少 `pyproject.toml` 中 API 依赖或缺少 `tests/conftest.py`

- [ ] **Step 4: 写 conftest.py**

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

- [ ] **Step 5: 重新运行测试**

Run: `uv run pytest tests/test_api_project_files.py -q`
Expected: PASS

- [ ] **Step 6: 提交本任务改动**

```bash
git add pyproject.toml requirements.txt tests/conftest.py tests/test_api_project_files.py
git commit -m "chore: init uv project with api dependencies and fixtures"
```

## Task 2: 实现服务层配置、图片校验与结果拼装

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
    def plot(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _FakeModel:
    def __init__(self):
        self.names = {0: "ship"}
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
    assert result["detections"][0]["pixel_center"] == [3.0, 4.0]
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


def test_detect_rejects_unknown_geo_filename(monkeypatch, sample_png_bytes):
    service = OBBGeoService(OBBGeoDetectionConfig())
    service._loaded = True
    service._model = _FakeModel()
    monkeypatch.setattr("obb_geo_service.build_image_summary", lambda filename: (_ for _ in ()).throw(KeyError(filename)))

    with pytest.raises(OBBGeoDetectionError) as exc_info:
        service.detect(sample_png_bytes, filename="missing.jpg")

    assert exc_info.value.code == "GEO_RECORD_NOT_FOUND"
    assert exc_info.value.status_code == 404
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_obb_geo_service.py -q`
Expected: FAIL，提示缺少 `obb_geo_service.py` 或缺少 `OBBGeoService`

- [ ] **Step 3: 写最小实现**

`obb_geo_service.py`

```python
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
    request_timeout_sec: float = float(os.getenv("REQUEST_TIMEOUT_SEC", "8"))
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
```

- [ ] **Step 4: 重新运行服务层测试**

Run: `uv run pytest tests/test_obb_geo_service.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_service.py tests/test_obb_geo_service.py
git commit -m "feat: add obb geo detection service"
```

## Task 3: 实现同步 API 接口、图片返回模式与 config 测试

**Files:**
- Create: `obb_geo_api_server.py`
- Create: `tests/test_obb_geo_api_sync.py`

**Review 修项：**
- 移除 startup 事件中的 eager `service.load()`，改为 lazy load（`detect()` 首次调用时自动加载）
- 默认 `image_mode` 从 `binary` 改为 `none`
- 鉴权：`API_KEY` 为空则跳过，有值则校验 `X-API-Key` header

- [ ] **Step 1: 写同步接口失败测试**

`tests/test_obb_geo_api_sync.py`

```python
from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from obb_geo_api_server import app


class _StubService:
    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def detect(self, image_bytes, filename, return_image, return_boxes, obj_thresh, nms_thresh):
        self.calls.append(
            {
                "filename": filename,
                "return_image": return_image,
                "return_boxes": return_boxes,
                "obj_thresh": obj_thresh,
                "nms_thresh": nms_thresh,
            }
        )
        return {
            "image": {
                "file_name": filename,
                "width": 1000,
                "height": 1000,
                "center_geo": [118.121057, 24.536394],
            },
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "ship",
                    "confidence": 0.91,
                    "pixel_center": [20.0, 30.0],
                    "geo_center": [118.12, 24.53],
                }
            ],
            "perf": {"total_ms": 12.5},
            "image_jpeg": b"\xff\xd8\xff\xd9",
        }


def test_detect_returns_json_payload_for_none_mode(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "none", "return_image": "true", "return_boxes": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image"]["file_name"] == sample_upload_name
    assert payload["image"]["center_geo"] == [118.121057, 24.536394]
    assert payload["detections"][0]["geo_center"] == [118.12, 24.53]
    assert payload["image_result"]["mode"] == "none"
    assert stub.calls == [
        {
            "filename": sample_upload_name,
            "return_image": True,
            "return_boxes": True,
            "obj_thresh": None,
            "nms_thresh": None,
        }
    ]


def test_detect_returns_base64_image(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "base64", "return_image": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_result"]["mode"] == "base64"
    assert base64.b64decode(payload["image_result"]["value"]) == b"\xff\xd8\xff\xd9"


def test_detect_returns_binary_image_for_binary_mode(monkeypatch, sample_png_bytes, sample_upload_name):
    stub = _StubService()
    monkeypatch.setattr("obb_geo_api_server.service", stub)
    client = TestClient(app)

    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "binary", "return_image": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-detections-count"] == "1"
    assert response.content == b"\xff\xd8\xff\xd9"


def test_detect_rejects_unknown_image_mode(sample_png_bytes, sample_upload_name):
    client = TestClient(app)
    response = client.post(
        "/v1/detect",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
        data={"image_mode": "bad-mode"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_IMAGE_MODE"


def test_get_config_returns_current_config():
    client = TestClient(app)
    response = client.get("/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "img_size" in data
    assert "obj_thresh" in data
    assert "nms_thresh" in data


def test_patch_config_updates_thresh():
    client = TestClient(app)
    response = client.patch("/v1/config", json={"obj_thresh": 0.5, "nms_thresh": 0.6})
    assert response.status_code == 200
    data = response.json()
    assert data["updated"] is True
    assert data["config"]["obj_thresh"] == 0.5
    assert data["config"]["nms_thresh"] == 0.6
    # restore
    client.patch("/v1/config", json={"obj_thresh": 0.25, "nms_thresh": 0.45})


def test_api_key_rejects_when_set(monkeypatch):
    monkeypatch.setattr("obb_geo_api_server.API_KEY", "test-secret")
    client = TestClient(app)
    response = client.get("/v1/health")
    assert response.status_code == 401

    response = client.get("/v1/health", headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200

    monkeypatch.setattr("obb_geo_api_server.API_KEY", "")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_obb_geo_api_sync.py -q`
Expected: FAIL，提示缺少 `obb_geo_api_server.py` 或缺少 `/v1/detect`

- [ ] **Step 3: 写最小实现**

`obb_geo_api_server.py`

```python
from __future__ import annotations

import base64
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from threading import Lock
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from obb_geo_service import OBBGeoDetectionConfig, OBBGeoDetectionError, OBBGeoService


APP_VERSION = "1.0.0"
START_TS = time.time()
API_KEY = os.getenv("API_KEY", "")
DEFAULT_RETURN_IMAGE_MODE = os.getenv("RETURN_IMAGE_MODE", "none")
ASYNC_WORKERS = int(os.getenv("ASYNC_WORKERS", "1"))

app = FastAPI(title="YOLO OBB Geo API", version=APP_VERSION)
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


@app.on_event("shutdown")
def shutdown_event() -> None:
    service.close()
    executor.shutdown(wait=False, cancel_futures=True)


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
```

**关键修项：**
- 无 `@app.on_event("startup")`，service 在首次 `detect()` 调用时 lazy load
- 默认 `image_mode` 为 `none`（返回纯 JSON 检测结果）
- 鉴权：`API_KEY` 为空则跳过，有值则校验 `X-API-Key` header

- [ ] **Step 4: 重新运行同步接口测试**

Run: `uv run pytest tests/test_obb_geo_api_sync.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_api_server.py tests/test_obb_geo_api_sync.py
git commit -m "feat: add sync obb geo api with config and auth"
```

## Task 4: 补齐异步任务、metrics 和配置接口测试

**Files:**
- Modify: `obb_geo_api_server.py`
- Create: `tests/test_obb_geo_api_async.py`

**Review 修项：**
- `get_detect_job` 返回时剥离 `image_jpeg`，图片走独立 `/v1/detect/jobs/{job_id}/image`
- 异步 job 的 `result` 字段只包含 JSON 安全数据

- [ ] **Step 1: 写异步接口失败测试**

`tests/test_obb_geo_api_async.py`

```python
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from obb_geo_api_server import app


class _AsyncStubService:
    def detect(self, image_bytes, filename, return_image, return_boxes, obj_thresh, nms_thresh):
        return {
            "image": {
                "file_name": filename,
                "width": 1000,
                "height": 1000,
                "center_geo": [118.121057, 24.536394],
            },
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "ship",
                    "confidence": 0.91,
                    "pixel_center": [20.0, 30.0],
                    "geo_center": [118.12, 24.53],
                }
            ],
            "perf": {"total_ms": 12.5},
            "image_jpeg": b"\xff\xd8\xff\xd9",
        }


def test_async_job_lifecycle(monkeypatch, sample_png_bytes, sample_upload_name):
    monkeypatch.setattr("obb_geo_api_server.service", _AsyncStubService())
    client = TestClient(app)

    create_response = client.post(
        "/v1/detect/jobs",
        files={"image": (sample_upload_name, sample_png_bytes, "image/png")},
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["job_id"]

    # Poll until succeeded
    for _ in range(10):
        status_response = client.get(f"/v1/detect/jobs/{job_id}")
        if status_response.json()["status"] == "succeeded":
            break
        time.sleep(0.05)

    final = status_response.json()
    assert final["status"] == "succeeded"
    # job result 不含 image_jpeg（JSON 安全）
    assert "image_jpeg" not in final["result"]
    assert final["result"]["image"]["center_geo"] == [118.121057, 24.536394]
    assert final["result"]["detections"][0]["geo_center"] == [118.12, 24.53]

    # 图片走独立端点
    image_response = client.get(f"/v1/detect/jobs/{job_id}/image")
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.content == b"\xff\xd8\xff\xd9"

    delete_response = client.delete(f"/v1/detect/jobs/{job_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True


def test_async_job_not_found():
    client = TestClient(app)
    response = client.get("/v1/detect/jobs/nonexistent-id")
    assert response.status_code == 404


def test_metrics_endpoint_returns_prometheus_text():
    client = TestClient(app)
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert "obb_geo_api_requests_total" in response.text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_obb_geo_api_async.py -q`
Expected: FAIL，提示缺少异步任务接口或 `/v1/metrics`

- [ ] **Step 3: 补最小实现**

在 `obb_geo_api_server.py` 追加实现：

```python
def _save_job(job_id: str, data: dict[str, Any]) -> None:
    with job_lock:
        jobs[job_id] = data


def _get_job(job_id: str) -> dict[str, Any] | None:
    with job_lock:
        return jobs.get(job_id)


def _json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    """剥离 image_jpeg 等 bytes 字段，确保 JSON 可序列化。"""
    return {k: v for k, v in result.items() if k != "image_jpeg"}


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
```

**关键修项：**
- `_json_safe_result()` 剥离 `image_jpeg`，`get_detect_job` 返回的 result 不会包含二进制数据
- 图片通过独立端点 `/v1/detect/jobs/{job_id}/image` 获取
- `/v1/metrics` 不需要鉴权（方便 Prometheus 拉取）

- [ ] **Step 4: 重新运行异步接口测试**

Run: `uv run pytest tests/test_obb_geo_api_async.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add obb_geo_api_server.py tests/test_obb_geo_api_async.py
git commit -m "feat: add async obb geo api with json-safe job results"
```

## Task 5: 补充 README_API 文档并做端到端回归

**Files:**
- Create: `README_API.md`
- Modify: `tests/test_api_project_files.py`

- [ ] **Step 1: 写文档存在性的失败测试**

在 `tests/test_api_project_files.py` 追加：

```python
def test_api_readme_exists_and_mentions_detect_endpoint():
    readme_path = ROOT / "README_API.md"
    assert readme_path.exists(), "missing README_API.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "obb_geo_api_server:app" in content
    assert "/v1/detect" in content
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_api_project_files.py -q`
Expected: FAIL，提示缺少 `README_API.md`

- [ ] **Step 3: 写最小文档**

`README_API.md`

```markdown
# YOLO OBB Geo API Service

基于 `yolo26n_obb_fair1m.pt` 与 `sample_100_mix/geo.json` 的 FastAPI 检测服务。

## 安装依赖

```bash
uv sync
```

## 启动服务

```bash
uv run uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8001 --reload
```

## 可选鉴权

设置环境变量 `API_KEY` 启用鉴权，请求时需携带 `X-API-Key` header：

```bash
API_KEY=my-secret uv run uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8001
curl -H "X-API-Key: my-secret" http://127.0.0.1:8001/v1/health
```

不设置 `API_KEY` 则鉴权关闭（默认）。

## 主要接口

- `GET /v1/health` — 健康检查
- `GET /v1/model/status` — 模型加载状态
- `GET /v1/config` — 当前配置
- `PATCH /v1/config` — 更新配置
- `POST /v1/detect` — 同步检测
- `POST /v1/detect/jobs` — 创建异步检测任务
- `GET /v1/detect/jobs/{job_id}` — 查询任务状态（JSON，不含图片二进制）
- `GET /v1/detect/jobs/{job_id}/image` — 获取任务结果图片
- `DELETE /v1/detect/jobs/{job_id}` — 删除任务
- `GET /v1/metrics` — Prometheus 格式指标

## image_mode 说明

| 模式 | 说明 | 默认 |
|------|------|------|
| `none` | 只返回 JSON 检测结果，不含图片 | **默认** |
| `base64` | JSON 中 `image_result.value` 含 base64 编码图片 | |
| `binary` | 直接返回 JPEG 图片二进制，检测元数据在响应头 | |

## 示例请求

```bash
# 默认模式（none）：只返回 JSON
curl -X POST "http://127.0.0.1:8001/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg"

# base64 模式：JSON 中包含图片
curl -X POST "http://127.0.0.1:8001/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "image_mode=base64"
```

## 本地验证

```bash
curl http://127.0.0.1:8001/v1/health
curl -X POST "http://127.0.0.1:8001/v1/detect" -F "image=@sample_100_mix/train__t_10144.jpg" -F "image_mode=none"
```
```

- [ ] **Step 4: 运行完整回归测试**

Run: `uv run pytest tests/test_api_project_files.py tests/test_obb_geo_service.py tests/test_obb_geo_api_sync.py tests/test_obb_geo_api_async.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add README_API.md tests/test_api_project_files.py
git commit -m "docs: add obb geo api usage guide"
```

## Task 6: 手工启动验证服务

**Files:**
- (无文件改动，纯手工验证)

- [ ] **Step 1: 启动服务并检查健康接口**

Run: `uv run uvicorn obb_geo_api_server:app --host 127.0.0.1 --port 8001`
Expected: 控制台显示 `Application startup complete.`

- [ ] **Step 2: 在新终端调用健康接口**

Run: `curl http://127.0.0.1:8001/v1/health`
Expected:

```json
{"status":"ok","version":"1.0.0","uptime_sec":1}
```

- [ ] **Step 3: 调用同步检测接口验证 JSON 返回**

Run: `curl -X POST "http://127.0.0.1:8001/v1/detect" -F "image=@sample_100_mix/train__t_10144.jpg" -F "image_mode=none"`
Expected: 返回 JSON，包含 `image.center_geo`、`detections[].pixel_center`、`detections[].geo_center`

- [ ] **Step 4: 验证带鉴权的启动**

Run: `API_KEY=test uv run uvicorn obb_geo_api_server:app --host 127.0.0.1 --port 8002`
在另一个终端：
```bash
curl http://127.0.0.1:8002/v1/health  # 401
curl -H "X-API-Key: test" http://127.0.0.1:8002/v1/health  # 200
```
