# YOLO OBB Geo API 使用文档

基于 YOLOv8-OBB 的旋转目标检测 + 地理坐标映射 API 服务。

> 交互式文档：启动服务后访问 `http://<host>:<port>/docs`（Swagger UI）或 `/redoc`（ReDoc）

---

## 目录

- [快速开始](#快速开始)
- [环境配置](#环境配置)
- [鉴权](#鉴权)
- [API 接口](#api-接口)
  - [健康检查](#1-健康检查)
  - [模型状态](#2-模型状态)
  - [获取配置](#3-获取配置)
  - [更新配置](#4-更新配置)
  - [同步检测](#5-同步检测)
  - [创建异步任务](#6-创建异步任务)
  - [查询异步任务](#7-查询异步任务)
  - [获取任务结果图片](#8-获取任务结果图片)
  - [删除任务](#9-删除任务)
  - [监控指标](#10-监控指标)
- [错误码](#错误码)
- [Python 调用示例](#python-调用示例)

---

## 快速开始

```bash
# 安装依赖
uv sync

# 启动服务（开发模式）
uv run uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8003 --reload

# 快速验证
curl http://127.0.0.1:8003/v1/health
```

## 环境配置

复制模板并按需修改：

```bash
cp .env.example .env
vim .env
```

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 模型文件路径 | `yolo26n_obb_fair1m.pt` |
| `IMG_SIZE` | 推理图像尺寸 | `1024` |
| `OBJ_THRESH` | 目标置信度阈值 | `0.25` |
| `NMS_THRESH` | NMS 阈值 | `0.45` |
| `REQUEST_TIMEOUT_SEC` | 同步请求超时（秒） | `30` |
| `MAX_IMAGE_BYTES` | 最大上传图片字节数 | `104857600`（100MB） |
| `MAX_PIXELS` | 最大像素数 | `100000000`（1亿） |
| `RETURN_IMAGE_MODE` | 默认图片返回模式 | `none` |
| `ASYNC_WORKERS` | 异步任务线程数 | `1` |
| `API_KEY` | 鉴权密钥（空=不鉴权） | 空 |

> 环境变量优先级高于 `.env` 文件：`API_KEY=xxx uv run ...` 会覆盖 `.env` 中的值。

## 鉴权

- `.env` 中 `API_KEY=` 留空或未设置 → **鉴权关闭**（默认）
- 设置 `API_KEY=your-secret` → **所有接口需鉴权**

鉴权方式：请求头 `X-API-Key`

```bash
# 鉴权开启时
curl -H "X-API-Key: your-secret" http://127.0.0.1:8003/v1/health

# 密钥错误 → 401
```

---

## API 接口

### 1. 健康检查

```
GET /v1/health
```

**响应**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_sec": 120
}
```

---

### 2. 模型状态

```
GET /v1/model/status
```

**响应**

```json
{
  "loaded": true,
  "model_path": "yolo26n_obb_fair1m.pt",
  "img_size": 1024,
  "obj_thresh": 0.25,
  "nms_thresh": 0.45
}
```

> 模型采用懒加载，首次检测时自动加载。`loaded: false` 时调用检测接口会触发加载。

---

### 3. 获取配置

```
GET /v1/config
```

**响应**

```json
{
  "model_path": "yolo26n_obb_fair1m.pt",
  "img_size": 1024,
  "obj_thresh": 0.25,
  "nms_thresh": 0.45,
  "request_timeout_sec": 30.0,
  "max_image_bytes": 104857600,
  "max_pixels": 100000000
}
```

---

### 4. 更新配置

```
PATCH /v1/config
```

**请求体**（JSON，只传需要修改的字段）

```json
{
  "obj_thresh": 0.5,
  "nms_thresh": 0.3,
  "request_timeout_sec": 60
}
```

可修改字段：`obj_thresh`、`nms_thresh`、`img_size`、`request_timeout_sec`、`max_image_bytes`、`max_pixels`

**响应**

```json
{
  "updated": true,
  "config": { ... }
}
```

---

### 5. 同步检测

```
POST /v1/detect
```

Content-Type: `multipart/form-data`

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 上传图片（jpg/png） |
| `image_mode` | string | 否 | 图片返回模式：`none`/`base64`/`binary`，默认取 `.env` 中 `RETURN_IMAGE_MODE` |
| `return_image` | bool | 否 | 是否生成标注图片，默认 `true` |
| `return_boxes` | bool | 否 | 是否返回检测框，默认 `true` |
| `obj_thresh` | float | 否 | 本次推理置信度阈值（0~1），不传则用全局配置 |
| `nms_thresh` | float | 否 | 本次推理 NMS 阈值（0~1），不传则用全局配置 |

**image_mode 说明**

| 模式 | 说明 |
|------|------|
| `none` | 只返回 JSON 检测结果，不含图片（**推荐，最常用**） |
| `base64` | JSON 中 `image_result.value` 含 base64 编码的标注图片 |
| `binary` | 直接返回 JPEG 图片二进制流，检测元数据在响应头 |

**响应（image_mode=none）**

```json
{
  "request_id": "ac757def-016e-4f9c-8230-51d3f14f12f1",
  "image": {
    "file_name": "train__t_10144.jpg",
    "width": 1000,
    "height": 1000,
    "center_geo": [118.12106, 24.53639]
  },
  "detections": [
    {
      "index": 0,
      "class_id": 0,
      "class_name": "A220",
      "confidence": 0.968,
      "polygon": [[598.94, 136.70], [619.30, 91.14], [581.04, 74.04], [560.67, 119.60]],
      "pixel_center": [589.99, 105.37],
      "geo_center": [118.12180, 24.53928]
    }
  ],
  "perf": {
    "total_ms": 156.3
  },
  "image_result": {
    "mode": "none",
    "value": null
  }
}
```

**响应（image_mode=binary）**

直接返回 `image/jpeg` 二进制流，元数据在响应头：

| 响应头 | 说明 |
|--------|------|
| `X-Request-ID` | 请求 ID |
| `X-Detections-Count` | 检测目标数 |
| `X-Perf-Total-Ms` | 推理耗时（ms） |

**示例**

```bash
# JSON 模式（推荐）
curl -X POST "http://127.0.0.1:8003/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "image_mode=none"

# base64 模式
curl -X POST "http://127.0.0.1:8003/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "image_mode=base64"

# binary 模式，保存图片
curl -X POST "http://127.0.0.1:8003/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "image_mode=binary" \
  -o result.jpg

# 自定义阈值
curl -X POST "http://127.0.0.1:8003/v1/detect" \
  -F "image=@sample_100_mix/train__t_10144.jpg" \
  -F "obj_thresh=0.5" \
  -F "nms_thresh=0.3"
```

---

### 6. 创建异步任务

适用于大图或批量场景，避免同步超时。

```
POST /v1/detect/jobs
```

**参数**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 上传图片（jpg/png） |
| `return_image` | bool | 否 | 是否生成标注图片，默认 `true` |
| `return_boxes` | bool | 否 | 是否返回检测框，默认 `true` |
| `obj_thresh` | float | 否 | 本次推理置信度阈值 |
| `nms_thresh` | float | 否 | 本次推理 NMS 阈值 |

**响应**

```json
{
  "job_id": "e0a15b42-2566-4722-9639-0152f4457aa5",
  "status": "queued"
}
```

**示例**

```bash
curl -X POST "http://127.0.0.1:8003/v1/detect/jobs" \
  -F "image=@sample_100_mix/train__t_10144.jpg"
```

---

### 7. 查询异步任务

```
GET /v1/detect/jobs/{job_id}
```

**响应（运行中）**

```json
{
  "job_id": "e0a15b42-...",
  "status": "running"
}
```

**响应（成功）**

```json
{
  "job_id": "e0a15b42-...",
  "status": "succeeded",
  "result": {
    "image": { ... },
    "detections": [ ... ],
    "perf": { "total_ms": 156.3 }
  }
}
```

> 注意：`result` 中不含 `image_jpeg` 二进制字段，需通过单独接口获取图片。

**响应（失败）**

```json
{
  "job_id": "e0a15b42-...",
  "status": "failed",
  "error": {
    "code": "INVALID_IMAGE",
    "message": "...",
    "details": {}
  }
}
```

**任务状态流转**

```
queued → running → succeeded
                 → failed
```

**示例**

```bash
curl "http://127.0.0.1:8003/v1/detect/jobs/e0a15b42-2566-4722-9639-0152f4457aa5"
```

---

### 8. 获取任务结果图片

```
GET /v1/detect/jobs/{job_id}/image
```

返回 `image/jpeg` 二进制流。任务状态为 `succeeded` 且 `return_image=true` 时可用。

**示例**

```bash
curl "http://127.0.0.1:8003/v1/detect/jobs/e0a15b42-.../image" -o result.jpg
```

---

### 9. 删除任务

```
DELETE /v1/detect/jobs/{job_id}
```

**响应**

```json
{
  "deleted": true,
  "job_id": "e0a15b42-..."
}
```

---

### 10. 监控指标

```
GET /v1/metrics
```

返回 Prometheus 文本格式指标，无需鉴权。

```
obb_geo_api_requests_total 5
obb_geo_api_requests_success_total 3
obb_geo_api_requests_failed_total 1
obb_geo_api_detect_sync_total 3
obb_geo_api_detect_async_total 2
obb_geo_api_detect_timeout_total 0
obb_geo_api_uptime_seconds 300
```

---

## 错误码

| HTTP 状态码 | code | 说明 |
|-------------|------|------|
| 400 | `INVALID_IMAGE_MODE` | image_mode 不是 none/base64/binary |
| 400 | `INVALID_OBJ_THRESH` | obj_thresh 不在 0~1 范围 |
| 400 | `INVALID_NMS_THRESH` | nms_thresh 不在 0~1 范围 |
| 400 | `INVALID_IMAGE` | 图片解码失败 |
| 400 | `UNKNOWN_GEO_FILENAME` | geo.json 中无对应文件记录 |
| 401 | `UNAUTHORIZED` | API 密钥无效 |
| 404 | `JOB_NOT_FOUND` | 任务不存在 |
| 404 | `IMAGE_NOT_FOUND` | 结果中无图片 |
| 409 | `JOB_NOT_READY` | 任务尚未完成，无法获取图片 |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | 仅支持 jpg/png |
| 500 | `MODEL_NOT_FOUND` | 模型文件不存在 |
| 504 | `REQUEST_TIMEOUT` | 同步检测超时 |

错误响应格式：

```json
{
  "detail": {
    "code": "INVALID_IMAGE_MODE",
    "message": "image_mode must be one of binary/base64/none.",
    "request_id": "2e488962-...",
    "details": {}
  }
}
```

---

## Python 调用示例

```python
import requests

BASE = "http://127.0.0.1:8003"
HEADERS = {"X-API-Key": "your-secret"}  # 鉴权开启时需要

# 同步检测
with open("sample_100_mix/train__t_10144.jpg", "rb") as f:
    resp = requests.post(
        f"{BASE}/v1/detect",
        files={"image": f},
        data={"image_mode": "none"},
        headers=HEADERS,
    )
result = resp.json()
for det in result["detections"]:
    print(f"{det['class_name']} conf={det['confidence']:.3f} geo={det['geo_center']}")

# 异步检测
with open("sample_100_mix/train__t_10144.jpg", "rb") as f:
    job = requests.post(f"{BASE}/v1/detect/jobs", files={"image": f}, headers=HEADERS).json()

import time
while True:
    status = requests.get(f"{BASE}/v1/detect/jobs/{job['job_id']}", headers=HEADERS).json()
    if status["status"] in ("succeeded", "failed"):
        break
    time.sleep(1)

if status["status"] == "succeeded":
    print(f"检测到 {len(status['result']['detections'])} 个目标")
    # 获取标注图片
    img = requests.get(f"{BASE}/v1/detect/jobs/{job['job_id']}/image", headers=HEADERS)
    with open("result.jpg", "wb") as f:
        f.write(img.content)
```
