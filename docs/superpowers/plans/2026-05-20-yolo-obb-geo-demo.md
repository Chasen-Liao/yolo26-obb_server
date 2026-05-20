# YOLO OBB 样例图地理检测 Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个仅面向 `sample_100_mix/` 的 Gradio 网页 demo，能对样例图执行 OBB 检测，展示检测结果图、图片中心点经纬度，以及最多 5 个折叠的检测框中心点经纬度信息。

**Architecture:** 采用单页 Gradio 应用作为界面层，使用 `inference.py` 封装 OBB 模型推理，使用 `data_loader.py` 读取样例图与 `geo.json` 元数据，使用 `geo_mapper.py` 负责像素点到经纬度的换算与结果整理。页面层只做输入选择、状态展示和 HTML 结果渲染，不引入独立 API 服务。

**Tech Stack:** Python 3、Gradio、Ultralytics YOLO、Pillow、Pytest

---

### Task 1: 引导项目依赖与基础文档

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/test_project_files.py`

- [ ] **Step 1: 写基础文件的失败测试**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    for filename in [".gitignore", "requirements.txt", "README.md"]:
        assert (ROOT / filename).exists(), f"missing {filename}"


def test_requirements_list_runtime_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package_name in ["gradio", "ultralytics", "pillow", "pytest"]:
        assert package_name in requirements


def test_readme_describes_demo_usage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Gradio" in readme
    assert "sample_100_mix" in readme
    assert "python app.py" in readme
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_project_files.py -q`
Expected: FAIL，提示缺少 `.gitignore`、`requirements.txt` 或 `README.md`

- [ ] **Step 3: 写最小实现**

`.gitignore`

```gitignore
__pycache__/
.pytest_cache/
.venv/
.gradio/
```

`requirements.txt`

```text
gradio>=4.44,<5.0
ultralytics>=8.3,<9.0
pillow>=10.0,<11.0
pytest>=8.0,<9.0
```

`README.md`

```markdown
# YOLO OBB 样例图地理检测 Demo

这是一个基于 `sample_100_mix/` 样例图与 `yolo26n_obb_fair1m.pt` 权重的轻量 Gradio demo。

## 功能
- 从 `sample_100_mix/` 中选择样例图
- 使用 OBB 模型执行单图检测
- 展示带检测框的结果图
- 展示图片中心点经纬度
- 展示最多 5 个折叠的检测框中心点经纬度结果

## 环境准备
```bash
python -m pip install -r requirements.txt
```

## 运行方式
```bash
python app.py
```

启动后在浏览器中打开 Gradio 输出的地址。

## 当前限制
- 仅支持 `sample_100_mix/` 目录中的样例图
- 不支持上传自定义图片
- 不返回检测框四角点经纬度
```

- [ ] **Step 4: 安装依赖并重新运行测试**

Run: `python -m pip install -r requirements.txt && python -m pytest tests/test_project_files.py -q`
Expected: PASS

- [ ] **Step 5: 提交本任务改动**

```bash
git add .gitignore requirements.txt README.md tests/test_project_files.py
git commit -m "chore: add geo demo project scaffold"
```

### Task 2: 实现样例图与地理元数据加载

**Files:**
- Create: `data_loader.py`
- Create: `geo_mapper.py`
- Create: `tests/test_data_loader.py`
- Create: `tests/test_geo_mapper.py`

- [ ] **Step 1: 写数据加载与坐标映射的失败测试**

`tests/test_data_loader.py`

```python
from data_loader import get_image_geo_record, get_image_path, list_sample_images


def test_list_sample_images_contains_known_files():
    images = list_sample_images()
    assert "train__t_10144.jpg" in images
    assert "val__v_611.jpg" in images


def test_get_image_path_resolves_existing_sample():
    image_path = get_image_path("train__t_10144.jpg")
    assert image_path.name == "train__t_10144.jpg"
    assert image_path.exists()


def test_get_image_geo_record_reads_expected_center():
    record = get_image_geo_record("train__t_10144.jpg")
    assert record["image_name"] == "t_10144.jpg"
    assert record["center"] == [118.1210570438674, 24.536394476542437]
```

`tests/test_geo_mapper.py`

```python
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
    assert lon == pytest.approx(118.1218881484858)
    assert lat == pytest.approx(24.539036221506002)


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
    assert enriched[0]["geo_center"][0] == pytest.approx(118.1218881484858)
    assert enriched[0]["geo_center"][1] == pytest.approx(24.539036221506002)
```

- [ ] **Step 2: 运行测试，确认它失败**

Run: `python -m pytest tests/test_data_loader.py tests/test_geo_mapper.py -q`
Expected: FAIL，提示 `ModuleNotFoundError` 或缺少待实现函数

- [ ] **Step 3: 写最小实现**

`data_loader.py`

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT_DIR / "sample_100_mix"
GEO_JSON_PATH = SAMPLE_DIR / "geo.json"
MODEL_PATH = ROOT_DIR / "yolo26n_obb_fair1m.pt"


@lru_cache(maxsize=1)
def load_geo_payload() -> dict[str, Any]:
    if not GEO_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing geo metadata file: {GEO_JSON_PATH}")
    return json.loads(GEO_JSON_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_geo_index() -> dict[str, dict[str, Any]]:
    payload = load_geo_payload()
    return payload["images"]


def list_sample_images() -> list[str]:
    images: list[str] = []
    for image_key in sorted(load_geo_index()):
        image_path = SAMPLE_DIR / f"{image_key}.jpg"
        if image_path.exists():
            images.append(image_path.name)
    return images


def get_image_key(filename: str) -> str:
    return Path(filename).stem


def get_image_path(filename: str) -> Path:
    image_path = SAMPLE_DIR / filename
    if not image_path.exists():
        raise FileNotFoundError(f"Missing sample image: {image_path}")
    return image_path


def get_image_geo_record(filename: str) -> dict[str, Any]:
    image_key = get_image_key(filename)
    geo_index = load_geo_index()
    if image_key not in geo_index:
        raise KeyError(f"Missing geo record for image: {filename}")
    return geo_index[image_key]


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
