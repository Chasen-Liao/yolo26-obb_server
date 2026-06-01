# YOLO OBB Geo API

基于 YOLOv8-OBB 旋转目标检测 + 地理坐标映射的 API 服务，同时提供交互式 Demo 界面。

**核心能力**：上传航拍/卫星图像，检测任意方向的旋转目标（飞机、船舶等），并直接将检测框中心点映射为经纬度坐标。

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45%2B-FF4B4B)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-UNLICENSED-red)](#)

> **Latest stable**: `v0.1.0` — API 基础配置、异步任务生命周期与 `.env` 模板已就绪；详见 [`README_API.md`](./README_API.md) 与 `docs/superpowers/` 设计文档。

---

## 快速体验

### Demo 界面（适合快速目视验证）

```bash
streamlit run demo/app.py --server.address 0.0.0.0
```

访问 `http://<host>:8501`，选择样例图即可查看检测结果与地理信息。

详细说明 → [`demo/README.md`](./demo/README.md)

### API 服务（适合程序调用或集成）

```bash
uv sync
uv run uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8003 --reload
```

启动后访问 `http://<host>:8003/docs`（Swagger UI）或 `/redoc`（ReDoc）。

详细接口文档 → [`README_API.md`](./README_API.md)

---

## 目录结构

```
├── obb_geo_api_server.py   # FastAPI 服务入口
├── obb_geo_service.py      # 核心检测/地理映射逻辑
├── main.py                  # 命令行入口
├── pyproject.toml           # 依赖管理
├── .env.example             # 环境变量模板
│
├── demo/                    # 交互式 Demo（Streamlit）
│   ├── app.py
│   ├── data_loader.py
│   ├── geo_mapper.py
│   ├── inference.py
│   └── tests/
│
├── tests/                   # 单元测试
│
├── sample_100_mix/          # 样例数据（图片 + geo.json）
│
└── docs/superpowers/        # 设计文档与计划
```

> **配套前端 Agent**：本仓库的 `sample_100_mix/` 样例数据会被同级的 [`@chasen-og/yolo-agent`](https://www.npmjs.com/package/@chasen-og/yolo-agent) 通过 `YOLO26_DEMO_IMAGE_DIR` 读取，提供 `/demo` 命令的可视化样例。推荐两个仓库平级 clone（`yolo26-obb_server/` 与 `yolomercury/`），无需任何额外配置。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 检测模型 | YOLOv8-OBB（`ultralytics`） |
| API 框架 | FastAPI + Uvicorn |
| Demo 界面 | Streamlit |
| 地理映射 | 仿射变换（基于 `geo.json` 元数据） |
| 环境管理 | `uv` |

---

## 配置

复制模板并按需修改：

```bash
cp .env.example .env
```

主要环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_PATH` | 模型权重路径 | `yolo26n_obb_fair1m.pt` |
| `IMG_SIZE` | 推理图像尺寸 | `1024` |
| `OBJ_THRESH` | 目标置信度阈值 | `0.25` |
| `NMS_THRESH` | NMS 阈值 | `0.45` |
| `API_KEY` | 鉴权密钥（空=不鉴权） | 空 |

---

## 测试

```bash
uv run pytest tests/ -q
```

> **会话快照**：根目录的 `.tmux_session` 记录了 `yolo-obb-api` 与 `yolo-obb-demo` 两个常用会话名，便于用 `tmuxinator`/`tmuxp` 一键拉起。可按需在 `.gitignore` 中加入 `.tmux_session*` 排除本地变种。

---

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/health` | 健康检查 |
| GET | `/v1/model/status` | 模型状态 |
| GET | `/v1/config` | 获取配置 |
| PATCH | `/v1/config` | 更新配置 |
| POST | `/v1/detect` | 同步检测 |
| POST | `/v1/detect/jobs` | 创建异步任务 |
| GET | `/v1/detect/jobs/{job_id}` | 查询异步任务 |
| GET | `/v1/detect/jobs/{job_id}/image` | 获取结果图片 |
| DELETE | `/v1/detect/jobs/{job_id}` | 删除任务 |
| GET | `/v1/metrics` | Prometheus 监控指标 |