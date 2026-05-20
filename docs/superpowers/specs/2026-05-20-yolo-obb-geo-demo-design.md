# YOLO OBB 地理检测 API 服务设计文档

## 背景
当前仓库已经具备 3 类基础能力：

- `demo/inference.py`：基于 `yolo26n_obb_fair1m.pt` 的单图 OBB 检测
- `demo/geo_mapper.py`：通过 `geo.json` 中的 affine 参数将像素点映射为经纬度
- `demo/data_loader.py`：读取 `sample_100_mix/` 样例集、模型文件和图片地理记录

现状问题不是“能否检测”，而是“缺少标准 API 服务封装”。

目标是将当前能力整理成一个整体风格对齐 `/data/RK/yolov8_server/` 的 HTTP API 服务：外部系统上传一张样例集中的图片文件，服务完成 OBB 检测，并返回检测框信息、图片中心经纬度和检测框中心经纬度等完整结果。

## 目标
构建一个与参考项目接口风格一致的 FastAPI 服务，满足以下要求：

1. 客户端通过 `multipart/form-data` 上传图片文件。
2. 服务保留上传文件原始文件名，不做重命名。
3. 服务使用原始文件名去 `sample_100_mix/geo.json` 匹配地理记录。
4. 匹配成功时，执行一次 OBB 检测并返回图片中心经纬度。
5. 每个检测框返回类别、置信度、像素中心点、经纬度中心点。
6. 主检测接口支持与参考项目一致的结果图返回模式：`binary`、`base64`、`none`。
7. 提供与参考项目一致的同步接口、异步任务接口、健康检查和基础指标接口。

## 非目标
首版 API 服务不包含以下能力：

- 对任意外部图片自动生成地理信息
- 上传后改名、重命名映射或模糊文件名匹配
- 返回 OBB 四顶点经纬度
- 数据库持久化
- 分布式任务队列
- 鉴权之外的复杂权限系统
- 与 `demo/` 页面合并成单一入口

## 方案对比

### 方案 A：单文件 API 入口直接复用 demo 模块
只新增一个 FastAPI 文件，在路由中直接调用 `demo/inference.py`、`demo/geo_mapper.py` 和 `demo/data_loader.py`，完成文件校验、推理和返回。

优点：改动最少、落地最快。  
缺点：路由层会承载过多业务逻辑，后续加异步任务、指标和配置更新时容易变乱。

### 方案 B：参考项目同构的 API 层 + Service 层
整体对齐 `/data/RK/yolov8_server/` 结构，新增 API 入口与服务层。API 层负责参数校验、同步/异步任务和响应格式；Service 层负责模型加载、图片解析、OBB 检测、地理匹配和结果组装。

优点：结构清晰、扩展成本低、与参考项目最一致。  
缺点：比方案 A 多一些文件和封装成本。

### 方案 C：一步到位做成完整产品化服务框架
除方案 B 外，再引入更完整的 schema、配置模型、响应模型、统一错误系统和更重的模块抽象。

优点：规范度最高。  
缺点：超出当前目标，首版实现成本偏高。

## 最终选择
选择 **方案 B：参考项目同构的 API 层 + Service 层**。

理由：用户明确要求“整体参考那个项目”，因此首版不应只借鉴单个接口，而应整体沿用参考项目的服务组织方式。方案 B 在保持实现克制的前提下，最大程度复用当前能力，并对外提供与参考项目一致的使用体验。

## 整体架构
服务按与参考项目一致的两层结构组织：

1. `obb_geo_api_server.py`  
   FastAPI 应用入口。负责路由注册、API Key 校验、请求参数校验、同步检测、异步任务管理、配置读取与更新、指标输出。

2. `obb_geo_service.py`  
   业务服务层。负责模型懒加载、图片解码、文件合法性检查、原始文件名匹配 `geo.json`、执行 OBB 推理、生成标注图、拼装地理结果与性能信息。

现有 `demo/` 模块作为底层复用能力来源：

- `demo/inference.py`：继续负责模型加载与 OBB 检测结果提取
- `demo/geo_mapper.py`：继续负责像素中心点到经纬度中心点的映射
- `demo/data_loader.py`：继续负责样例集图片记录与 `geo.json` 数据读取

如果在实现中发现 `demo/` 命名不适合作为服务依赖，仅允许做小范围职责抽取，不做无关重构。

## 接口设计
首版接口集合与 `/data/RK/yolov8_server/` 对齐：

- `GET /v1/health`
- `GET /v1/model/status`
- `GET /v1/config`
- `PATCH /v1/config`
- `POST /v1/detect`
- `POST /v1/detect/jobs`
- `GET /v1/detect/jobs/{job_id}`
- `GET /v1/detect/jobs/{job_id}/image`
- `DELETE /v1/detect/jobs/{job_id}`
- `GET /v1/metrics`

其中 `POST /v1/detect` 是主同步接口；异步任务接口主要用于保持与参考项目一致的调用模式。

## 请求与返回规则

### `POST /v1/detect` 请求字段
请求格式为 `multipart/form-data`，字段尽量与参考项目一致：

- `image`：必填，上传图片文件
- `return_image`：可选，是否返回结果图
- `return_boxes`：可选，是否返回检测框列表
- `image_mode`：可选，`binary | base64 | none`
- `obj_thresh`：可选，请求级置信度阈值覆盖
- `nms_thresh`：可选，请求级 NMS 阈值覆盖

### 文件名匹配规则
这是本服务与普通检测服务的关键差异：

1. 服务读取上传文件的原始文件名。
2. 不对文件名做重命名、标准化重写或随机化处理。
3. 以原始文件名的 stem 去 `sample_100_mix/geo.json` 中匹配图片记录。
4. 匹配失败时，返回明确业务错误，不尝试模糊匹配。

### JSON 返回结构
当返回 JSON 时，响应主体建议包含以下结构：

- `request_id`
- `image`
  - `file_name`
  - `width`
  - `height`
  - `center_geo`
- `detections`
  - `class_id`
  - `class_name`
  - `confidence`
  - `pixel_center`
  - `geo_center`
- `perf`
- `image_result`
  - `mode`
  - `value`

字段语义：

- `image.center_geo`：图片中心点经纬度，直接取自 `geo.json` 中记录
- `detections[].pixel_center`：OBB 四点均值中心，沿用当前 `demo/inference.py` 结果
- `detections[].geo_center`：通过 affine 映射计算出的检测框中心经纬度

### 图片返回模式
图片返回行为与参考服务保持一致：

- `image_mode=binary` 且 `return_image=true`：直接返回 JPEG 二进制
- `image_mode=base64` 且 `return_image=true`：在 JSON 中返回 base64 图片
- `image_mode=none` 或 `return_image=false`：只返回结构化检测结果

当返回二进制图片时，仍通过响应头返回基础元信息，例如：

- `X-Request-ID`
- `X-Detections-Count`
- `X-Perf-Total-Ms`

这一点也与参考项目保持一致。

## 数据流与处理流程

### 同步检测流程
1. API 层接收上传文件和表单参数。
2. 校验 `image_mode`、阈值范围、媒体类型和 API Key。
3. 读取上传文件 bytes，并保留原始文件名。
4. Service 层解码图片，确认图片内容有效。
5. 使用原始文件名在 `geo.json` 中查找图片记录。
6. 调用 OBB 模型执行检测。
7. 生成检测结果图和结构化检测框列表。
8. 从 `geo.json` 中提取图片尺寸、图片中心点和 affine 参数。
9. 将每个检测框的像素中心点映射为经纬度中心点。
10. 组装检测结果、性能信息和可选结果图并返回。

### 异步任务流程
异步流程与参考项目保持一致：

1. `POST /v1/detect/jobs` 创建任务并提交线程池执行。
2. `GET /v1/detect/jobs/{job_id}` 查询任务状态和结构化结果。
3. `GET /v1/detect/jobs/{job_id}/image` 获取任务结果图。
4. `DELETE /v1/detect/jobs/{job_id}` 删除任务记录。

异步接口不新增额外业务语义，只负责保持调用体验一致。

## 数据模型

### 图片结果模型
每次成功检测至少返回以下图片级字段：

- `file_name`
- `width`
- `height`
- `center_geo`

其中 `center_geo = [lon, lat]`。

### 单个检测框模型
每个检测框至少包含：

- `index`
- `class_id`
- `class_name`
- `confidence`
- `pixel_center`
- `geo_center`

其中：

- `pixel_center = [x, y]`
- `geo_center = [lon, lat]`

首版不强制返回 polygon 顶点和顶点经纬度，但内部可继续保留 polygon 原始信息，便于后续扩展。

## 配置项
配置项也尽量与参考项目对齐，首版建议支持：

- `MODEL_PATH`
- `API_KEY`
- `RETURN_IMAGE_MODE`
- `ASYNC_WORKERS`
- `REQUEST_TIMEOUT_SEC`
- `OBJ_THRESH`
- `NMS_THRESH`

如果现有 OBB 推理逻辑没有必要参数，则不额外引入新的环境变量。首版遵循“对齐参考项目，但不做推测性扩展”。

## 错误处理
错误风格与参考项目一致，并补足当前业务场景需要的错误码。

### 需要覆盖的错误场景
1. 文件类型不支持：仅支持 `jpg/png`
2. 上传文件名为空或原始文件名不合法
3. 文件名在 `sample_100_mix/geo.json` 中不存在
4. 图片字节存在但内容损坏，无法解码
5. 模型推理失败
6. 地理记录存在但关键字段缺失，无法完成映射
7. 推理超时

### 建议状态码
- 参数错误：`400`
- 鉴权失败：`401`
- 地理记录找不到：`404`
- 文件类型不支持：`415`
- 任务结果未就绪：`409`
- 推理超时：`504`
- 其他内部错误：`500`

### 错误返回原则
- 返回结构尽量沿用参考服务的 `code`、`message`、`request_id`、`details`
- 不吞掉底层异常，但对外只暴露足够定位问题的信息
- 文件名无法匹配地理记录时，明确告诉调用方是“地理记录未找到”，而不是笼统的检测失败

## 测试与验证范围

### 自动化测试
首版至少覆盖以下行为：

- `GET /v1/health` 正常返回
- `GET /v1/model/status` 正常返回
- `POST /v1/detect` 成功路径
- `POST /v1/detect/jobs` 到结果查询的异步链路
- `image_mode=binary | base64 | none` 三种模式
- 文件名无法匹配 `geo.json` 时返回预期错误
- 成功结果中包含图片中心经纬度
- 成功结果中包含检测框中心像素点和检测框中心经纬度

### 手工验证
至少验证以下命令链路：

- 启动 `uvicorn obb_geo_api_server:app --host 0.0.0.0 --port 8001`
- 访问 `/docs` 可查看 Swagger 文档
- 上传样例集中的图片文件可返回检测结果
- `image_mode=base64` 时 JSON 中能拿到结果图
- `image_mode=binary` 时能直接输出标注图

## 文件规划
实现完成后，项目应至少新增或修改这些文件：

- 新增：`obb_geo_api_server.py`
- 新增：`obb_geo_service.py`
- 修改：`requirements.txt`（加入 API 服务依赖）
- 修改：`README.md` 或新增 API 使用说明文档
- 新增：与 API 服务对应的测试文件

现有 `demo/` 目录保留，用于继续支撑演示和复用底层逻辑，不与 API 服务入口混合。

## 成功标准
当以下条件满足时，视为首版 API 服务达成目标：

1. 服务可通过 `uvicorn` 正常启动。
2. 接口集合与 `/data/RK/yolov8_server/` 的主体结构保持一致。
3. 主接口可接收上传图片文件并执行 OBB 检测。
4. 服务保留原始文件名并用其匹配 `geo.json`。
5. 成功返回中包含图片中心经纬度。
6. 成功返回中每个检测框包含类别、置信度、像素中心点、经纬度中心点。
7. 结果图返回模式支持 `binary`、`base64`、`none`。
8. 文件名不匹配、图片非法、超时等常见失败路径有清晰错误响应。

## 后续可扩展方向
这些能力明确不进入首版，但为后续扩展预留空间：

- 返回 OBB 四顶点像素坐标和四顶点经纬度
- 支持上传任意图片并配套上传地理参数文件
- 引入更细粒度的响应模型和 OpenAPI schema
- 增加结果缓存、批量检测和限流
- 扩展更多数据集与多模型切换
