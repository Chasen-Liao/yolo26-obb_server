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
当前仓库处于 Task 1 阶段，尚未提供可直接运行的 `app.py`。后续实现完成后，计划使用以下命令启动：

```bash
python app.py
```

届时启动后可在浏览器中打开 Gradio 输出的地址。

## 当前限制
- 仅支持 `sample_100_mix/` 目录中的样例图
- 不支持上传自定义图片
- 不返回检测框四角点经纬度
