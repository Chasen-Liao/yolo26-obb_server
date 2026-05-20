from __future__ import annotations

from html import escape
from typing import Any

from data_loader import get_image_path, list_sample_images
from geo_mapper import attach_geo_centers, build_image_summary
from inference import run_inference


EMPTY_SUMMARY_HTML = "<p>暂无图片信息。</p>"
EMPTY_DETECTIONS_HTML = "<p>暂无检测结果。</p>"
FILE_READ_ERROR_MESSAGE = "图片文件不存在或无法读取。"
GEO_MAPPING_ERROR_MESSAGE = "地理信息缺失，无法完成映射。"


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
    except FileNotFoundError:
        return FILE_READ_ERROR_MESSAGE, EMPTY_SUMMARY_HTML, None, EMPTY_DETECTIONS_HTML
    except KeyError:
        return GEO_MAPPING_ERROR_MESSAGE, EMPTY_SUMMARY_HTML, None, EMPTY_DETECTIONS_HTML

    status_text = "检测完成。" if enriched_detections else "未检测到目标。"
    return (
        status_text,
        render_image_summary(summary),
        plotted_image,
        render_detection_details(enriched_detections),
    )


def build_app():
    import gradio as gr

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
