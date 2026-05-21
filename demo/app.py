from __future__ import annotations

from html import escape
from typing import Any

try:
    from .data_loader import get_image_path, list_sample_images
    from .geo_mapper import attach_geo_centers, build_image_summary
    from .inference import run_inference
except ImportError:
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


def get_preview_image_path(selected_image: str) -> str | None:
    if not selected_image:
        return None
    try:
        return str(get_image_path(selected_image))
    except FileNotFoundError:
        return None


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
    import streamlit as st

    sample_images = list_sample_images()
    default_value = sample_images[0] if sample_images else None

    st.set_page_config(page_title="YOLO OBB 样例图地理检测 Demo", layout="wide")
    st.title("YOLO OBB 样例图地理检测 Demo")
    st.write("选择 `sample_100_mix/` 中的一张图片，执行一次 OBB 检测并查看地理结果。")

    left_col, right_col = st.columns([2, 1])
    with left_col:
        selected_image = st.selectbox("样例图片", options=sample_images, index=0 if default_value else None)
    with right_col:
        st.caption("当前选中图片预览")
        preview_image_path = get_preview_image_path(selected_image)
        if preview_image_path is not None:
            st.image(preview_image_path, caption=selected_image, use_container_width=True)
        else:
            st.info("暂无可预览图片。")

    trigger = st.button("开始检测", type="primary")

    if not trigger:
        st.info("请选择一张样例图后开始检测。")
        st.markdown(EMPTY_SUMMARY_HTML, unsafe_allow_html=True)
        st.markdown(EMPTY_DETECTIONS_HTML, unsafe_allow_html=True)
        return

    status, summary_html, plotted_image, details_html = run_demo(selected_image)
    if status == "检测完成。":
        st.success(status)
    elif status == "未检测到目标。":
        st.warning(status)
    else:
        st.error(status)

    st.markdown(summary_html, unsafe_allow_html=True)
    if plotted_image is not None:
        st.image(plotted_image, caption="检测结果图", use_container_width=True)
    st.markdown(details_html, unsafe_allow_html=True)


if __name__ == "__main__":
    build_app()