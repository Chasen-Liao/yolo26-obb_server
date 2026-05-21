from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

# Ensure project root modules (obb_geo_service, obb_geo_api_server) are importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
