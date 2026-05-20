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
