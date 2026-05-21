from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_includes_api_dependencies():
    content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package_name in ["fastapi", "uvicorn", "python-multipart", "httpx", "pytest"]:
        assert package_name in content


def test_api_fixture_module_exists():
    fixture_file = ROOT / "tests" / "conftest.py"
    assert fixture_file.exists(), "missing tests/conftest.py"


def test_api_readme_exists_and_mentions_detect_endpoint():
    readme_path = ROOT / "README_API.md"
    assert readme_path.exists(), "missing README_API.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "obb_geo_api_server:app" in content
    assert "/v1/detect" in content
