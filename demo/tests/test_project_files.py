from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_ROOT.parent


def test_required_project_files_exist():
    for filename in ["requirements.txt"]:
        assert (REPO_ROOT / filename).exists(), f"missing {filename}"
    for filename in ["README.md", "app.py", "data_loader.py", "geo_mapper.py", "inference.py"]:
        assert (DEMO_ROOT / filename).exists(), f"missing demo/{filename}"


def test_requirements_list_runtime_dependencies():
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package_name in ["streamlit", "ultralytics", "pillow", "pytest"]:
        assert package_name in requirements


def test_readme_describes_demo_usage():
    readme = (DEMO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Streamlit" in readme
    assert "sample_100_mix" in readme
    assert "streamlit run demo/app.py" in readme