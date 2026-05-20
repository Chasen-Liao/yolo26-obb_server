from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    for filename in [".gitignore", "requirements.txt", "README.md"]:
        assert (ROOT / filename).exists(), f"missing {filename}"


def test_requirements_list_runtime_dependencies():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package_name in ["streamlit", "ultralytics", "pillow", "pytest"]:
        assert package_name in requirements


def test_readme_describes_demo_usage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Streamlit" in readme
    assert "sample_100_mix" in readme
    assert "streamlit run app.py" in readme
