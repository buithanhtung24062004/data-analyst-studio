import ast
from pathlib import Path


def test_app_parses():
    app = Path(__file__).resolve().parents[1] / "app.py"
    ast.parse(app.read_text(encoding="utf-8"))


def test_required_files_exist():
    root = Path(__file__).resolve().parents[1]
    for name in ["app.py", "README.md", "requirements.txt", ".gitignore"]:
        assert (root / name).exists()
