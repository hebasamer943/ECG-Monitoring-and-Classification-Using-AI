from pathlib import Path
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication, project_root: Path) -> None:
    qss_path = project_root / "desktop_app" / "ui" / "theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
