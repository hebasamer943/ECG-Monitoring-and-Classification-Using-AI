import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from desktop_app.theme import apply_theme
from desktop_app.ui.main_window import MainWindow


def run():
    app = QApplication(sys.argv)
    project_root = Path(__file__).resolve().parents[1]
    apply_theme(app, project_root)
    win = MainWindow(project_root=project_root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()