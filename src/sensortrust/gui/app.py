"""Application entry point of the graphical interface."""

from __future__ import annotations

import multiprocessing
import os
import sys

STYLE = """
QMainWindow, QWidget { background: #f9f9f7; color: #0b0b0b; font-size: 10pt; }
QToolBar { background: #fcfcfb; border-bottom: 1px solid #e1e0d9; spacing: 6px; padding: 4px; }
QLabel#title { font-size: 12pt; padding-left: 6px; }
QTabWidget::pane { border: 1px solid #e1e0d9; background: #fcfcfb; }
QTabBar::tab { background: #f0efec; border: 1px solid #e1e0d9; padding: 6px 12px; margin-right: 1px; }
QTabBar::tab:selected { background: #fcfcfb; border-bottom: 2px solid #2a78d6; }
QGroupBox { border: 1px solid #e1e0d9; border-radius: 4px; margin-top: 12px; padding-top: 8px; background: #fcfcfb; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #0d366b; font-weight: bold; }
QPushButton { background: #fcfcfb; border: 1px solid #c3c2b7; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #eef4fc; }
QPushButton:disabled { color: #898781; }
QPushButton#primary { background: #2a78d6; color: white; border: 1px solid #256abf; font-weight: bold; }
QPushButton#primary:hover { background: #256abf; }
QPushButton#primary:disabled { background: #9ec5f4; }
QPushButton#lang { font-weight: bold; min-width: 70px; }
QTableWidget, QListWidget, QPlainTextEdit, QTextBrowser, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: white; border: 1px solid #c3c2b7; border-radius: 3px; }
QHeaderView::section { background: #f0efec; border: none; border-right: 1px solid #e1e0d9; padding: 4px; }
QTableWidget { alternate-background-color: #f6f6f3; gridline-color: #e1e0d9; }
QStatusBar { background: #fcfcfb; border-top: 1px solid #e1e0d9; }
"""


def _log_file():
    """Session log of the interface (``<user data dir>/sensortrust_gui.log``), or None if not writable."""
    import logging

    from ..utils.logging import LOGGER_NAME
    from ..utils.paths import user_data_dir
    try:
        d = user_data_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "sensortrust_gui.log"
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logging.getLogger(LOGGER_NAME).addHandler(h)
        return path
    except OSError:
        return None


def install_exception_guard(parent_getter=lambda: None, log_path=None):
    """Unexpected exceptions are logged and reported in a dialog; the application never closes because of them."""
    import logging
    import traceback

    log = logging.getLogger("sensortrust.gui")

    def hook(etype, value, tb):
        if issubclass(etype, KeyboardInterrupt):
            return
        text = "".join(traceback.format_exception(etype, value, tb))
        log.error("unexpected exception\n%s", text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            from ..i18n import tr
            from ..i18n_errors import translate_error
            if QApplication.instance() is not None:
                box = QMessageBox(QMessageBox.Warning, tr("Unexpected error"),
                                  translate_error(f"{etype.__name__}: {value}") + "\n\n" + tr(
                                      "The operation could not be completed. The application remains open; "
                                      "details were written to the log file:") + f"\n{log_path or '-'}",
                                  parent=parent_getter())
                box.setDetailedText(text)
                box.exec()
        except Exception:  # never let the guard itself fail
            pass

    sys.excepthook = hook
    return hook


def main(lang: str | None = None) -> None:
    multiprocessing.freeze_support()
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    from PySide6.QtWidgets import QApplication

    from ..utils.logging import configure_logging
    from . import strings  # noqa: F401  (registers the Spanish interface texts)
    from .main_window import MainWindow

    configure_logging()
    log_path = _log_file()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setApplicationName("SensorTrust Fusion")
    lang = lang or os.environ.get("SENSORTRUST_LANG", "en")
    win = MainWindow(lang=lang)
    install_exception_guard(lambda: win, log_path)
    win.show()
    _close_splash()
    smoke = os.environ.get("SENSORTRUST_SMOKE_TEST")
    if smoke:  # used by the build pipeline: open the real window, write a marker file and exit
        from PySide6.QtCore import QTimer

        def _done():
            from pathlib import Path
            Path(smoke).write_text(f"{win.windowTitle()}\n{win.tabs.count()} tabs\n", encoding="utf-8")
            app.quit()
        QTimer.singleShot(3000, _done)
    sys.exit(app.exec())


def _close_splash():
    """Close the start-up splash screen of the Windows executable (PyInstaller), if any."""
    try:
        import pyi_splash  # type: ignore  # only present in the frozen executable
        pyi_splash.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
