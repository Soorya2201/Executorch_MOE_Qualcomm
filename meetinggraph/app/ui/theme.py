"""One consistent light look, independent of the Windows dark/light setting."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#1f6feb"
GOOD = "#1a7f37"
WARN_BG = "#fff8c5"
WARN_BORDER = "#d4a72c"
BAD = "#cf222e"
MUTED = "#57606a"

QSS = f"""
QWidget {{ font-size: 10.5pt; }}
QPushButton {{ padding: 6px 14px; border: 1px solid #c9d1d9; border-radius: 6px; background: #f6f8fa; }}
QPushButton:hover {{ background: #eef1f4; }}
QPushButton:disabled {{ color: #9aa4ae; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; font-weight: 600; }}
QPushButton#primary:hover {{ background: #1a5fd0; }}
QPushButton#primary:disabled, QPushButton#good:disabled {{ background: #c9d1d9; border-color: #c9d1d9; color: #f6f8fa; }}
QTextBrowser {{ background: white; }}
QPushButton#good {{ background: {GOOD}; color: white; border: 1px solid {GOOD}; font-weight: 600; }}
QPushButton#danger {{ color: {BAD}; }}
QPushButton#link {{ border: none; background: transparent; color: {ACCENT}; padding: 2px 4px; text-align: left; }}
QFrame#banner {{ background: #ddf4ff; border: 1px solid #9cd1f7; border-radius: 8px; }}
QFrame#bannerWarn {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 8px; }}
QFrame#card {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; }}
QFrame#cardOff {{ background: #f6f8fa; border: 1px dashed #d0d7de; border-radius: 8px; }}
QFrame#question {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 6px; }}
QFrame#section {{ background: white; border: 1px solid #d0d7de; border-radius: 8px; }}
QLabel#h1 {{ font-size: 15pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 12pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#pillAdd {{ background: #dafbe1; color: {GOOD}; border-radius: 9px; padding: 2px 8px; font-weight: 600; }}
QLabel#pillChange {{ background: #ddf4ff; color: #0969da; border-radius: 9px; padding: 2px 8px; font-weight: 600; }}
QLabel#pillArchive {{ background: #eaeef2; color: {MUTED}; border-radius: 9px; padding: 2px 8px; font-weight: 600; }}
QTabBar::tab {{ padding: 8px 16px; }}
QListWidget::item {{ padding: 6px; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.Window, "#f6f8fa"), (QPalette.WindowText, "#1f2328"), (QPalette.Base, "#ffffff"),
        (QPalette.AlternateBase, "#f6f8fa"), (QPalette.Text, "#1f2328"), (QPalette.Button, "#f6f8fa"),
        (QPalette.ButtonText, "#1f2328"), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, "#ffffff"), (QPalette.ToolTipText, "#1f2328"), (QPalette.PlaceholderText, "#8c959f"),
        (QPalette.Link, ACCENT),
    ):
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor("#8c959f"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#8c959f"))
    app.setPalette(pal)
    f = QFont("Segoe UI")
    f.setPointSizeF(10.5)
    app.setFont(f)
    app.setStyleSheet(QSS)
