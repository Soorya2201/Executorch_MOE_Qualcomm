"""Design system: one calm light theme (slate neutrals, one indigo accent, 4/8 px grid, 10-12 px radii).

Every widget is styled through objectName selectors here, so screens stay consistent. Colour names used
elsewhere (GOOD, BAD, MUTED, ACCENT, WARN_*) are kept.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# ---- tokens ----
BG = "#f5f6f8"          # window
SURFACE = "#ffffff"     # cards
SUNKEN = "#f9fafb"      # inputs, inner wells
BORDER = "#e4e7ec"
BORDER_STRONG = "#d0d5dd"
TEXT = "#101828"
TEXT_2 = "#344054"
MUTED = "#667085"
FAINT = "#98a2b3"
ACCENT = "#4f46e5"      # indigo-600
ACCENT_HOVER = "#4338ca"
ACCENT_SOFT = "#eef2ff"
ACCENT_TEXT = "#3730a3"
GOOD = "#079455"
GOOD_SOFT = "#ecfdf3"
WARN = "#dc6803"
WARN_BG = "#fffaeb"
WARN_BORDER = "#fedf89"
BAD = "#d92d20"
BAD_SOFT = "#fef3f2"

QSS = f"""
QWidget {{ font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {TEXT}; color: white; border: none; padding: 6px 8px; border-radius: 6px; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{ padding: 7px 14px; border: 1px solid {BORDER_STRONG}; border-radius: 8px;
    background: {SURFACE}; color: {TEXT_2}; font-weight: 600; }}
QPushButton:hover, QToolButton:hover {{ background: {SUNKEN}; border-color: #98a2b3; }}
QPushButton:pressed, QToolButton:pressed {{ background: #f2f4f7; }}
QPushButton:focus {{ border: 1px solid {ACCENT}; }}
QPushButton:disabled {{ color: {FAINT}; background: {SUNKEN}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#good {{ background: {GOOD}; color: white; border: 1px solid {GOOD}; }}
QPushButton#good:hover {{ background: #067647; }}
QPushButton#primary:disabled, QPushButton#good:disabled {{ background: #c7d0dd; border-color: #c7d0dd; color: white; }}
QPushButton#danger {{ color: {BAD}; }}
QPushButton#danger:hover {{ background: {BAD_SOFT}; border-color: #fda29b; }}
QPushButton#ghost {{ border: 1px solid transparent; background: transparent; color: {TEXT_2}; }}
QPushButton#ghost:hover {{ background: #eaecf0; }}
QPushButton#link {{ border: none; background: transparent; color: {ACCENT}; padding: 2px 4px; text-align: left;
    font-weight: 600; }}
QPushButton#link:hover {{ color: {ACCENT_HOVER}; text-decoration: underline; }}
QPushButton#hero {{ background: {ACCENT}; color: white; border: none; border-radius: 10px; padding: 11px 22px;
    font-size: 10.5pt; }}
QPushButton#hero:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#hero:disabled {{ background: #c7d0dd; }}

/* ---------- inputs ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{ background: {SURFACE}; border: 1px solid {BORDER_STRONG};
    border-radius: 8px; padding: 6px 10px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT_TEXT}; padding: 4px; outline: 0; }}
QCheckBox {{ spacing: 10px; color: {TEXT_2}; font-weight: 600; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {BORDER_STRONG};
    background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox#switch::indicator {{ width: 36px; height: 20px; border-radius: 10px; border: none; background: #d0d5dd; }}
QCheckBox#switch::indicator:checked {{ background: {ACCENT}; }}

/* ---------- lists, trees, text ---------- */
QListWidget, QTreeWidget, QTextBrowser {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 4px; outline: 0; }}
QListWidget::item {{ padding: 9px 10px; border-radius: 8px; margin: 1px 2px; color: {TEXT_2}; }}
QListWidget::item:hover {{ background: {SUNKEN}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QTreeWidget::item {{ padding: 6px 4px; border: none; }}
QTreeWidget::item:hover {{ background: {SUNKEN}; }}
QTreeWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QHeaderView::section {{ background: {SURFACE}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 6px 8px; font-weight: 600; font-size: 9pt; }}
QListWidget#bare, QTextBrowser#bare, QTreeWidget#bare {{ border: none; background: transparent; }}

/* ---------- tabs ---------- */
QTabWidget::pane {{ border: none; background: transparent; top: -1px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 10px 18px; margin-right: 4px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600; font-size: 10pt; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}

/* ---------- app bar navigation ---------- */
QPushButton#nav {{ border: none; background: transparent; color: {MUTED}; padding: 8px 14px; border-radius: 8px;
    font-weight: 600; }}
QPushButton#nav:hover {{ background: {SUNKEN}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QPushButton#nav:focus {{ border: none; }}
QToolButton#more {{ border: none; background: transparent; color: {MUTED}; padding: 6px 10px; border-radius: 8px;
    font-weight: 800; }}
QToolButton#more:hover {{ background: {SUNKEN}; color: {TEXT}; }}
QToolButton#more::menu-indicator {{ image: none; width: 0; }}

/* ---------- surfaces ---------- */
QFrame#appbar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QFrame#sidebar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px; }}
QFrame#card, QFrame#section {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#cardOff {{ background: {SUNKEN}; border: 1px dashed {BORDER_STRONG}; border-radius: 12px; }}
QFrame#tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#banner {{ background: {ACCENT_SOFT}; border: 1px solid #c7d7fe; border-radius: 12px; }}
QFrame#bannerWarn {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 12px; }}
QFrame#question {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 10px; }}
QFrame#chip {{ background: {SUNKEN}; border: 1px solid {BORDER}; border-radius: 13px; }}
QFrame#chipGood {{ background: {GOOD_SOFT}; border: 1px solid #abefc6; border-radius: 13px; }}
QFrame#chipWarn {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 13px; }}
QFrame#divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}

/* ---------- typography ---------- */
QLabel {{ background: transparent; }}
QLabel#brand {{ font-size: 13pt; font-weight: 700; color: {TEXT}; }}
QLabel#display {{ font-size: 20pt; font-weight: 700; color: {TEXT}; }}
QLabel#h1 {{ font-size: 15pt; font-weight: 700; color: {TEXT}; }}
QLabel#h2 {{ font-size: 11pt; font-weight: 700; color: {TEXT}; }}
QLabel#eyebrow {{ font-size: 8.5pt; font-weight: 700; color: {FAINT}; letter-spacing: 1px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#small {{ color: {MUTED}; font-size: 9pt; }}
QLabel#hint {{ background: {SUNKEN}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 10px; color: {TEXT_2}; }}
QLabel#warnText {{ color: {WARN}; font-weight: 600; }}
QLabel#statValue {{ font-size: 20pt; font-weight: 700; color: {TEXT}; }}
QLabel#statLabel {{ font-size: 9pt; font-weight: 600; color: {MUTED}; }}
QLabel#chipText {{ font-size: 9pt; font-weight: 600; color: {TEXT_2}; }}
QLabel#chipGoodText {{ font-size: 9pt; font-weight: 600; color: #067647; }}
QLabel#pillAdd {{ background: {GOOD_SOFT}; color: #067647; border-radius: 9px; padding: 2px 9px; font-weight: 700; }}
QLabel#pillChange {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; border-radius: 9px; padding: 2px 9px;
    font-weight: 700; }}
QLabel#pillArchive {{ background: #f2f4f7; color: {MUTED}; border-radius: 9px; padding: 2px 9px; font-weight: 700; }}

/* ---------- misc ---------- */
QProgressBar {{ background: #eaecf0; border: none; border-radius: 4px; max-height: 8px; min-height: 8px;
    text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSplitter::handle {{ background: transparent; width: 10px; height: 10px; }}
QStatusBar {{ background: {BG}; color: {MUTED}; font-size: 9pt; }}
QStatusBar::item {{ border: none; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #d0d5dd; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #98a2b3; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #d0d5dd; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QMessageBox QLabel {{ color: {TEXT_2}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE),
        (QPalette.AlternateBase, SUNKEN), (QPalette.Text, TEXT), (QPalette.Button, SURFACE),
        (QPalette.ButtonText, TEXT_2), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, TEXT), (QPalette.ToolTipText, "#ffffff"), (QPalette.PlaceholderText, FAINT),
        (QPalette.Link, ACCENT),
    ):
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(FAINT))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(FAINT))
    app.setPalette(pal)
    f = QFont()
    f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"])
    f.setPointSizeF(10)
    f.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(f)
    app.setStyleSheet(QSS)
