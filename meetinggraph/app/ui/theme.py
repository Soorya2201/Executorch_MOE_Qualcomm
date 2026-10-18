"""Design system, after Microsoft's Fluent 2 (Windows 11): neutral greys, one brand blue, Segoe UI Variable,
4 px radius on controls and 8 px on surfaces, soft layered shadows (painted by art.Backdrop), a faint doodle
pattern behind the panels.

Every widget is styled through objectName selectors here, so screens stay consistent. Colour names used
elsewhere (GOOD, BAD, MUTED, ACCENT, WARN_*) are kept.
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# ---- tokens (Fluent 2 light) ----
BG = "#f3f3f3"          # window ("Mica" base)
SURFACE = "#ffffff"     # cards / panels
SUNKEN = "#fafafa"      # inputs, inner wells, hover
SUBTLE = "#f5f5f5"      # pressed / subtle fills
BORDER = "#e5e5e5"
BORDER_STRONG = "#d1d1d1"
TEXT = "#242424"
TEXT_2 = "#424242"
MUTED = "#616161"
FAINT = "#9e9e9e"
ACCENT = "#0f6cbd"      # Fluent brand blue
ACCENT_HOVER = "#115ea3"
ACCENT_PRESSED = "#0c3b5e"
ACCENT_SOFT = "#ebf3fc"
ACCENT_TEXT = "#0f548c"
GOOD = "#107c10"
GOOD_SOFT = "#f1faf1"
WARN = "#bc4b09"
WARN_BG = "#fff9f5"
WARN_BORDER = "#fdcfb4"
BAD = "#c50f1f"
BAD_SOFT = "#fdf3f4"

QSS = f"""
QWidget {{ font-size: 10pt; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px 9px; border-radius: 4px; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{ padding: 6px 14px; border: 1px solid {BORDER_STRONG}; border-bottom-color: #c0c0c0;
    border-radius: 4px; background: {SURFACE}; color: {TEXT}; font-weight: 600; }}
QPushButton:hover, QToolButton:hover {{ background: {SUNKEN}; border-color: #c7c7c7; }}
QPushButton:pressed, QToolButton:pressed {{ background: {SUBTLE}; color: {TEXT_2}; }}
QPushButton:focus {{ border: 1px solid {ACCENT}; }}
QPushButton:disabled {{ color: #bdbdbd; background: {SUBTLE}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; border-color: {ACCENT_PRESSED}; }}
QPushButton#good {{ background: {GOOD}; color: white; border: 1px solid {GOOD}; }}
QPushButton#good:hover {{ background: #0e700e; }}
QPushButton#primary:disabled, QPushButton#good:disabled {{ background: #e0e0e0; border-color: #e0e0e0; color: #bdbdbd; }}
QPushButton#danger {{ color: {BAD}; }}
QPushButton#danger:hover {{ background: {BAD_SOFT}; border-color: #eeacb2; }}
QPushButton#record {{ background: {BAD}; color: white; border: 1px solid {BAD}; }}
QPushButton#record:hover {{ background: #b10e1c; border-color: #b10e1c; }}
QPushButton#ghost {{ border: 1px solid transparent; background: transparent; color: {TEXT_2}; }}
QPushButton#ghost:hover {{ background: {SUBTLE}; }}
QPushButton#link {{ border: none; background: transparent; color: {ACCENT}; padding: 2px 4px; text-align: left;
    font-weight: 600; }}
QPushButton#link:hover {{ color: {ACCENT_HOVER}; text-decoration: underline; }}
QPushButton#hero {{ background: {ACCENT}; color: white; border: none; border-radius: 6px; padding: 10px 22px;
    font-size: 10.5pt; }}
QPushButton#hero:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#hero:disabled {{ background: #e0e0e0; color: #bdbdbd; }}

/* ---------- inputs (Fluent: accent underline on focus) ---------- */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{ background: {SURFACE}; border: 1px solid {BORDER_STRONG};
    border-bottom: 1px solid #8a8a8a; border-radius: 4px; padding: 5px 10px; selection-background-color: {ACCENT}; }}
QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover {{ border-color: #c7c7c7; border-bottom-color: #575757; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {BORDER_STRONG};
    border-bottom: 2px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT_TEXT}; padding: 4px; outline: 0; }}
QCheckBox {{ spacing: 10px; color: {TEXT_2}; font-weight: 600; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 4px; border: 1px solid #8a8a8a;
    background: {SURFACE}; }}
QCheckBox::indicator:hover {{ background: {SUBTLE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox#switch::indicator {{ width: 36px; height: 20px; border-radius: 10px; border: none; background: #d1d1d1; }}
QCheckBox#switch::indicator:checked {{ background: {ACCENT}; }}

/* ---------- lists, trees, text ---------- */
QListWidget, QTreeWidget, QTextBrowser {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 4px; outline: 0; }}
QListWidget::item {{ padding: 9px 10px; border-radius: 4px; margin: 1px 2px; color: {TEXT_2}; }}
QListWidget::item:hover {{ background: {SUBTLE}; }}
QListWidget::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QTreeWidget::item {{ padding: 6px 4px; border: none; }}
QTreeWidget::item:hover {{ background: {SUBTLE}; }}
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

/* ---------- app bar navigation (Fluent "pivot": text + icon, accent underline when selected) ---------- */
QPushButton#nav {{ border: none; border-radius: 4px; background: transparent; color: {MUTED};
    padding: 7px 12px 7px 10px; font-weight: 600; text-align: left; }}
QPushButton#nav:hover {{ background: {SUBTLE}; color: {TEXT}; }}
QPushButton#nav:pressed {{ background: #ebebeb; }}
QPushButton#nav:checked {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QPushButton#nav:focus {{ border: none; }}
QToolButton#more, QToolButton#appTool {{ border: none; background: transparent; color: {MUTED}; padding: 6px 8px;
    border-radius: 4px; font-weight: 600; }}
QToolButton#more:hover, QToolButton#appTool:hover {{ background: {SUBTLE}; color: {TEXT}; }}
QToolButton#appTool:checked {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; }}
QToolButton#more::menu-indicator {{ image: none; width: 0; }}

/* ---------- surfaces (shadows are painted behind them by art.Backdrop) ---------- */
QFrame#appbar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
QFrame#sidebar {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#panel {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#card, QFrame#section {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#cardOff {{ background: {SUNKEN}; border: 1px dashed {BORDER_STRONG}; border-radius: 8px; }}
QFrame#tile {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#banner {{ background: {ACCENT_SOFT}; border: 1px solid #b4d6fa; border-radius: 8px; }}
QFrame#bannerWarn {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 8px; }}
QFrame#question {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 6px; }}
QFrame#chip {{ background: {SUNKEN}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#chipGood {{ background: {GOOD_SOFT}; border: 1px solid #9fd89f; border-radius: 12px; }}
QFrame#chipWarn {{ background: {WARN_BG}; border: 1px solid {WARN_BORDER}; border-radius: 12px; }}
QFrame#divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}

/* ---------- typography (Fluent type ramp) ---------- */
QLabel {{ background: transparent; }}
QLabel#brand {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 12pt; font-weight: 600; color: {TEXT}; }}
QLabel#display {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 20pt; font-weight: 600; }}
QLabel#h1 {{ font-family: 'Segoe UI Variable Display', 'Segoe UI'; font-size: 15pt; font-weight: 600; color: {TEXT}; }}
QLabel#h2 {{ font-size: 11pt; font-weight: 600; color: {TEXT}; }}
QLabel#eyebrow {{ font-size: 8.5pt; font-weight: 600; color: {MUTED}; letter-spacing: 0.6px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#small {{ color: {MUTED}; font-size: 9pt; }}
QLabel#hint {{ background: {SUNKEN}; border: 1px solid {BORDER}; border-radius: 6px; padding: 8px 10px; color: {TEXT_2}; }}
QLabel#warnText {{ color: {WARN}; font-weight: 600; }}
QLabel#statValue {{ font-size: 20pt; font-weight: 600; color: {TEXT}; }}
QLabel#statLabel {{ font-size: 9pt; font-weight: 600; color: {MUTED}; }}
QLabel#chipText {{ font-size: 9pt; font-weight: 600; color: {TEXT_2}; }}
QLabel#chipGoodText {{ font-size: 9pt; font-weight: 600; color: {GOOD}; }}
QLabel#pillAdd {{ background: {GOOD_SOFT}; color: {GOOD}; border-radius: 4px; padding: 2px 8px; font-weight: 700; }}
QLabel#pillChange {{ background: {ACCENT_SOFT}; color: {ACCENT_TEXT}; border-radius: 4px; padding: 2px 8px;
    font-weight: 700; }}
QLabel#pillArchive {{ background: {SUBTLE}; color: {MUTED}; border-radius: 4px; padding: 2px 8px; font-weight: 700; }}

/* ---------- misc ---------- */
QProgressBar {{ background: #e0e0e0; border: none; border-radius: 2px; max-height: 4px; min-height: 4px;
    text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QSplitter::handle {{ background: transparent; width: 12px; height: 12px; }}
QStatusBar {{ background: transparent; color: {MUTED}; font-size: 9pt; }}
QStatusBar::item {{ border: none; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #c2c2c2; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #8a8a8a; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #c2c2c2; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 7px 22px 7px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {SUBTLE}; color: {TEXT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
QMessageBox QLabel {{ color: {TEXT_2}; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, SURFACE),
        (QPalette.AlternateBase, SUNKEN), (QPalette.Text, TEXT), (QPalette.Button, SURFACE),
        (QPalette.ButtonText, TEXT), (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
        (QPalette.ToolTipBase, SURFACE), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, FAINT),
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
