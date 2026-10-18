"""Applies the design tokens to the whole app: Qt palette + one stylesheet, light or dark.

Follows the Windows app-mode setting (Settings > Personalization > Colors) and switches live when it changes.
Set the environment variable MEETINGGRAPH_THEME=light|dark to force one (useful for screenshots).
Everything here is generated from tokens.py; no colors are invented in this file.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

from . import tokens as tk


class _KeyboardFocusRings(QObject):
    """Focus rings only when focus arrives from the keyboard (Tab, shortcuts), not from a mouse click,
    the way Windows and macOS both behave. Sets a 'kbd' property the stylesheet keys on."""
    KBD = (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.FocusIn and obj.isWidgetType():
            kbd = ev.reason() in self.KBD
            if obj.property("kbd") != kbd:
                obj.setProperty("kbd", kbd)
                obj.style().unpolish(obj)
                obj.style().polish(obj)
        return False


_focus_rings = _KeyboardFocusRings()


class _Notifier(QObject):
    changed = Signal(str)          # "light" | "dark"; views re-render HTML content on this


signals = _Notifier()

# Extra control shades (derived per mode, kept with the theme because they're only used by the stylesheet).
_CONTROL = {
    "light": {"control": "#E8E8ED", "control_hover": "#DFDFE4", "control_press": "#D4D4DA"},
    "dark": {"control": "#3A3A3C", "control_hover": "#444446", "control_press": "#505052"},
}

# Back-compat names some modules still import; refreshed on every apply().
ACCENT = tk.ACCENT
GOOD = BAD = MUTED = WARN_BG = WARN_BORDER = ""


def system_mode() -> str:
    forced = os.environ.get("MEETINGGRAPH_THEME", "").lower()
    if forced in ("light", "dark"):
        return forced
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"
    except Exception:
        return "light"


def c(name: str) -> str:
    """Shortcut for the current color token (also covers the control shades)."""
    p = _CONTROL[tk.mode()]
    return p[name] if name in p else tk.c(name)


def qcolor(name: str) -> QColor:
    v = c(name)
    if v.startswith("rgba("):
        r, g, b, a = [x.strip() for x in v[5:-1].split(",")]
        col = QColor(int(r), int(g), int(b))
        col.setAlphaF(float(a))
        return col
    return QColor(v)


def font(kind: str = "body") -> QFont:
    px, weight = tk.TYPE[kind]
    f = QFont()
    f.setFamilies([tk.FONT_FAMILY, tk.FONT_FALLBACK])
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    return f


def stylesheet() -> str:
    t = dict(tk.palette(), **_CONTROL[tk.mode()])
    R, RC = tk.RADIUS["control"], tk.RADIUS["card"]
    body, meta = tk.TYPE["body"][0], tk.TYPE["meta"][0]
    title, section = tk.TYPE["title"][0], tk.TYPE["section"][0]
    return f"""
QWidget {{ color: {t['text']}; font-size: {body}px; }}
QMainWindow, QDialog {{ background: {t['bg']}; }}
QToolTip {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['separator_solid']};
            border-radius: {R}px; padding: 4px 8px; font-size: {meta}px; }}

/* ----- buttons: quiet by default, one accent ----- */
QPushButton, QToolButton {{ background: {t['control']}; color: {t['text']}; border: 2px solid transparent;
              border-radius: {R}px; padding: 4px 12px; min-height: 22px; }}
QPushButton:hover, QToolButton:hover {{ background: {t['control_hover']}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t['control_press']}; }}
QPushButton[kbd="true"]:focus, QToolButton[kbd="true"]:focus {{ border: 2px solid {t['accent']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t['tertiary']}; }}
QPushButton#primary, QPushButton#good {{ background: {t['accent']}; color: {t['on_accent']}; font-weight: 600; }}
QPushButton#primary:hover, QPushButton#good:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary:pressed, QPushButton#good:pressed {{ background: {t['accent_press']}; }}
QPushButton#primary[kbd="true"]:focus, QPushButton#good[kbd="true"]:focus {{ border: 2px solid {t['text']}; }}
QPushButton#primary:disabled, QPushButton#good:disabled {{ background: {t['control']}; color: {t['tertiary']}; }}
QPushButton#danger {{ background: transparent; color: {t['danger']}; }}
QPushButton#danger:hover {{ background: {t['hover']}; }}
QPushButton#record {{ background: {t['danger_soft']}; color: {t['danger']}; font-weight: 600; }}
QPushButton#record:hover {{ background: {t['danger_soft']}; border: 2px solid {t['danger_soft']}; }}
QPushButton#good {{ background: {t['success_strong']}; color: #FFFFFF; }}
QPushButton#good:hover, QPushButton#good:pressed {{ background: {t['success_hover']}; }}
QPushButton#link, QPushButton#quiet {{ background: transparent; color: {t['accent_text']}; padding: 2px 4px;
              text-align: left; }}
QPushButton#quiet {{ color: {t['secondary']}; }}
QPushButton#link:hover, QPushButton#quiet:hover {{ background: {t['hover']}; }}
QPushButton#chip {{ background: {t['surface2']}; color: {t['text']}; border-radius: 12px; padding: 2px 10px;
              min-height: 18px; font-size: {meta}px; }}
QPushButton#chip:hover {{ background: {t['control_hover']}; }}

/* ----- inputs ----- */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{ background: {t['input']}; color: {t['text']};
              border: 1px solid {t['separator_solid']}; border-radius: {R}px; padding: 5px 8px;
              selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{ border: 1px solid {t['accent']}; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; border: 1px solid {t['separator_solid']};
              selection-background-color: {t['accent_soft']}; selection-color: {t['text']}; }}
QTextBrowser {{ background: {t['surface']}; border: none; border-radius: {RC}px; padding: 8px; }}
QCheckBox, QRadioButton {{ spacing: 8px; }}

/* ----- lists: rounded soft selection, light hover ----- */
QListWidget, QTreeWidget, QListView {{ background: transparent; border: none; outline: 0; }}
QListWidget::item, QListView::item {{ padding: 7px 10px; margin: 1px 4px; border-radius: {R}px; }}
QListWidget::item:hover, QListView::item:hover {{ background: {t['hover']}; }}
QListWidget::item:selected, QListView::item:selected {{ background: {t['accent_soft']}; color: {t['accent_text']}; }}

/* ----- tabs (replaced by a segmented control in phase 2) ----- */
QTabWidget::pane {{ border: none; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: transparent; color: {t['secondary']}; padding: 6px 16px; margin: 2px;
              border-radius: {R}px; }}
QTabBar::tab:selected {{ background: {t['surface']}; color: {t['accent_text']}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {t['text']}; background: {t['hover']}; }}

/* ----- structure: whitespace + hairlines, no boxes around everything ----- */
QSplitter::handle {{ background: {t['separator_solid']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {t['separator_solid']}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {t['tertiary']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QStatusBar {{ background: {t['bg']}; color: {t['secondary']}; font-size: {meta}px;
              border-top: 1px solid {t['separator']}; }}
QStatusBar QLabel {{ color: {t['secondary']}; font-size: {meta}px; }}
QMenuBar {{ background: {t['bg']}; }}
QMenu {{ background: {t['surface']}; border: 1px solid {t['separator_solid']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: {R}px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
QMenu::separator {{ height: 1px; background: {t['separator_solid']}; margin: 4px 8px; }}
QProgressBar {{ background: {t['control']}; border: none; border-radius: 3px; max-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}

/* ----- surfaces ----- */
QFrame#card, QFrame#section {{ background: {t['surface']}; border: none; border-radius: {RC}px; }}
QFrame#cardOff {{ background: transparent; border: 1px dashed {t['separator_solid']}; border-radius: {RC}px; }}
QFrame#question {{ background: {t['warning_soft']}; border: none; border-radius: {R}px; }}
QFrame#banner, QFrame#bannerWarn {{ background: {t['surface']}; border: none; border-radius: {RC}px; }}
QFrame#hairline {{ background: {t['separator']}; max-height: 1px; min-height: 1px; border: none; }}

/* ----- type ----- */
QLabel#h1, QLabel#title {{ font-size: {title}px; font-weight: 600; }}
QLabel#h2, QLabel#section {{ font-size: {section}px; font-weight: 600; }}
QLabel#muted, QLabel#meta {{ color: {t['secondary']}; font-size: {meta}px; }}
QLabel#caption {{ color: {t['secondary']}; font-size: {meta}px; font-weight: 600; }}
QLabel#warn {{ color: {t['warning']}; }}
QLabel#note {{ background: {t['surface2']}; color: {t['text']}; border-radius: {R}px; padding: 8px; }}
QLabel#pillAdd, QLabel#pillChange, QLabel#pillArchive {{ border-radius: 9px; padding: 2px 8px;
              font-size: {meta}px; font-weight: 600; }}
QLabel#pillAdd {{ background: {t['accent_soft']}; color: {t['success']}; }}
QLabel#pillChange {{ background: {t['accent_soft']}; color: {t['accent_text']}; }}
QLabel#pillArchive {{ background: {t['surface2']}; color: {t['secondary']}; }}
"""


def html_css() -> str:
    """<style> for rich-text views (QTextBrowser), from the same tokens."""
    t = tk.palette()
    return f"""<style>
body {{ font-family: '{tk.FONT_FAMILY}', '{tk.FONT_FALLBACK}'; font-size: {tk.TYPE['body'][0]}px; color: {t['text']};
       line-height: {int(tk.LINE_HEIGHT * 100)}%; }}
h1 {{ font-size: {tk.TYPE['title'][0]}px; font-weight: 600; margin: 0 0 4px 0; }}
h3 {{ font-size: {tk.TYPE['section'][0]}px; font-weight: 600; margin: 20px 0 6px 0; }}
.muted {{ color: {t['secondary']}; font-size: {tk.TYPE['meta'][0]}px; }}
.warn {{ color: {t['warning']}; }}
.bad {{ color: {t['danger']}; }}
.pill {{ color: white; font-size: {tk.TYPE['meta'][0]}px; font-weight: 600; }}
a {{ color: {t['accent_text']}; text-decoration: none; }}
ul {{ margin-top: 2px; }}
</style>"""


def apply(app: QApplication, mode: str | None = None) -> str:
    """(Re)apply palette + stylesheet. Returns the mode used."""
    global GOOD, BAD, MUTED, WARN_BG, WARN_BORDER
    tk.set_mode(mode or system_mode())
    t = tk.palette()
    app.setStyle("Fusion")
    pal = QPalette()
    for role, name in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"), (QPalette.Base, "surface"),
                       (QPalette.AlternateBase, "surface2"), (QPalette.Text, "text"),
                       (QPalette.Button, "control"), (QPalette.ButtonText, "text"),
                       (QPalette.Highlight, "accent"), (QPalette.HighlightedText, "on_accent"),
                       (QPalette.ToolTipBase, "surface"), (QPalette.ToolTipText, "text"),
                       (QPalette.PlaceholderText, "tertiary"), (QPalette.Link, "accent_text"),
                       (QPalette.LinkVisited, "accent_text"), (QPalette.BrightText, "on_accent"),
                       (QPalette.Mid, "separator_solid"), (QPalette.Dark, "tertiary"),
                       (QPalette.Light, "surface"), (QPalette.Midlight, "surface2")):
        pal.setColor(role, qcolor(name))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        pal.setColor(QPalette.Disabled, role, qcolor("tertiary"))
    app.setPalette(pal)
    app.setFont(font("body"))
    app.setStyleSheet(stylesheet())
    app.removeEventFilter(_focus_rings)
    app.installEventFilter(_focus_rings)
    GOOD, BAD, MUTED = t["success"], t["danger"], t["secondary"]
    WARN_BG, WARN_BORDER = t["warning_soft"], t["warning"]
    return tk.mode()


def follow_system(app: QApplication) -> None:
    """Re-theme live when Windows switches between light and dark."""
    def on_change(*_):
        signals.changed.emit(apply(app))
    try:
        QGuiApplication.styleHints().colorSchemeChanged.connect(on_change)
    except Exception:
        pass
