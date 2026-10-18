"""Artwork, all vector (SVG): the doodle wallpaper, the app logo, Fluent-style line icons, and the Backdrop
that paints the wallpaper plus soft layered shadows behind the white panels. Also the Windows 11 title bar
tint, so the window frame matches the app bar.
"""
from __future__ import annotations

import sys

from PySide6.QtCore import QByteArray, QEvent, QObject, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPixmap, QTransform
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from . import theme

# ---------------------------------------------------------------------------------------------------------
# Doodle wallpaper: hand-drawn line art about meetings, notes, voice and on-device AI. Generated once into a
# seamless 594x594 tile: 81 doodles at full size on a staggered 66 px grid (each with its own small tilt and nudge, never
# next to an identical one) plus tiny sparkles in the gaps. Tiled very faintly behind the panels.
# ---------------------------------------------------------------------------------------------------------
_S = 'fill="none" stroke="#3b4a6b" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'

# name: (svg markup in its own coordinates, centre x, centre y). Every doodle fits in roughly 48x48.
GLYPHS = {
    "bubble": ('<path d="M4 4h44a8 8 0 0 1 8 8v20a8 8 0 0 1-8 8H22l-10 9v-9H4a8 8 0 0 1-8-8V12a8 8 0 0 1 8-8z"/>'
               '<circle cx="14" cy="22" r="1.6"/><circle cx="26" cy="22" r="1.6"/><circle cx="38" cy="22" r="1.6"/>',
               26, 26),
    "mic": ('<rect x="8" y="0" width="16" height="28" rx="8"/><path d="M2 20a14 14 0 0 0 28 0M16 34v8M8 42h16"/>',
            16, 21),
    "note": ('<path d="M0 0h40v28l-12 12H0z M40 28H28v12"/><path d="M8 12h22M8 20h16"/>', 20, 20),
    "wave": ('<path d="M0 14v8M8 6v24M16 0v36M24 9v18M32 3v30M40 12v12M48 16v4"/>', 24, 18),
    "bulb": ('<path d="M14 0a14 14 0 0 0-8 25c2 2 3 4 3 7h10c0-3 1-5 3-7a14 14 0 0 0-8-25z M9 37h10M10 42h8"/>'
             '<path d="M-8 6l-5-3M36 6l5-3M14 -8v-4"/>', 14, 15),
    "calendar": ('<rect x="0" y="6" width="44" height="38" rx="5"/><path d="M0 16h44M12 0v10M32 0v10"/>'
                 '<circle cx="12" cy="26" r="1.4"/><circle cx="22" cy="26" r="1.4"/><circle cx="32" cy="26" r="1.4"/>'
                 '<circle cx="12" cy="35" r="1.4"/><circle cx="22" cy="35" r="1.4"/>', 22, 22),
    "graph": ('<circle cx="6" cy="30" r="6"/><circle cx="38" cy="8" r="6"/><circle cx="50" cy="38" r="6"/>'
              '<path d="M11 27l22-15M42 13l6 19M12 31l32 6"/>', 28, 23),
    "clock": ('<circle cx="20" cy="20" r="19"/><path d="M20 9v11l8 5"/>', 20, 20),
    "checklist": ('<rect x="0" y="0" width="10" height="10" rx="2"/><path d="M2.5 5l2 2 3.5-4M18 5h26"/>'
                  '<rect x="0" y="18" width="10" height="10" rx="2"/><path d="M2.5 23l2 2 3.5-4M18 23h20"/>'
                  '<rect x="0" y="36" width="10" height="10" rx="2"/><path d="M18 41h24"/>', 22, 23),
    "coffee": ('<path d="M0 14h30v14a12 12 0 0 1-12 12h-6A12 12 0 0 1 0 28z M30 18h4a6 6 0 0 1 0 12h-4"/>'
               '<path d="M8 8c-3-3 3-5 0-8M16 8c-3-3 3-5 0-8M24 8c-3-3 3-5 0-8"/>', 18, 20),
    "pencil": ('<path d="M0 6h44l10 4-10 4H0z M44 6v8M8 6v8"/>', 27, 10),
    "headphones": ('<path d="M2 30v-8a20 20 0 0 1 40 0v8"/><rect x="0" y="28" width="10" height="16" rx="4"/>'
                   '<rect x="34" y="28" width="10" height="16" rx="4"/>', 22, 23),
    "chip": ('<rect x="6" y="6" width="34" height="34" rx="5"/><rect x="15" y="15" width="16" height="16" rx="2"/>'
             '<path d="M14 0v6M23 0v6M32 0v6M14 40v6M23 40v6M32 40v6M0 14h6M0 23h6M0 32h6M40 14h6M40 23h6M40 32h6"/>',
             23, 23),
    "laptop": ('<rect x="6" y="0" width="44" height="28" rx="3"/><path d="M0 34h56l-4 5H4z"/>', 28, 20),
    "envelope": ('<rect x="0" y="0" width="42" height="30" rx="4"/><path d="M2 3l19 14 19-14"/>', 21, 15),
    "bars": ('<path d="M0 0v44h50"/><path d="M10 36v-12M22 36v-22M34 36v-16M46 36v-28"/>', 25, 22),
    "magnifier": ('<circle cx="16" cy="16" r="14"/><path d="M26 26l14 14"/>', 20, 20),
    "lock": ('<rect x="0" y="18" width="34" height="26" rx="5"/><path d="M7 18v-6a10 10 0 0 1 20 0v6"/>'
             '<circle cx="17" cy="30" r="2.2"/><path d="M17 32v5"/>', 17, 28),
    "paperclip": ('<path d="M10 40V10a6 6 0 0 1 12 0v28a10 10 0 0 1-20 0V12"/>', 12, 26),
    "chats": ('<rect x="0" y="0" width="34" height="22" rx="7"/><path d="M8 22l-2 8 9-8"/>'
              '<rect x="22" y="14" width="30" height="20" rx="7"/><path d="M44 34l3 7-9-7"/>', 26, 20),
    "pin": ('<path d="M8 0h16l-3 12 7 8H4l7-8z M16 20v16"/>', 16, 18),
    "heart": ('<path d="M16 30S0 20 0 9a8 8 0 0 1 16-2 8 8 0 0 1 16 2c0 11-16 21-16 21z"/>', 16, 15),
    "folder": ('<path d="M0 6a4 4 0 0 1 4-4h12l5 6h21a4 4 0 0 1 4 4v20a4 4 0 0 1-4 4H4a4 4 0 0 1-4-4z"/>', 23, 19),
    "arrow": ('<path d="M0 26C10 6 30 2 44 14"/><path d="M36 8l8 6-9 4"/>', 22, 15),
    "star": ('<path d="M20 2l5 11 12 1-9 8 3 12-11-6-11 6 3-12-9-8 12-1z"/>', 20, 19),
    "music": ('<path d="M14 30V6l18-4v22"/><circle cx="10" cy="30" r="4"/><circle cx="28" cy="25" r="4"/>', 20, 17),
    "camera": ('<rect x="0" y="6" width="32" height="24" rx="5"/><path d="M32 14l12-6v20l-12-6"/>', 22, 18),
    "book": ('<path d="M2 4h16a6 6 0 0 1 6 6v28a5 5 0 0 0-5-5H2z M46 4H30a6 6 0 0 0-6 6v28a5 5 0 0 1 5-5h17z"/>',
             24, 20),
    "smile": ('<circle cx="18" cy="18" r="16"/><circle cx="12" cy="14" r="1.6"/><circle cx="24" cy="14" r="1.6"/>'
              '<path d="M11 22a8 8 0 0 0 14 0"/>', 18, 18),
    "bell": ('<path d="M18 4a10 10 0 0 0-10 10v8l-4 6h28l-4-6v-8A10 10 0 0 0 18 4z M14 32a4 4 0 0 0 8 0"/>', 18, 18),
    "offline": ('<path d="M4 14a24 24 0 0 1 32 0M10 21a15 15 0 0 1 20 0M16 28a6 6 0 0 1 8 0M4 4l32 32"/>'
                '<circle cx="20" cy="34" r="1.6"/>', 20, 19),
    "target": ('<circle cx="18" cy="18" r="16"/><circle cx="18" cy="18" r="10"/><circle cx="18" cy="18" r="4"/>',
               18, 18),
    "hash": ('<path d="M8 14h28M6 26h28M16 4l-4 32M28 4l-4 32"/>', 20, 20),
    "sticky2": ('<rect x="0" y="0" width="36" height="36" rx="3"/><path d="M7 10h22M7 18h22M7 26h12"/>', 18, 18),
    "gear": ('<circle cx="18" cy="18" r="6"/><path d="M18 2v6M18 28v6M2 18h6M28 18h6M6.7 6.7l4.2 4.2M25.1 25.1l4.2 4.2'
             'M6.7 29.3l4.2-4.2M25.1 10.9l4.2-4.2"/>', 18, 18),
    "brush": ('<path d="M30 2l6 6-16 16-6-6z M14 18c-6 0-9 4-9 9 0 3-2 5-3 6 6 1 13 0 15-6 1-2 1-5-3-9z"/>', 19, 18),
}
_FILLERS = ('<path d="M0 -6l1.8 4.2 4.2 1.8-4.2 1.8L0 6l-1.8-4.2L-6 0l4.2-1.8z"/>',     # sparkle
            '<path d="M0 -4v8M-4 0h8"/>',                                                # plus
            '<circle cx="0" cy="0" r="1.8"/>',                                           # dot
            '<circle cx="0" cy="0" r="3.2"/>')                                           # ring


def build_doodle_svg(tile: int = 594, cell: int = 66, seed: int = 11) -> str:
    """Staggered grid of doodles, each with its own tilt/nudge, plus fillers in the gaps. Seamless: anything
    that crosses an edge is drawn again on the opposite side."""
    import random
    rnd = random.Random(seed)
    names = list(GLYPHS)
    n = tile // cell
    grid: dict[tuple[int, int], str] = {}
    parts = []

    def put(markup: str, x: float, y: float, rot: float, scale: float = 1.0, cx: float = 0, cy: float = 0):
        for dx in (-tile, 0, tile):
            for dy in (-tile, 0, tile):
                parts.append(f'<g transform="translate({x + dx:.1f} {y + dy:.1f}) rotate({rot:.1f}) '
                             f'scale({scale:.2f}) translate({-cx} {-cy})">{markup}</g>')

    bag: list[str] = []
    for row in range(n):
        for col in range(n):
            if not bag:
                bag = names[:]
                rnd.shuffle(bag)
            near = {grid.get((row, col - 1)), grid.get((row - 1, col)), grid.get((row - 1, col + 1)),
                    grid.get((row, (col + 1) % n) if col == n - 1 else None)}
            pick = next((g for g in bag if g not in near), bag[0])
            bag.remove(pick)
            grid[(row, col)] = pick
            markup, cx, cy = GLYPHS[pick]
            x = col * cell + cell / 2 + (cell / 2 if row % 2 else 0) + rnd.uniform(-7, 7)
            y = row * cell + cell / 2 + rnd.uniform(-7, 7)
            put(markup, x % tile, y, rnd.uniform(-24, 24), rnd.uniform(0.9, 1.0), cx, cy)
            # a filler in the gap to the lower right
            put(rnd.choice(_FILLERS), (x + cell / 2 + rnd.uniform(-5, 5)) % tile,
                (y + cell / 2 + rnd.uniform(-5, 5)) % tile, rnd.uniform(0, 45))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{tile}" height="{tile}" viewBox="0 0 {tile} {tile}">'
            f'<g {_S}>' + "".join(parts) + "</g></svg>")


DOODLE_TILE = 594
DOODLE_SVG = build_doodle_svg(DOODLE_TILE)

# ---- app logo: a rounded tile (brand blue -> violet) with a speech bubble holding a small graph ----
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
  <stop offset="0" stop-color="#2b88d8"/><stop offset="0.55" stop-color="#0f6cbd"/><stop offset="1" stop-color="#5b5fc7"/>
</linearGradient></defs>
<rect x="2" y="2" width="60" height="60" rx="14" fill="url(#g)"/>
<rect x="2" y="2" width="60" height="30" rx="14" fill="#ffffff" opacity="0.08"/>
<path d="M14 16h36a6 6 0 0 1 6 6v18a6 6 0 0 1-6 6H30l-9 8v-8h-7a6 6 0 0 1-6-6V22a6 6 0 0 1 6-6z" fill="#ffffff"/>
<g stroke="#0f6cbd" stroke-width="2.6" stroke-linecap="round">
  <path d="M22 36l10-8M32 28l10 6"/>
</g>
<circle cx="22" cy="36" r="4" fill="#0f6cbd"/><circle cx="32" cy="27" r="4" fill="#5b5fc7"/>
<circle cx="43" cy="34" r="4" fill="#2b88d8"/>
</svg>"""

# ---- Fluent-style line icons (20x20, 1.5 stroke); "currentColor" is replaced per state ----
_I = 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"'
ICONS = {
    "review": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}>'
              '<rect x="4" y="3.5" width="12" height="14" rx="2"/><path d="M7.5 3.5V2.8a.8.8 0 0 1 .8-.8h3.4a.8.8 0 0 1 '
              '.8.8v.7"/><path d="M7 11l2 2 4-4.5"/></g></svg>',
    "meetings": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}>'
                '<rect x="2.5" y="4" width="15" height="13" rx="2"/><path d="M2.5 8h15M6.5 2.5v3M13.5 2.5v3"/>'
                '<path d="M6 11.5h3M6 14h6"/></g></svg>',
    "features": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}>'
                '<path d="M10 2.5l7.5 4-7.5 4-7.5-4z"/><path d="M2.5 10l7.5 4 7.5-4"/><path d="M2.5 13.5l7.5 4 7.5-4"/>'
                '</g></svg>',
    "map": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}>'
           '<circle cx="4.5" cy="14.5" r="2"/><circle cx="10" cy="5" r="2"/><circle cx="15.5" cy="13" r="2"/>'
           '<path d="M5.6 12.8l3.4-6M11.2 6.7l3.2 4.6M6.5 14.3l7-1.1"/></g></svg>',
    "ask": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}>'
           '<path d="M4.5 3.5h11a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H9l-3.5 3v-3h-1a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2z"/>'
           '<path d="M8.3 7.6a1.8 1.8 0 1 1 2.4 1.7c-.5.2-.7.6-.7 1.1v.3"/><circle cx="10" cy="12.4" r=".4"/></g></svg>',
    "more": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g fill="currentColor">'
            '<circle cx="4.5" cy="10" r="1.4"/><circle cx="10" cy="10" r="1.4"/><circle cx="15.5" cy="10" r="1.4"/>'
            '</g></svg>',
    "add": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><g {_I}><path d="M10 4v12M4 10h12"/></g></svg>',
    "record": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><circle cx="10" cy="10" r="7" fill="none" '
              'stroke="currentColor" stroke-width="1.5"/><circle cx="10" cy="10" r="3.6" fill="currentColor"/></svg>',
}


def _render(svg: str, size: QSize, dpr: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size.width() * dpr), int(size.height() * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(p)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str = theme.MUTED, on_color: str = theme.ACCENT_TEXT, size: int = 20) -> QIcon:
    """A line icon; the checked/on state is drawn in the accent colour."""
    svg = ICONS[name]
    ic = QIcon()
    s = QSize(size, size)
    ic.addPixmap(_render(svg.replace("currentColor", color), s), QIcon.Normal, QIcon.Off)
    ic.addPixmap(_render(svg.replace("currentColor", on_color), s), QIcon.Normal, QIcon.On)
    ic.addPixmap(_render(svg.replace("currentColor", theme.TEXT), s), QIcon.Active, QIcon.Off)
    ic.addPixmap(_render(svg.replace("currentColor", "#bdbdbd"), s), QIcon.Disabled, QIcon.Off)
    return ic


def logo_icon() -> QIcon:
    ic = QIcon()
    for px in (16, 20, 24, 32, 48, 64, 128, 256):
        ic.addPixmap(_render(LOGO_SVG, QSize(px, px), 1.0))
    return ic


def logo_pixmap(size: int) -> QPixmap:
    return _render(LOGO_SVG, QSize(size, size))


# ---------------------------------------------------------------------------------------------------------
class Backdrop(QWidget):
    """Page background: Fluent base grey, the doodle wallpaper at very low contrast, and a soft two-layer
    shadow (Fluent 'shadow4') under each registered panel, so white panels float slightly above it."""
    DOODLE_OPACITY = 0.075
    TILE = DOODLE_TILE

    def __init__(self):
        super().__init__()
        self._panels: list[QWidget] = []
        self._tile: QPixmap | None = None
        self._tile_dpr = 0.0

    def add_panels(self, *panels: QWidget):
        for w in panels:
            self._panels.append(w)
            w.installEventFilter(self._watch)

    @property
    def _watch(self) -> QObject:
        if not hasattr(self, "_watcher"):
            self._watcher = _Repaint(self)
        return self._watcher

    def _doodles(self) -> QPixmap:
        dpr = self.devicePixelRatioF()
        if self._tile is None or self._tile_dpr != dpr:
            self._tile = _render(DOODLE_SVG, QSize(self.TILE, self.TILE), dpr)
            self._tile_dpr = dpr
        return self._tile

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        p.setOpacity(self.DOODLE_OPACITY)
        brush = QBrush(self._doodles())
        brush.setTransform(QTransform().translate(-36, -24))      # offset so the pattern does not start flush
        p.fillRect(self.rect(), brush)
        p.setOpacity(1.0)
        p.setPen(Qt.NoPen)
        for w in self._panels:
            if not w.isVisible() or not self.isAncestorOf(w):
                continue
            r = QRectF(w.mapTo(self, w.rect().topLeft()), w.size())
            for grow, dy, alpha in ((0.5, 0.5, 26), (2, 1.5, 12), (5, 3, 6), (9, 5, 3)):   # tight + ambient
                path = QPainterPath()
                path.addRoundedRect(r.adjusted(-grow, -grow + dy, grow, grow + dy), 8 + grow, 8 + grow)
                p.fillPath(path, QColor(0, 0, 0, alpha))


class _Repaint(QObject):
    def __init__(self, backdrop: Backdrop):
        super().__init__(backdrop)
        self.backdrop = backdrop

    def eventFilter(self, _obj, ev):
        if ev.type() in (QEvent.Move, QEvent.Resize, QEvent.Show, QEvent.Hide):
            self.backdrop.update()
        return False


# ---------------------------------------------------------------------------------------------------------
def tint_title_bar(widget: QWidget, caption: str = theme.SURFACE, border: str = theme.BORDER,
                   text: str = theme.TEXT) -> None:
    """Windows 11: colour the native title bar like the app bar (DWM caption/border/text colour)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = wintypes.HWND(int(widget.winId()))
        dwm = ctypes.windll.dwmapi

        def colorref(hex_: str) -> ctypes.c_uint:
            c = QColor(hex_)
            return ctypes.c_uint(c.red() | (c.green() << 8) | (c.blue() << 16))

        for attr, value in ((35, colorref(caption)), (34, colorref(border)), (36, colorref(text))):
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value), ctypes.sizeof(value))
        corner = ctypes.c_int(2)                                   # DWMWCP_ROUND
        dwm.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(corner), ctypes.sizeof(corner))
    except Exception:
        pass                                                       # older Windows: keep the default frame


def set_app_id(app_id: str = "MeetingGraph.Desktop") -> None:
    """Own taskbar identity, so Windows shows the MeetingGraph icon instead of Python's."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass
