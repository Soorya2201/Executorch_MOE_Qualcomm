"""Design tokens: the ONLY place colors, type sizes, spacing, radii and motion timings are defined.

Widgets ask for values by name (`tokens.c("secondary")`, `tokens.SPACE[16]`, `tokens.MOTION["view"]`), never
hard-code them. Light and dark palettes follow the Windows app-mode setting automatically (see theme.py).
"""
from __future__ import annotations

import sys

# ---------- type ----------
FONT_FAMILY = "Segoe UI Variable"
FONT_FALLBACK = "Segoe UI"
TYPE = {                 # px, weight
    "title": (26, 600),      # page title
    "section": (17, 600),    # section header
    "body": (14, 400),
    "meta": (12, 400),       # secondary / meta
}
LINE_HEIGHT = 1.4

# ---------- layout ----------
GRID = 4
SPACE = {n: n for n in (4, 8, 12, 16, 24, 32)}      # use SPACE[16] etc. so intent is visible in code
RADIUS = {"control": 6, "card": 10, "pill": 999}
SIDEBAR_W = 240
INSPECTOR_W = 340
READ_MAX_W = 720

# ---------- color ----------
ACCENT = "#0A84FF"
RECORD_RED = "#FF3B30"
STATUS = {               # feature status -> dot color (dots only, never big fills)
    "proposed": "#8E8E93",   # gray
    "active": "#34C759",     # accepted
    "done": "#34C759",
    "changed": "#FF9F0A",    # changed since first saved
    "blocked": "#FF3B30",    # conflict / blocked
    "archived": "#C7C7CC",
}
STAGE = {                # review-queue stage -> dot color
    "ready": ACCENT, "failed": "#FF3B30", "cleanup_failed": "#FF3B30",
    "transcribing": "#8E8E93", "summarizing": "#8E8E93", "saving": "#8E8E93", "recorded": "#8E8E93",
    "recording": RECORD_RED,
}

PALETTES = {
    "light": {
        "bg": "#F5F5F7", "surface": "#FFFFFF", "sidebar": "#EDEDF0", "surface2": "#F0F0F3",
        "text": "#1D1D1F", "secondary": "#6E6E73", "tertiary": "#8E8E93",
        "separator": "rgba(0,0,0,0.08)", "separator_solid": "#E3E3E6",
        "hover": "rgba(0,0,0,0.05)", "press": "rgba(0,0,0,0.09)",
        "accent": ACCENT, "accent_hover": "#0077ED", "accent_press": "#006EDB",
        "accent_soft": "rgba(10,132,255,0.12)", "accent_text": "#0066CC",
        "on_accent": "#FFFFFF", "danger": RECORD_RED, "success": "#248A3D", "warning": "#B25000",
        "warning_soft": "rgba(255,159,10,0.14)", "input": "#FFFFFF",
        "danger_soft": "rgba(255,59,48,0.12)", "success_strong": "#248A3D", "success_hover": "#1E7A35",
        "shadow": (0, 0, 0, 38),
    },
    "dark": {
        "bg": "#1C1C1E", "surface": "#2C2C2E", "sidebar": "#232325", "surface2": "#3A3A3C",
        "text": "#F5F5F7", "secondary": "#A1A1A6", "tertiary": "#8E8E93",
        "separator": "rgba(255,255,255,0.10)", "separator_solid": "#3A3A3C",
        "hover": "rgba(255,255,255,0.06)", "press": "rgba(255,255,255,0.11)",
        "accent": ACCENT, "accent_hover": "#3395FF", "accent_press": "#0A7AEF",
        "accent_soft": "rgba(10,132,255,0.22)", "accent_text": "#64B0FF",
        "on_accent": "#FFFFFF", "danger": "#FF453A", "success": "#30D158", "warning": "#FFB340",
        "warning_soft": "rgba(255,159,10,0.18)", "input": "#1C1C1E",
        "danger_soft": "rgba(255,69,58,0.20)", "success_strong": "#248A3D", "success_hover": "#2C9A49",
        "shadow": (0, 0, 0, 110),
    },
}

# ---------- motion ----------
MOTION = {"hover": 120, "select": 200, "view": 200, "panel": 280, "toast": 250, "max": 350}
VIEW_SLIDE_PX = 8
TOAST_MS = 6000

_mode = "light"


def set_mode(mode: str) -> None:
    global _mode
    _mode = "dark" if mode == "dark" else "light"


def mode() -> str:
    return _mode


def c(name: str) -> str:
    """Current-theme color by token name."""
    return PALETTES[_mode][name]


def palette() -> dict:
    return PALETTES[_mode]


def status_color(status: str | None) -> str:
    return STATUS.get(status or "", STATUS["proposed"])


def animations_enabled() -> bool:
    """Respect Windows Settings > Accessibility > Visual effects > Animation effects."""
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        val = ctypes.c_int(1)
        SPI_GETCLIENTAREAANIMATION = 0x1042
        if ctypes.windll.user32.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(val), 0):
            return bool(val.value)
    except Exception:
        pass
    return True


# ---------- map (knowledge graph) ----------
GRAPH = {
    "bg": {"light": "#FAFAFA", "dark": "#1E1E20"},
    "conflict": {"light": "#FF3B30", "dark": "#FF453A"},
    "node.base_r": 4.0, "node.deg_k": 2.2, "node.min_r": 4.0, "node.max_r": 14.0,
    "node.ring_w": 2.0, "node.ring_gap": 3.0,
    "meeting.size": 13.0, "meeting.radius": 3.5,
    "edge.width": 1.0, "edge.min_w": 0.75, "edge.max_w": 2.0,
    "edge.alpha.strong": 0.22,      # depends on / produces
    "edge.alpha.weak": 0.14,        # uses / is part of / related
    "edge.alpha.meeting": 0.30,     # transcript/meeting -> feature (dashed 3/3, accent tint)
    "edge.alpha.tint.strong": 0.45, # links tinted with their node's group colour
    "edge.alpha.tint.weak": 0.32,
    "ungrouped": {"light": "#AEAEB2", "dark": "#636366"},   # features with no links
    "edge.alpha.hover": 0.60,       # edges to a hovered node's neighbours (node color)
    "dim.alpha": 0.18,              # everything outside the hover focus
    "arrow.size": 5.0,
    "label.px": 12, "label.gap": 4, "label.max_chars": 24, "label.top_n": 5,
    "label.zoom.hide": 0.6, "label.zoom.full": 1.0,
    "zoom.min": 0.2, "zoom.max": 4.0, "zoom.step": 1.2, "zoom.fit_max": 1.6,
    "fit.pad": 48, "link.distance": 90.0, "meeting.link.distance": 160.0,
    "anim.hover_ms": 120, "anim.card_ms": 120, "anim.wheel_ms": 120, "anim.filter_ms": 150,
    "anim.fit_ms": 200, "anim.panel_ms": 200, "anim.glide_ms": 300,
    "pin.r": 2.0,
    # "Color by first meeting": Apple system colors, readable on both backgrounds
    # blue is kept for transcripts/meetings, so topic groups start with other hues
    "categorical": ["#34C759", "#FF9F0A", "#AF52DE", "#30B0C7", "#FF2D55", "#5856D6", "#A2845E", "#FFCC00"],
    "card.w": 260, "card.offset": 14, "card.pad": 12,
}


def g(key: str):
    """Graph token; mode-dependent ones resolve for the current light/dark mode."""
    v = GRAPH[key]
    return v[_mode] if isinstance(v, dict) else v
