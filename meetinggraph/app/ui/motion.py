"""Small motion helpers. Every animation goes through here so the Windows "Animation effects" setting is
respected (off -> instant) and durations/easing stay consistent (tokens.MOTION, OutCubic, never > 350 ms)."""
from __future__ import annotations

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QParallelAnimationGroup, QPoint, QPropertyAnimation, \
    QVariantAnimation
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from . import tokens as tk

EASE = QEasingCurve.OutCubic
_enabled: bool | None = None


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = tk.animations_enabled()
    return _enabled


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = on


def ms(kind: str) -> int:
    return min(tk.MOTION.get(kind, 200), tk.MOTION["max"]) if enabled() else 0


def _keep(owner, anim):
    """Keep a reference so the animation isn't garbage-collected mid-flight; replace any previous one."""
    old = getattr(owner, "_mg_anim", None)
    if old is not None and old.state() == QAbstractAnimation.Running:
        old.stop()
    owner._mg_anim = anim
    return anim


def value(owner, start, end, kind: str, on_value, on_done=None):
    """Animate a plain value (float, QRectF, QColor...) and call on_value(v) each frame."""
    if not enabled():
        on_value(end)
        if on_done:
            on_done()
        return None
    a = QVariantAnimation(owner)
    a.setStartValue(start)
    a.setEndValue(end)
    a.setDuration(ms(kind))
    a.setEasingCurve(EASE)
    a.valueChanged.connect(on_value)
    if on_done:
        a.finished.connect(on_done)
    _keep(owner, a).start()
    return a


def fade_slide_in(w: QWidget, kind: str = "view", dy: int = tk.VIEW_SLIDE_PX):
    """New content: cross-fade in with a slight upward slide."""
    if not enabled():
        return
    eff = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(eff)
    fade = QPropertyAnimation(eff, b"opacity", w)
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)
    fade.setDuration(ms(kind))
    fade.setEasingCurve(EASE)
    end = w.pos()
    slide = QPropertyAnimation(w, b"pos", w)
    slide.setStartValue(end + QPoint(0, dy))
    slide.setEndValue(end)
    slide.setDuration(ms(kind))
    slide.setEasingCurve(EASE)
    g = QParallelAnimationGroup(w)
    g.addAnimation(fade)
    g.addAnimation(slide)
    # remove the effect afterwards: opacity effects cost rendering time and blur text slightly
    g.finished.connect(lambda: w.setGraphicsEffect(None))
    _keep(w, g).start()


def fade(w: QWidget, to: float, kind: str = "toast", on_done=None):
    """Fade a widget's opacity to `to` (0..1)."""
    eff = w.graphicsEffect()
    if not isinstance(eff, QGraphicsOpacityEffect):
        eff = QGraphicsOpacityEffect(w)
        eff.setOpacity(1.0 if to == 0 else 0.0)
        w.setGraphicsEffect(eff)
    if not enabled():
        eff.setOpacity(to)
        if on_done:
            on_done()
        return
    a = QPropertyAnimation(eff, b"opacity", w)
    a.setStartValue(eff.opacity())
    a.setEndValue(to)
    a.setDuration(ms(kind))
    a.setEasingCurve(EASE)
    if on_done:
        a.finished.connect(on_done)
    _keep(w, a).start()
