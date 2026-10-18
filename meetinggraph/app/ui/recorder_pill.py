"""Small, always-on-top recording capsule (like the Windows Game Bar capture widget).

  [● rec] [❚❚] [■]   ~~~ live sound wave ~~~   00:42 · 12 lines   ✕

Pause and stop are disabled until recording starts. ✕ always asks whether to keep what was recorded.
Records the microphone AND the computer's sound (speaker loopback) as two tracks. The wave follows the
loudest track; icons are painted (not font glyphs) so they look the same on every machine.
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from ..capture import recorder

BG = QColor(18, 20, 26, 238)
EDGE = QColor(255, 255, 255, 28)
TEXT = "#f3f4f6"
MUTED = "#9ca3af"
RED = QColor("#ef4444")
WAVE = (QColor("#479ef5"), QColor("#7f6bd9"), QColor("#d660c9"))   # Fluent blue -> violet -> magenta
FPS_MS = 16

PILL_QSS = f"""
QLabel {{ background: transparent; font-family: 'Segoe UI Variable Text', 'Segoe UI'; }}
QLabel#time {{ color: {TEXT}; font-size: 13pt; font-weight: 600; }}
QLabel#state {{ color: {MUTED}; font-size: 8.5pt; }}
"""


class _IconButton(QAbstractButton):
    """Round button with a painted icon: rec | pause | stop | close."""

    def __init__(self, kind: str, size: int, tip: str):
        super().__init__()
        self.kind, self._size = kind, size
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tip)
        self.pulse = 0.0          # 0..1 recording halo (rec button only), driven by the capsule's animation
        self.paused = False       # rec button shows "resume" while paused

    def sizeHint(self):
        return QSize(self._size, self._size)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        c = r.center()
        on, hover = self.isEnabled(), self.underMouse() and self.isEnabled()
        if self.kind == "close":
            if hover:
                p.setBrush(QColor(239, 68, 68, 210))
                p.setPen(Qt.NoPen)
                p.drawEllipse(r.adjusted(2, 2, -2, -2))
            p.setPen(QPen(QColor(TEXT if hover else MUTED), 1.6, Qt.SolidLine, Qt.RoundCap))
            d = r.width() * 0.17
            p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
            p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))
            return
        if self.kind == "rec":
            if self.pulse > 0:                                   # soft halo while recording
                halo = QRadialGradient(c, r.width() / 2)
                a = int(90 * self.pulse)
                halo.setColorAt(0.55, QColor(239, 68, 68, a))
                halo.setColorAt(1.0, QColor(239, 68, 68, 0))
                p.setBrush(halo)
                p.setPen(Qt.NoPen)
                p.drawEllipse(r)
            inner = r.adjusted(6, 6, -6, -6)
            live = self.pulse > 0                                # recording: disabled, but shown as a live light
            base = QColor("#f87171") if hover else RED
            if not on and not live:
                base = QColor(127, 29, 29)
            p.setBrush(base)
            p.setPen(Qt.NoPen)
            p.drawEllipse(inner)
            p.setBrush(QColor(255, 255, 255, 235 if on or live else 110))
            if self.paused:                                      # resume: play triangle
                s = inner.width() * 0.22
                tri = QPainterPath(QPointF(c.x() - s * 0.7, c.y() - s))
                tri.lineTo(QPointF(c.x() + s, c.y()))
                tri.lineTo(QPointF(c.x() - s * 0.7, c.y() + s))
                tri.closeSubpath()
                p.drawPath(tri)
            else:
                d = inner.width() * 0.2
                p.drawEllipse(c, d, d)
            return
        # pause / stop: subtle glass circle
        p.setBrush(QColor(255, 255, 255, 34 if hover else 18 if on else 8))
        p.setPen(Qt.NoPen)
        p.drawEllipse(r.adjusted(4, 4, -4, -4))
        p.setBrush(QColor(TEXT) if on else QColor(255, 255, 255, 60))
        s = r.width() * 0.13
        if self.kind == "pause":
            w = s * 0.62
            p.drawRoundedRect(QRectF(c.x() - s - w * 0.2, c.y() - s * 1.2, w, s * 2.4), 1.5, 1.5)
            p.drawRoundedRect(QRectF(c.x() + s * 0.2 + w * 0.2, c.y() - s * 1.2, w, s * 2.4), 1.5, 1.5)
        else:
            p.drawRoundedRect(QRectF(c.x() - s, c.y() - s, 2 * s, 2 * s), 2.5, 2.5)


class _Wave(QWidget):
    """Mirrored bars that pulse with the sound level: fast attack, slow release, a travelling wobble so it
    never looks mechanical, and a gentle 'breathing' line when it is quiet."""
    BARS = 30

    def __init__(self):
        super().__init__()
        self.setMinimumSize(140, 40)
        self.level = 0.0          # smoothed 0..1
        self.target = 0.0
        self.state = "idle"       # idle | live | paused
        self.t0 = time.monotonic()
        self.peaks = [0.0] * self.BARS

    def feed(self, raw: float):
        # perceptual curve: quiet speech still moves the wave, shouting does not slam the ceiling
        self.target = min(1.0, math.sqrt(max(0.0, raw)) * 1.15)

    def step(self):
        k = 0.45 if self.target > self.level else 0.07
        self.level += (self.target - self.level) * k
        t = time.monotonic() - self.t0
        n = self.BARS
        for i in range(n):
            x = (i + 0.5) / n
            env = math.sin(math.pi * x) ** 1.4                    # taller in the middle
            wob = 0.62 + 0.38 * math.sin(t * 6.2 - i * 0.55) * math.sin(t * 3.9 + i * 0.27)
            if self.state == "live":
                h = 0.07 + 0.93 * self.level * env * wob
                h = max(h, 0.05 + 0.04 * (0.5 + 0.5 * math.sin(t * 2.2 - i * 0.4)))   # breathing floor
            elif self.state == "idle":
                h = 0.05 + 0.05 * env * (0.5 + 0.5 * math.sin(t * 1.6 - i * 0.35))
            else:
                h = 0.04
            # each bar eases toward its height: smooth even when the level jumps
            self.peaks[i] += (h - self.peaks[i]) * (0.5 if h > self.peaks[i] else 0.18)
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        n = self.BARS
        gap = w / n
        bw = max(2.0, gap * 0.44)
        mid = h / 2
        grad = QLinearGradient(0, 0, w, 0)
        alpha = 255 if self.state == "live" else 120 if self.state == "idle" else 70
        for pos, col in zip((0.0, 0.5, 1.0), WAVE):
            c = QColor(col)
            c.setAlpha(alpha)
            grad.setColorAt(pos, c)
        if self.state == "live" and self.level > 0.04:            # soft light behind the wave while sound is heard
            # a circle of radius h/2 stretched to the width: fades to nothing inside the widget (no clipped edges)
            p.save()
            p.translate(w / 2, mid)
            p.scale((w / 2) / (h / 2), 1.0)
            amb = QRadialGradient(QPointF(0, 0), h / 2)
            amb.setColorAt(0.0, QColor(192, 132, 252, int(75 * min(1.0, self.level * 1.5))))
            amb.setColorAt(1.0, QColor(192, 132, 252, 0))
            p.setPen(Qt.NoPen)
            p.setBrush(amb)
            p.drawEllipse(QPointF(0, 0), h / 2, h / 2)
            p.restore()
        p.setPen(QPen(QBrush(grad), bw, Qt.SolidLine, Qt.RoundCap))
        for i in range(n):
            x = gap * (i + 0.5)
            half = max(0.5, self.peaks[i] * (h * 0.46))
            p.drawLine(QPointF(x, mid - half), QPointF(x, mid + half))


class RecorderPill(QWidget):
    finished = Signal(str, bool)      # (session id or "", keep the recording?)

    def __init__(self, ctx):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.ctx = ctx
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(PILL_QSS)
        self.setFixedSize(440, 64)
        self.rec: recorder.DualRecorder | None = None
        self.sid: str | None = None
        self.project_id: str | None = None
        self.project_name = ""
        self.title = "Meeting"
        self._drag: QPoint | None = None
        self.live = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        self.b_rec = _IconButton("rec", 48, "Record")
        self.b_pause = _IconButton("pause", 40, "Pause")
        self.b_stop = _IconButton("stop", 40, "Stop and make the transcript")
        for b in (self.b_rec, self.b_pause, self.b_stop):
            lay.addWidget(b, 0, Qt.AlignVCenter)
        lay.addSpacing(6)
        self.wave = _Wave()
        lay.addWidget(self.wave, 1)
        lay.addSpacing(8)
        info = QVBoxLayout()
        info.setSpacing(0)
        info.setContentsMargins(0, 0, 0, 0)
        self.time = QLabel("00:00")
        self.time.setObjectName("time")
        self.state = QLabel("Ready")
        self.state.setObjectName("state")
        self.time.setFixedWidth(78)
        self.state.setFixedWidth(78)
        info.addStretch(1)
        info.addWidget(self.time)
        info.addWidget(self.state)
        info.addStretch(1)
        lay.addLayout(info)
        self.b_close = _IconButton("close", 28, "Close")
        lay.addWidget(self.b_close, 0, Qt.AlignVCenter)

        self.b_rec.clicked.connect(self._on_record)
        self.b_pause.clicked.connect(self._on_pause)
        self.b_stop.clicked.connect(lambda: self._finish(True))
        self.b_close.clicked.connect(self._on_close)
        self.timer = QTimer(self)                 # clock + status text
        self.timer.timeout.connect(self._tick)
        self.anim = QTimer(self)                  # ~60 fps wave, only while the capsule is visible
        self.anim.setTimerType(Qt.PreciseTimer)
        self.anim.timeout.connect(self._animate)
        self._set_buttons()

    # ---------- dialogs ----------
    # Dialogs get NO parent: a child of the capsule would inherit its light-text style and show light
    # text on a light dialog. Parentless dialogs use the app's normal theme.
    @staticmethod
    def _box(icon, title: str, text: str, info: str = "") -> QMessageBox:
        box = QMessageBox(icon, title, text)
        if info:
            box.setInformativeText(info)
        box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        return box

    def _message(self, icon, title: str, text: str):
        box = self._box(icon, title, text)
        box.addButton("OK", QMessageBox.AcceptRole)
        box.exec()

    # ---------- look ----------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(r, r.height() / 2, r.height() / 2)
        p.fillPath(path, QBrush(BG))
        sheen = QLinearGradient(0, r.top(), 0, r.bottom())          # faint top highlight: glassy, not flat
        sheen.setColorAt(0, QColor(255, 255, 255, 18))
        sheen.setColorAt(0.5, QColor(255, 255, 255, 0))
        p.fillPath(path, QBrush(sheen))
        p.setPen(QPen(EDGE, 1))
        p.drawPath(path)

    def showEvent(self, ev):
        self.anim.start(FPS_MS)
        super().showEvent(ev)

    def hideEvent(self, ev):
        self.anim.stop()
        super().hideEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _ev):
        self._drag = None

    # ---------- public ----------
    @property
    def recording(self) -> bool:
        return self.rec is not None

    def begin(self, project_id: str, project_name: str, title: str):
        """Show the capsule, ready to record (nothing is recorded until ● is pressed)."""
        self.project_id, self.project_name, self.title = project_id, project_name, title or "Meeting"
        self.sid, self.rec = None, None
        self.time.setText("00:00")
        self._status("Ready", f"{self.title} · {self.project_name}")
        self.wave.state = "idle"
        self.wave.feed(0)
        self._set_buttons()
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - self.width() // 2, scr.top() + 24)
        self.show()
        self.raise_()

    def stop(self):
        """Used by the tray menu / app quit: stop and keep the recording."""
        if self.isVisible():
            self._finish(True)

    # ---------- buttons ----------
    def _set_buttons(self):
        started = self.rec is not None
        paused = started and self.rec.is_paused
        self.b_rec.setEnabled(not started or paused)       # ● starts, or resumes after a pause
        self.b_rec.paused = paused
        self.b_rec.setToolTip("Resume" if paused else "Record")
        self.b_pause.setEnabled(started and not paused)
        self.b_stop.setEnabled(started)
        for b in (self.b_rec, self.b_pause, self.b_stop):
            b.update()

    def _status(self, text: str, tip: str = ""):
        # elide so it never runs into the close button
        self.state.setText(self.state.fontMetrics().elidedText(text, Qt.ElideRight, self.state.width()))
        self.state.setToolTip(tip or text)

    def _on_record(self):
        if self.rec is not None and self.rec.is_paused:
            self.rec.resume()
            self._set_buttons()
            return
        devs = recorder.list_devices()
        if devs.get("error"):
            self._message(QMessageBox.Warning, "Can't record", f"Recording isn't available on this computer:\n{devs['error']}")
            return
        mic, spk = devs.get("default_mic"), devs.get("default_speaker")
        s = self.ctx.sm.create("recording", self.project_id, self.project_name, self.title)
        mic_path, loop_path = self.ctx.sm.recording_paths(s["id"])
        try:
            rec = recorder.DualRecorder(mic, spk, mic_path, loop_path)
            warnings = rec.start()
        except Exception as e:
            self.ctx.sm.discard(s["id"])
            self._message(QMessageBox.Warning, "Can't record", str(e))
            return
        self.rec, self.sid = rec, s["id"]
        try:
            self.live = self.ctx.sm.start_live(s["id"], rec)      # transcript written every 30 s
        except Exception:
            self.live = None                                       # falls back to transcribing after Stop
        self._warnings = "\n".join(warnings)
        self.timer.start(250)
        self._set_buttons()
        self._tick()

    def _on_pause(self):
        if self.rec:
            self.rec.pause()
            self._set_buttons()
            self._tick()

    def _on_close(self):
        box = self._box(QMessageBox.Question, "Close recorder", "Save the transcript recorded so far?",
                        "Yes: stop and turn what was recorded into a transcript.\n"
                        "No: throw the recording away.")
        yes = box.addButton("Yes, save it", QMessageBox.YesRole)
        no = box.addButton("No, discard", QMessageBox.NoRole)
        box.addButton("Keep recording", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is yes:
            if self.rec is None:
                self._message(QMessageBox.Information, "Nothing recorded",
                              "Nothing has been recorded yet, so there is nothing to save.")
                self._finish(False)
            else:
                self._finish(True)
        elif clicked is no:
            self._finish(False)

    def _finish(self, keep: bool):
        self.timer.stop()
        sid = self.sid
        if self.rec is not None:
            offsets, notes = self.rec.stop()
            self.rec = None
            if keep and sid:
                self.ctx.sm.finish_recording(sid, offsets, notes)
            elif sid:
                try:
                    self.ctx.sm.stop_live(sid)
                    self.ctx.sm.discard(sid)
                except Exception:
                    pass
        self.sid = None
        self.live = None
        self.wave.state = "idle"
        self.b_rec.pulse = 0.0
        self.hide()
        self.finished.emit((sid or "") if keep else "", bool(keep and sid))

    # ---------- live display ----------
    def _animate(self):
        live = self.rec is not None and not self.rec.is_paused
        if self.rec is None:
            self.wave.state = "idle"
            self.wave.feed(0)
        elif self.rec.is_paused:
            self.wave.state = "paused"
            self.wave.feed(0)
        else:
            self.wave.state = "live"
            lv = self.rec.levels()
            self.wave.feed(max(lv.values() or [0.0]))
        self.wave.step()
        pulse = (0.5 + 0.5 * math.sin(time.monotonic() * 3.4)) if live else 0.0
        if abs(pulse - self.b_rec.pulse) > 0.01:
            self.b_rec.pulse = pulse
            self.b_rec.update()

    def _tick(self):
        if not self.rec:
            return
        e = int(self.rec.elapsed())
        self.time.setText(f"{e // 3600:d}:{(e % 3600) // 60:02d}:{e % 60:02d}" if e >= 3600 else f"{e // 60:02d}:{e % 60:02d}")
        warn = getattr(self, "_warnings", "")
        if self.rec.is_paused:
            self._status("Paused", "Paused · press ▶ to continue")
        else:
            health = self.rec.health()
            st = self.live.stats if self.live else None
            if health:
                self._status("⚠ " + health[0], "\n".join(health))
            elif st and (st["lines"] or st["batches"]):
                notes = f" · {st['notes_done']} notes" if st["notes_done"] else ""
                self._status(f"{st['lines']} lines{notes}", f"Latest: {st['last']}" if st["last"] else "Listening…")
            elif st:
                self._status("Listening…", "The first transcript lines appear after about 30 seconds."
                             + (f"\n{warn}" if warn else ""))
            else:
                self._status("Recording", warn or "Recording your microphone and the computer's sound")
        if e > 60 * int(self.ctx.cfg["audio"].get("max_minutes", 60)):
            self._finish(True)
