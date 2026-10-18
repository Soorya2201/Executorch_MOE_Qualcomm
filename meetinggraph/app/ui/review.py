"""Review screen for one meeting: 1) summary  2) changes as plain sentences + questions  3) save."""
from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QFormLayout, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
                               QScrollArea, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

from ..pipeline.match import _card_from_existing, sync_relations
from ..pipeline.transcript import fmt_ts, norm
from ..workspace.store import RELATION_LABEL, STATUSES
from .common import STAGE_TEXT, esc
from . import theme

HIDDEN_FLAG_PREFIXES = ("Meeting said", "Possible duplicate", "No existing feature matches", "Uncertain:",
                        "No supporting transcript evidence")


def _lines(xs) -> list[str]:
    return [x for x in (xs or []) if str(x).strip()]


REL_TYPES = ("needs", "part of", "produces", "controls", "uses", "related")


def relations_to_text(rels) -> str:
    return "\n".join(f"{r.get('type', 'related')} | {r.get('to', '')} | {r.get('why', '')}".rstrip(" |")
                     for r in rels or [] if r.get("to"))


def relations_from_text(text: str) -> list[dict]:
    """Lines like:  needs | Ribosomes | proteins are built on ribosomes"""
    out = []
    for ln in text.splitlines():
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 2 or not parts[1]:
            continue
        typ = parts[0].lower() if parts[0].lower() in REL_TYPES else "related"
        out.append({"to": parts[1], "type": typ, "why": " | ".join(parts[2:]).strip()})
    return out


def relation_html(r: dict) -> str:
    why = r.get("why") or "<i>no reason given</i>"
    if r.get("why"):
        why = esc(why)
    return f"{esc(RELATION_LABEL.get(r.get('type'), r.get('type', '')))} <b>{esc(r.get('to', ''))}</b> — {why}"


class ChangeCard(QFrame):
    changed = Signal()
    policy = Signal(object)       # archive policy chosen on this card

    def __init__(self, card: dict, features: list[dict]):
        super().__init__()
        self.card = card
        self.features = features
        self.by_id = {f["id"]: f for f in features}
        self.titles = {f["id"]: f["title"] for f in features}
        self.mention = card.get("mention") or {
            "title": card.get("title", ""), "action": card.get("action", "add"),
            "description": card.get("description", ""), "status": card.get("status", ""),
            "subtasks": card.get("subtasks", []), "depends_on": card.get("depends_on", []),
            "impact": card.get("impact", ""), "key_points": card.get("key_points", []),
            "relations": card.get("relations", [])}
        self.said = card.get("said_as") or card.get("title", "")
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        head = QHBoxLayout()
        self.pill = QLabel()
        self.title_lbl = QLabel()
        self.title_lbl.setObjectName("h2")
        self.include = QCheckBox("Include this change")
        self.include.setChecked(card.get("include", True))
        head.addWidget(self.pill)
        head.addWidget(self.title_lbl, 1)
        head.addWidget(self.include)
        lay.addLayout(head)

        self.what = QLabel()
        self.what.setWordWrap(True)
        self.what.setTextFormat(Qt.RichText)
        lay.addWidget(self.what)

        self.qbox = QFrame()
        self.qbox.setObjectName("question")
        self.qlay = QVBoxLayout(self.qbox)
        lay.addWidget(self.qbox)
        self.qgroup: QButtonGroup | None = None

        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.RichText)
        lay.addWidget(self.notes)

        self.toggle = QPushButton("Edit details ▸")
        self.toggle.setObjectName("link")
        lay.addWidget(self.toggle, 0, Qt.AlignLeft)
        self.form_w = QWidget()
        form = QFormLayout(self.form_w)
        self.f_title = QLineEdit()
        self.f_status = QComboBox()
        self.f_status.addItems([s for s in STATUSES if s != "archived"])
        self.f_desc = QPlainTextEdit()
        self.f_desc.setFixedHeight(90)
        self.f_kp = QPlainTextEdit()
        self.f_kp.setPlaceholderText("One fact per line.")
        self.f_kp.setFixedHeight(90)
        self.f_rel = QPlainTextEdit()
        self.f_rel.setPlaceholderText("One per line:  kind | other item | why   (\"needs\" = needs it first)\n"
                                      "kind = needs, part of, produces, controls, uses or related\n"
                                      "e.g.  produces | ATP | it breaks down glucose to release energy")
        self.f_rel.setFixedHeight(80)
        self.f_subs = QPlainTextEdit()
        self.f_subs.setPlaceholderText("One subtask per line. Start a line with [x] if it's done.")
        self.f_subs.setFixedHeight(70)
        self.f_deps = QPlainTextEdit()
        self.f_deps.setPlaceholderText("One feature name per line (a feature in this project or in this meeting).")
        self.f_deps.setFixedHeight(54)
        self.f_impact = QLineEdit()
        form.addRow("Name", self.f_title)
        form.addRow("Status", self.f_status)
        form.addRow("What it is", self.f_desc)
        form.addRow("Key points", self.f_kp)
        form.addRow("Subtasks", self.f_subs)
        # dependencies are the "needs" lines of the connections field (one place to edit them)
        form.addRow("How it connects\n(and why)", self.f_rel)
        form.addRow("Impact", self.f_impact)
        self.form_w.setVisible(False)
        lay.addWidget(self.form_w)

        self._load_form()
        self._build_question()
        self.include.toggled.connect(self._on_change)
        self.toggle.clicked.connect(self._toggle)
        for w in (self.f_title, self.f_impact):
            w.textEdited.connect(self._on_edit)
        for w in (self.f_desc, self.f_subs, self.f_deps, self.f_kp, self.f_rel):
            w.textChanged.connect(self._on_edit)
        self.f_status.currentTextChanged.connect(self._on_edit)
        self.refresh()

    # ----- form <-> card -----
    def _load_form(self):
        sync_relations(self.card)          # every dependency shows up as a "needs" line
        c = self.card
        for w in (self.f_title, self.f_desc, self.f_subs, self.f_deps, self.f_impact, self.f_status, self.f_kp,
                  self.f_rel):
            w.blockSignals(True)
        self.f_title.setText(c.get("title", ""))
        self.f_status.setCurrentText(c.get("status") or "proposed")
        self.f_desc.setPlainText(c.get("description", ""))
        self.f_subs.setPlainText("\n".join(_lines(c.get("subtasks"))))
        self.f_deps.setPlainText("\n".join(_lines(c.get("depends_on"))))
        self.f_impact.setText(c.get("impact", ""))
        self.f_kp.setPlainText("\n".join(_lines(c.get("key_points"))))
        self.f_rel.setPlainText(relations_to_text(c.get("relations")))
        for w in (self.f_title, self.f_desc, self.f_subs, self.f_deps, self.f_impact, self.f_status, self.f_kp,
                  self.f_rel):
            w.blockSignals(False)

    def value(self) -> dict:
        c = dict(self.card)
        c.update({
            "include": self.include.isChecked(),
            "title": self.f_title.text().strip(), "status": self.f_status.currentText(),
            "description": self.f_desc.toPlainText().strip(),
            "subtasks": [s for s in self.f_subs.toPlainText().splitlines() if s.strip()],
            "impact": self.f_impact.text().strip(),
            "key_points": [s.strip() for s in self.f_kp.toPlainText().splitlines() if s.strip()],
            "relations": relations_from_text(self.f_rel.toPlainText()),
        })
        c["depends_on"] = [r["to"] for r in c["relations"] if r["type"] == "needs"]
        sync_relations(c)
        return c

    def _on_edit(self, *_):
        self.card = self.value()
        self.card["user_edited"] = True
        self.refresh()
        self.changed.emit()

    def _on_change(self, *_):
        self.card = self.value()
        self.refresh()
        self.changed.emit()

    def _toggle(self):
        v = not self.form_w.isVisible()
        self.form_w.setVisible(v)
        self.toggle.setText("Hide details ▾" if v else "Edit details ▸")

    # ----- switching between "update existing" and "new feature" -----
    def _as_new(self):
        m = self.mention
        self.card.update({"action": "add", "feature_id": None, "needs_confirmation": False, "match_confirmed": False,
                          "title": self.said, "description": m.get("description", ""),
                          "status": m.get("status") or "proposed", "subtasks": list(m.get("subtasks") or []),
                          "depends_on": list(m.get("depends_on") or []), "impact": m.get("impact", ""),
                          "key_points": list(m.get("key_points") or []),
                          "relations": [dict(r) for r in m.get("relations") or []]})

    def _as_existing(self, fid: str):
        m = self.mention
        base = {"action": "archive" if m.get("action") == "archive" else "change", "title": self.said,
                "description": m.get("description", ""), "status": m.get("status", ""),
                "subtasks": list(m.get("subtasks") or []), "depends_on": list(m.get("depends_on") or []),
                "impact": m.get("impact", ""), "key_points": list(m.get("key_points") or []),
                "relations": [dict(r) for r in m.get("relations") or []]}
        merged = _card_from_existing(base, self.by_id[fid], self.titles)
        self.card.update(merged)
        self.card.update({"needs_confirmation": True, "match_confirmed": True})

    def _build_question(self):
        while self.qlay.count():
            it = self.qlay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.qgroup = QButtonGroup(self)
        c = self.card
        q, options, checked = None, [], None
        cands = [x for x in c.get("candidates") or [] if x["feature_id"] in self.by_id]
        if c.get("needs_confirmation") and c.get("feature_id") in self.by_id:
            ex = self.by_id[c["feature_id"]]["title"]
            if c.get("action") == "archive":
                q = f"The meeting wants to drop “{esc(self.said)}”. Is that your existing feature “{esc(ex)}”?"
                options = [("yes", f"Yes — archive “{ex}”"), ("skip", "No — leave this change out")]
            else:
                q = f"The meeting talked about “{esc(self.said)}”. Is that your existing feature “{esc(ex)}”?"
                options = [("yes", f"Yes — update “{ex}”"), ("new", "No — it's a different, new feature")]
            checked = "yes" if c.get("match_confirmed") else None
            self._q_target = c["feature_id"]
        elif c.get("action") == "add" and cands and cands[0]["score"] >= 0.6:
            ex = cands[0]["title"]
            q = f"This looks similar to your existing feature “{esc(ex)}”."
            options = [("new", "It's a new feature — add it"), ("yes", f"It's the same — update “{ex}” instead")]
            checked = "new"
            self._q_target = cands[0]["feature_id"]
        if c.get("action") == "archive" and c.get("feature_id") in self.by_id:
            deps = [f["title"] for f in self.features
                    if f.get("status") != "archived" and c["feature_id"] in (f.get("depends_on") or [])]
            if deps:
                lbl = QLabel(f"<b>Other features need “{esc(self.by_id[c['feature_id']]['title'])}”:</b> "
                             f"{esc(', '.join(deps))}. What should happen to them?")
                lbl.setWordWrap(True)
                self.qlay.addWidget(lbl)
                pg = QButtonGroup(self)
                for key, text in (("remove_dependency", "Remove the link (they no longer need it)"),
                                  ("mark_blocked", "Remove the link and mark them “blocked”")):
                    rb = QRadioButton(text)
                    rb.toggled.connect(lambda on, k=key: on and self.policy.emit(k))
                    pg.addButton(rb)
                    self.qlay.addWidget(rb)
                self._policy_group = pg
        if q:
            lbl = QLabel(f"<b>Question:</b> {q}")
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.RichText)
            self.qlay.addWidget(lbl)
            for key, text in options:
                rb = QRadioButton(text)
                rb.setChecked(key == checked)
                rb.toggled.connect(lambda on, k=key: on and self._answer(k))
                self.qgroup.addButton(rb)
                self.qlay.addWidget(rb)
        self.qbox.setVisible(self.qlay.count() > 0)

    def _answer(self, key: str):
        if key == "yes":
            self._as_existing(self._q_target)
            self.include.setChecked(True)
        elif key == "new":
            self._as_new()
            self.include.setChecked(True)
        elif key == "skip":
            self.include.setChecked(False)
        self.card["user_edited"] = True
        self._load_form()
        self.refresh()
        self.changed.emit()

    # ----- plain-language view -----
    def unanswered(self) -> bool:
        c = self.value()
        return bool(c.get("include") and c.get("needs_confirmation") and not c.get("match_confirmed"))

    def refresh(self):
        c = self.value()
        action = c.get("action", "add")
        on = c.get("include", True)
        self.setObjectName("card" if on else "cardOff")
        self.style().unpolish(self)
        self.style().polish(self)
        prev = self.by_id.get(c.get("feature_id")) if action != "add" else None
        name = (prev or {}).get("title") if action == "archive" else c.get("title")
        self.pill.setText({"add": "NEW FEATURE", "change": "UPDATE", "archive": "ARCHIVE"}[action])
        self.pill.setObjectName({"add": "pillAdd", "change": "pillChange", "archive": "pillArchive"}[action])
        self.pill.style().unpolish(self.pill)
        self.pill.style().polish(self.pill)
        self.title_lbl.setText(esc(name or self.said))

        items = []
        if action == "archive":
            items.append("Marks this feature as archived. It stays in the history; nothing is deleted.")
        elif action == "add":
            items.append("Adds a new feature to the project.")
            if c.get("description"):
                items.append(f"<b>What it is:</b> {esc(c['description'])}")
            if _lines(c.get("key_points")):
                items.append("<b>Key points:</b><br>" + "<br>".join("• " + esc(k) for k in _lines(c["key_points"])))
            if _lines(c.get("subtasks")):
                items.append("<b>Subtasks:</b> " + esc("; ".join(_lines(c["subtasks"]))))
            if c.get("relations"):
                items.append("<b>How it connects:</b><br>" + "<br>".join("• " + relation_html(r)
                                                                       for r in c["relations"]))
            if c.get("impact"):
                items.append(f"<b>Impact:</b> {esc(c['impact'])}")
        else:
            p = prev or {}
            if c.get("status") and c.get("status") != p.get("status"):
                items.append(f"<b>Status:</b> {esc(p.get('status', '?'))} → <b>{esc(c['status'])}</b>")
            if c.get("description") and c.get("description") != p.get("description"):
                items.append(f"<b>What it is (updated):</b> {esc(c['description'])}")
            old_k = {norm(k) for k in p.get("key_points") or []}
            new_k = [k for k in _lines(c.get("key_points")) if norm(k) not in old_k]
            if new_k:
                items.append("<b>New key points:</b><br>" + "<br>".join("• " + esc(k) for k in new_k))
            old_s = {norm(s.get('text', '')) for s in p.get("subtasks") or []}
            new_s = [s for s in _lines(c.get("subtasks")) if norm(s.replace("[x]", "")) not in old_s]
            if new_s:
                items.append("<b>New subtasks:</b> " + esc("; ".join(new_s)))
            old_d = {norm(self.titles.get(d, d)) for d in p.get("depends_on") or []}
            new_d = [d for d in _lines(c.get("depends_on")) if norm(d) not in old_d]
            old_r = {(r.get("type"), norm(r.get("to_title") or self.titles.get(r.get("to"), r.get("to", ""))),
                      r.get("why", "")) for r in p.get("relations") or []}
            new_r = [r for r in c.get("relations") or []
                     if (r.get("type"), norm(r.get("to", "")), r.get("why", "")) not in old_r]
            if new_r:
                items.append("<b>How it connects (new or updated):</b><br>"
                             + "<br>".join("• " + relation_html(r) for r in new_r))
            elif new_d:
                items.append("<b>Now needs first:</b> " + esc(", ".join(new_d)))
            if c.get("impact") and c.get("impact") != p.get("impact"):
                items.append(f"<b>Impact:</b> {esc(c['impact'])}")
            if not items:
                items.append("No visible differences from the current version.")
        self.what.setText("<ul style='margin:0 0 0 -20px'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")

        notes = []
        for e in c.get("evidence") or []:
            if e.get("supported"):
                notes.append(f"<span style='color:{theme.c('secondary')}'>Heard at {fmt_ts(e.get('start', 0))}: "
                             f"“{esc(e.get('quote', ''))}”</span>")
            else:
                notes.append(f"<span style='color:{theme.c('danger')}'>Not found in the transcript: “{esc(e.get('quote', ''))}”"
                             "</span>")
        if c.get("excerpts") and action != "archive":
            notes.append(f"<span style='color:{theme.c('secondary')}'>{len(c['excerpts'])} line(s) of what was said are saved "
                         "with it, so you can later ask why.</span>")
        if not any(e.get("supported") for e in c.get("evidence") or []) and not c.get("user_edited"):
            notes.append(f"<span style='color:{theme.c('danger')}'>This wasn't clearly said in the meeting, so it starts "
                         "switched off. Tick “Include this change” only if it is right.</span>")
        for f in c.get("flags") or []:
            if f.startswith("Uncertain:"):
                notes.append(f"<span style='color:{theme.c('warning')}'>The AI was unsure: {esc(f[10:].strip())}</span>")
            elif not f.startswith(HIDDEN_FLAG_PREFIXES):
                notes.append(f"<span style='color:{theme.c('warning')}'>{esc(f)}</span>")
        self.notes.setText("<br>".join(notes))
        self.notes.setVisible(bool(notes))
        for w in (self.what, self.qbox, self.toggle):
            w.setEnabled(on)
        self.form_w.setEnabled(on and action != "archive")


class TranscriptDialog(QDialog):
    def __init__(self, text: str, parent=None, can_rerun: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Meeting transcript (temporary)")
        self.resize(820, 600)
        self.rerun = False
        lay = QVBoxLayout(self)
        info = QLabel("This transcript is temporary. It is deleted automatically after you save the meeting to "
                      "the project. You can fix wrong words here. Lines starting with “?” were hard to hear.")
        info.setWordWrap(True)
        info.setObjectName("muted")
        lay.addWidget(info)
        self.edit = QPlainTextEdit(text)
        lay.addWidget(self.edit, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Close")
        save = QPushButton("Save corrections and re-read the meeting")
        save.setObjectName("primary")
        save.setEnabled(can_rerun)
        row.addWidget(cancel)
        row.addWidget(save)
        lay.addLayout(row)
        cancel.clicked.connect(self.reject)
        save.clicked.connect(self._save)

    def _save(self):
        self.rerun = True
        self.accept()


class ReviewPanel(QWidget):
    """Emits (action, session_id, payload); the main window runs them."""
    request = Signal(str, str, object)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.sid: str | None = None
        self.proposal: dict | None = None
        self.cards: list[ChangeCard] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        self.h = QLabel()
        self.h.setObjectName("h1")
        self.sub = QLabel()
        self.sub.setObjectName("muted")
        self.sub.setWordWrap(True)
        lay.addWidget(self.h)
        lay.addWidget(self.sub)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)

        # page 0: nothing selected
        self.p_empty = self._message_page("Nothing selected",
                                          "Meetings you add appear in the list on the left. Click one to review it.")
        # page 1: working
        self.p_busy = self._message_page("Working on it…", "")
        self.live_view = QPlainTextEdit()
        self.live_view.setReadOnly(True)
        self.live_view.setVisible(False)
        self.p_busy.layout().insertWidget(2, self.live_view, 1)
        # page 2: problem / transcript only
        self.p_problem = QWidget()
        pl = QVBoxLayout(self.p_problem)
        self.problem_title = QLabel()
        self.problem_title.setObjectName("h2")
        self.problem_text = QLabel()
        self.problem_text.setWordWrap(True)
        pl.addWidget(self.problem_title)
        pl.addWidget(self.problem_text)
        prow = QHBoxLayout()
        self.b_retry = QPushButton("Try again")
        self.b_retry.setObjectName("primary")
        self.b_tr1 = QPushButton("View / fix transcript")
        self.b_del1 = QPushButton("Delete this meeting")
        self.b_del1.setObjectName("danger")
        self.b_cleanup = QPushButton("Retry deleting temporary files")
        self.b_cleanup.setObjectName("primary")
        for b in (self.b_retry, self.b_cleanup, self.b_tr1, self.b_del1):
            prow.addWidget(b)
        prow.addStretch(1)
        pl.addLayout(prow)
        pl.addStretch(1)
        # page 3: review
        self.p_review = QWidget()
        rl = QVBoxLayout(self.p_review)
        rl.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setSpacing(12)

        s1 = QFrame()
        s1.setObjectName("section")
        s1l = QVBoxLayout(s1)
        t1 = QLabel("Step 1 · Check the summary")
        t1.setObjectName("h2")
        s1l.addWidget(t1)
        f1 = QFormLayout()
        self.m_title = QLineEdit()
        self.m_summary = QPlainTextEdit()
        self.m_summary.setFixedHeight(84)
        self.m_items = QLabel()
        self.m_items.setWordWrap(True)
        self.m_items.setTextFormat(Qt.RichText)
        f1.addRow("Meeting name", self.m_title)
        f1.addRow("Summary", self.m_summary)
        s1l.addLayout(f1)
        s1l.addWidget(self.m_items)
        il.addWidget(s1)

        self.t2 = QLabel()
        self.t2.setObjectName("h2")
        il.addWidget(self.t2)
        h2 = QLabel("Each box is one change the app will make to the project. Untick any change that is "
                    "wrong, and answer the yellow questions. Click “Edit details” to correct a change.")
        h2.setWordWrap(True)
        h2.setObjectName("muted")
        il.addWidget(h2)
        self.cards_box = QVBoxLayout()
        self.cards_box.setSpacing(10)
        il.addLayout(self.cards_box)
        self.b_add_card = QPushButton("+ Add a change the AI missed")
        self.b_add_card.setObjectName("link")
        il.addWidget(self.b_add_card, 0, Qt.AlignLeft)
        il.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(inner)
        rl.addWidget(sc, 1)

        bar = QFrame()
        bar.setObjectName("section")
        bl = QHBoxLayout(bar)
        self.step3 = QLabel("Step 3 ·")
        self.step3.setObjectName("h2")
        self.b_save = QPushButton("Save to project")
        self.b_save.setObjectName("good")
        self.b_discard = QPushButton("Discard these changes")
        self.b_more = QPushButton("More ▾")
        menu = QMenu(self)
        menu.addAction("View / fix transcript", self._transcript)
        menu.addAction("Check again against the project", self._rematch)
        menu.addAction("Delete this meeting", self._delete)
        self.b_more.setMenu(menu)
        self.status_line = QLabel()
        self.status_line.setWordWrap(True)
        bl.addWidget(self.step3)
        bl.addWidget(self.b_save)
        bl.addWidget(self.status_line, 1)
        bl.addWidget(self.b_discard)
        bl.addWidget(self.b_more)
        rl.addWidget(bar)

        for p in (self.p_empty, self.p_busy, self.p_problem, self.p_review):
            self.stack.addWidget(p)

        self.b_retry.clicked.connect(self._retry)
        self.b_tr1.clicked.connect(self._transcript)
        self.b_del1.clicked.connect(self._delete)
        self.b_cleanup.clicked.connect(lambda: self.request.emit("cleanup", self.sid, None))
        self.b_save.clicked.connect(self._save)
        self.b_discard.clicked.connect(self._discard_changes)
        self.b_add_card.clicked.connect(self._add_card)
        self.clear()

    def _message_page(self, title: str, text: str) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        t = QLabel(title)
        t.setObjectName("h2")
        b = QLabel(text)
        b.setWordWrap(True)
        b.setObjectName("muted")
        l.addWidget(t)
        l.addWidget(b)
        l.addStretch(1)
        w.body = b
        return w

    # ---------- loading ----------
    def clear(self):
        self.sid = None
        self.proposal = None
        self.h.setText("Review")
        self.sub.setText("")
        self.stack.setCurrentWidget(self.p_empty)

    def load(self, sid: str):
        sm = self.ctx.sm
        try:
            s = sm.load(sid)
        except OSError:
            self.clear()
            return
        self.sid = sid
        stage = s.get("stage", "")
        self.h.setText(s["title"])
        labels = s.get("labels") or {}
        self.sub.setText(f"Project: {s.get('project_name', '')}  ·  {s.get('date', '')}  ·  "
                         f"{STAGE_TEXT.get(stage, stage)}  ·  AI: {labels.get('llm', '')}")
        if stage in ("recording", "transcribing", "summarizing", "saving", "recorded"):
            msg = {"recording": "This meeting is being recorded. Press Stop in the recording window when it ends.",
                   "recorded": "Recording finished. Getting ready to read it…",
                   "transcribing": "Turning the audio into text on this laptop. This can take a minute or two.",
                   "summarizing": "The AI is reading the meeting and comparing it with your project. "
                                  "This usually takes 10–60 seconds.",
                   "saving": "Saving to the project…"}[stage]
            live = s.get("live") and stage in ("recording", "recorded", "transcribing")
            if live:
                st = s.get("live_stats") or {}
                msg = ("Recording. The transcript below grows every 30 seconds"
                       + (f"; notes are ready for {st.get('notes_done', 0)} part(s) so far" if st.get("notes_done")
                          else "") + ".") if stage == "recording" else \
                    "Recording stopped. Finishing the last seconds of the transcript and the notes…"
            self.p_busy.body.setText(msg + "\n\nYou can keep using the app meanwhile.")
            self.live_view.setVisible(bool(live))
            if live:
                bar = self.live_view.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 4
                self.live_view.setPlainText(sm.transcript(sid) or "(the first lines appear after about 30 seconds)")
                if at_end:
                    bar.setValue(bar.maximum())
            self.stack.setCurrentWidget(self.p_busy)
            return
        if stage in ("failed", "transcribed", "cleanup_failed"):
            self.b_retry.setVisible(stage != "cleanup_failed")
            self.b_cleanup.setVisible(stage == "cleanup_failed")
            self.b_tr1.setVisible(stage != "cleanup_failed")
            if stage == "failed":
                self.problem_title.setText("Something went wrong")
                self.problem_text.setText((s.get("error") or "") +
                                          "\n\nClick “Try again”. If it keeps failing, view the transcript to check it "
                                          "has text, or delete this meeting.")
                self.b_retry.setText("Try again")
            elif stage == "transcribed":
                self.problem_title.setText("No suggested changes yet")
                self.problem_text.setText((s.get("note") or "The transcript is ready.") +
                                          "\n\nClick “Read the meeting again” to get suggestions, or delete it.")
                self.b_retry.setText("Read the meeting again")
            else:
                self.problem_title.setText("Saved — but temporary files are still on disk")
                self.problem_text.setText(s.get("error", ""))
            self.stack.setCurrentWidget(self.p_problem)
            return

        # ready
        self.proposal = sm.proposal(sid)
        if not self.proposal:
            self.problem_title.setText("No suggested changes")
            self.problem_text.setText("Click “Read the meeting again”.")
            self.b_retry.setText("Read the meeting again")
            self.stack.setCurrentWidget(self.p_problem)
            return
        m = self.proposal.get("meeting", {})
        self.m_title.setText(m.get("title", ""))
        self.m_summary.setPlainText(m.get("summary", ""))
        parts = []
        for key, name in (("decisions", "Decisions"), ("action_items", "Action items"),
                          ("open_questions", "Open questions")):
            items = m.get(key) or []
            if items:
                lis = "".join(f"<li>{esc(x['text'])}" + (f" <i>({esc(x['owner'])})</i>" if x.get("owner") else "")
                              + "</li>" for x in items)
                parts.append(f"<b>{name}</b><ul style='margin-top:2px'>{lis}</ul>")
        self.m_items.setText("".join(parts) or f"<span style='color:{theme.c('secondary')}'>No decisions, action items or "
                                               "questions were found.</span>")
        while self.cards_box.count():
            it = self.cards_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.cards = []
        pid = self.proposal.get("project_id")
        feats = self.ctx.store.features(pid) if pid else []
        for c in self.proposal.get("changes", []):
            self._append_card(c, feats)
        self._update_status()
        self.stack.setCurrentWidget(self.p_review)

    def _append_card(self, c: dict, feats: list[dict]):
        card = ChangeCard(c, feats)
        card.changed.connect(self._update_status)
        card.policy.connect(self._set_policy)
        self.cards.append(card)
        self.cards_box.addWidget(card)

    def _set_policy(self, key):
        if self.proposal is not None:
            self.proposal["archive_policy"] = key

    def _update_status(self):
        n_on = sum(1 for c in self.cards if c.include.isChecked())
        self.t2.setText(f"Step 2 · Check the changes ({n_on} of {len(self.cards)} included)")
        open_q = [c for c in self.cards if c.unanswered()]
        if open_q:
            self.status_line.setText(f"<span style='color:{theme.c('warning')}'>Answer {len(open_q)} yellow question(s) "
                                     "before saving.</span>")
        else:
            self.status_line.setText(f"<span style='color:{theme.c('success')}'>Ready to save.</span>")

    # ---------- actions ----------
    def current_proposal(self) -> dict | None:
        if not self.proposal:
            return None
        p = json.loads(json.dumps(self.proposal))
        m = p.setdefault("meeting", {})
        m["title"] = self.m_title.text().strip() or "Meeting"
        m["summary"] = self.m_summary.toPlainText().strip()
        p["changes"] = [c.value() for c in self.cards]
        return p

    def _save(self):
        p = self.current_proposal()
        if not p:
            return
        open_q = [c.value().get("title") for c in self.cards if c.unanswered()]
        if open_q:
            QMessageBox.information(self, "One more thing",
                                    "Please answer the yellow question on: " + ", ".join(open_q))
            return
        self.ctx.sm.save_proposal(self.sid, p)
        plan = self.ctx.sm.preview(self.sid, p)
        if plan.errors:
            QMessageBox.warning(self, "Please fix before saving",
                                "These changes can't be saved yet:\n\n• " + "\n• ".join(plan.errors))
            return
        lines = [d for a, f, d in plan.preview if not f.startswith(("project.json", "MASTER_LOG"))]
        box = QMessageBox(self)
        box.setWindowTitle("Save to project?")
        box.setText(f"This will update the project “{p.get('project_name', '')}”:")
        box.setInformativeText("• " + "\n• ".join(lines) +
                               "\n\nThe temporary transcript is deleted after saving.")
        save = box.addButton("Save to project", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is save:
            self.request.emit("approve", self.sid, p)

    def _discard_changes(self):
        if QMessageBox.question(self, "Discard these changes?",
                                "The project stays exactly as it is. The meeting stays in the list so you can "
                                "read it again or delete it.") == QMessageBox.Yes:
            self.request.emit("reject", self.sid, None)

    def _delete(self):
        if QMessageBox.question(self, "Delete this meeting?",
                                "This deletes the meeting's temporary transcript, audio and suggestions. "
                                "Your project is not changed.") == QMessageBox.Yes:
            self.request.emit("discard", self.sid, None)

    def _rematch(self):
        if QMessageBox.question(self, "Check again?",
                                "Rebuild the suggested changes against the project as it is now? "
                                "Your edits to the boxes are lost.") == QMessageBox.Yes:
            self.request.emit("rematch", self.sid, None)

    def _retry(self):
        s = self.ctx.sm.load(self.sid)
        audio_only = s.get("audio") and not self.ctx.sm.transcript(self.sid).strip()
        self.request.emit("process" if audio_only else "extract", self.sid, None)

    def _transcript(self):
        if not self.sid:
            return
        d = TranscriptDialog(self.ctx.sm.transcript(self.sid), self)
        if d.exec() and d.rerun:
            self.request.emit("save_transcript", self.sid, d.edit.toPlainText())
            self.request.emit("extract", self.sid, None)

    def _add_card(self):
        if not self.proposal:
            return
        c = {"change_id": f"m{len(self.cards) + 1}", "include": True, "action": "add", "feature_id": None,
             "title": "New feature", "description": "", "status": "proposed", "subtasks": [], "depends_on": [],
             "impact": "", "evidence": [], "flags": ["Added by you (not from the transcript)."],
             "user_edited": True}
        pid = self.proposal.get("project_id")
        self._append_card(c, self.ctx.store.features(pid) if pid else [])
        self.cards[-1]._toggle()
        self._update_status()
