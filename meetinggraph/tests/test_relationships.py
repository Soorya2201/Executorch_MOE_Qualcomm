"""Rich feature notes + "why" questions about how two features are connected.

The fake model plays the part of Qwen with the new prompt: longer descriptions, key points and
connections that carry a reason. Real Qwen output varies; these tests check everything around it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import load_config
from app.models.asr import NoASR
from app.models.llm import BaseLLM, HeuristicLLM
from app.pipeline.query import IDK, answer, mentioned_features
from app.pipeline.session import SessionManager
from app.workspace.graph import build_graph
from app.workspace.paths import Layout
from app.workspace.store import Store

FIX = Path(__file__).parent / "fixtures"

NOTES = {"summary": "A lecture on the parts of the cell and how they work together.",
         "decisions": [], "action_items": [{"text": "Read chapter four", "owner": "",
                                            "quote": "read chapter four on membrane transport"}],
         "open_questions": []}

FEATURES = {"feature_mentions": [
    {"title": "Protein synthesis", "action": "add",
     "description": "Protein synthesis is how the cell builds proteins from the instructions in DNA. The "
                    "instructions are copied into messenger RNA, which ribosomes read to join amino acids "
                    "together. It uses a lot of energy because every amino acid added costs ATP.",
     "key_points": ["DNA instructions are copied into messenger RNA, which leaves through the nuclear pores.",
                    "Every amino acid that is added costs energy."],
     "relations": [
         {"to": "Ribosomes", "type": "needs",
          "why": "Ribosomes read the messenger RNA and join the amino acids, so without them DNA never becomes protein."},
         {"to": "Mitochondria", "type": "needs",
          "why": "It needs a lot of ATP, and mitochondria supply that ATP."}],
     "status": "", "subtasks": [],
     "evidence": [{"quote": "so protein synthesis depends on ribosomes"},
                  {"quote": "Protein synthesis needs a lot of ATP"}]},
    {"title": "Ribosomes", "action": "add",
     "description": "Ribosomes read messenger RNA and join amino acids together into proteins. Many sit on the "
                    "rough endoplasmic reticulum.",
     "key_points": ["The rough endoplasmic reticulum is covered in ribosomes."],
     "relations": [{"to": "Endoplasmic reticulum", "type": "part of",
                    "why": "The rough endoplasmic reticulum is covered in ribosomes."}],
     "status": "", "subtasks": [], "evidence": [{"quote": "Ribosomes read the messenger RNA"}]},
    {"title": "Mitochondria", "action": "add",
     "description": "Mitochondria are the powerhouse of the cell. They break down glucose with oxygen in "
                    "cellular respiration and release the energy as ATP. Muscle cells have thousands of them.",
     "key_points": ["Muscle cells have thousands of mitochondria because contraction uses so much ATP."],
     "relations": [], "status": "", "subtasks": [],
     "evidence": [{"quote": "Mitochondria are the powerhouse of the cell"}]},
    {"title": "Chloroplasts", "action": "add",
     "description": "Chloroplasts capture light energy in photosynthesis and make glucose. Only plant cells "
                    "have them.",
     "key_points": [],
     "relations": [{"to": "Mitochondria", "type": "produces",
                    "why": "The glucose made by chloroplasts is the fuel that mitochondria break down."}],
     "status": "", "subtasks": [], "evidence": [{"quote": "Plant cells also have chloroplasts"}]},
    {"title": "Endoplasmic reticulum", "action": "add",
     "description": "A network of membranes. The rough part is covered in ribosomes; proteins made there are "
                    "folded and sent on to the Golgi apparatus.",
     "key_points": [], "relations": [], "status": "", "subtasks": [],
     "evidence": [{"quote": "The endoplasmic reticulum is a network of membranes"}]},
]}


class LectureLLM(BaseLLM):
    """Returns the notes / features above, and records answer prompts."""
    name = "fake-qwen"

    def __init__(self, answer_text="Because ribosomes do the joining [1]."):
        super().__init__("fake-qwen")
        self.prompts, self.answer_text = [], answer_text

    def generate(self, prompt, max_tokens=1024, json_mode=False, schema=None):
        self.prompts.append(prompt)
        if "list the main ITEMS" in prompt:
            return json.dumps(FEATURES)
        if "write short meeting notes" in prompt:
            return json.dumps(NOTES)
        return self.answer_text


@pytest.fixture()
def bio(tmp_path):
    cfg = load_config()
    cfg["workspace"] = str(tmp_path / "ws")
    layout = Layout(cfg["workspace"])
    store = Store(layout)
    store.init_workspace()
    llm = LectureLLM()
    sm = SessionManager(cfg, layout, store, NoASR(), llm)
    p = sm.create_project("Biology")
    s = sm.import_text((FIX / "biology_01_cell_lecture.txt").read_text(encoding="utf-8"), p["id"], "Biology",
                       "The cell")
    s = sm.extract(s["id"])
    assert s["stage"] == "ready", s.get("error")
    prop = sm.proposal(s["id"])
    r = sm.approve(s["id"], prop)
    assert r["ok"], r
    return store, p["id"], llm, prop


def test_cards_carry_details_reasons_and_transcript_lines(bio):
    store, pid, llm, prop = bio
    ps = next(c for c in prop["changes"] if c["title"] == "Protein synthesis")
    assert len(ps["description"]) > 150 and ps["key_points"]
    assert ps["depends_on"] == ["Ribosomes", "Mitochondria"]
    assert {r["to"]: r["why"] for r in ps["relations"]}["Mitochondria"].startswith("It needs a lot of ATP")
    said = " ".join(e["text"] for e in ps["excerpts"])
    assert "depends on mitochondria" in said and "depends on ribosomes" in said   # transcript, not summary


def test_feature_file_keeps_reasons_and_what_was_said(bio):
    store, pid, *_ = bio
    f = next(x for x in store.features(pid) if x["title"] == "Protein synthesis")
    body = f["_body"]
    assert "## Key points" in body and "why: It needs a lot of ATP" in body
    assert "## What was said" in body and "since they supply that ATP" in body
    ch = next(x for x in store.features(pid) if x["title"] == "Chloroplasts")
    assert "Chloroplasts produces [[" in ch["_body"] and "fuel that mitochondria break down" in ch["_body"]
    # the map shows the connection with its reason
    edges = build_graph(store, pid)["edges"]
    e = next(e for e in edges if e["type"] == "feature_relates_to_feature" and "Chloroplasts" in e["reason"])
    assert "Why: The glucose made by chloroplasts" in e["reason"]


def test_question_names_features():
    feats = [{"id": "1", "title": "Protein synthesis"}, {"id": "2", "title": "Mitochondria"},
             {"id": "3", "title": "Ribosomes"}, {"id": "4", "title": "Cell membrane"}]
    got = [f["title"] for f in mentioned_features("Why does protein synthesis need mitochondria?", feats)]
    assert got[:2] == ["Protein synthesis", "Mitochondria"] and "Ribosomes" not in got
    assert [f["title"] for f in mentioned_features("what does the membrane do", feats)] == ["Cell membrane"]
    assert [f["title"] for f in mentioned_features("how do ribosome work", feats)] == ["Ribosomes"]


def test_why_question_gives_model_the_reason_and_transcript(bio):
    store, pid, llm, _ = bio
    a = answer(store, llm, pid, "Why does protein synthesis depend on mitochondria?")
    p = llm.prompts[-1]
    assert "Connections between the things asked about" in p
    assert "Protein synthesis needs Mitochondria. Why: It needs a lot of ATP" in p
    assert "What was said" in p and "since they supply that ATP" in p
    assert a["citations"] and a["answer"] != IDK


def test_indirect_connection_is_found_through_a_chain(bio):
    store, pid, llm, _ = bio
    answer(store, llm, pid, "How are chloroplasts related to protein synthesis?")
    p = llm.prompts[-1]
    # chloroplasts -> mitochondria -> protein synthesis
    assert "Chloroplasts produces Mitochondria" in p and "Protein synthesis needs Mitochondria" in p


def test_why_question_without_model_returns_saved_reason(bio):
    store, pid, _, _ = bio
    a = answer(store, HeuristicLLM(), pid, "Why does protein synthesis need ribosomes?")
    assert "Why: Ribosomes read the messenger RNA" in a["answer"]
    assert a["citations"]


def test_unrelated_question_still_says_i_dont_know(bio):
    store, pid, llm, _ = bio
    assert answer(store, llm, pid, "When is the pricing page launching?")["answer"] == IDK
