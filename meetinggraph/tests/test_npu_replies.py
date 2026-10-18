"""Reply shapes seen from Phi-4-mini on the NPU: a bare list instead of the wrapper object, plain-text quotes."""
from app.pipeline import extract as ex, prompts


class OneReply:
    heuristic = False

    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, max_tokens=1024, json_mode=False, schema=None):
        return self.reply


BARE = """[
  {"title": "Conflict review", "action": "add", "description": "Pick between two versions.",
   "evidence": ["let the user pick"]},
  {"title": "Offline sync", "action": "change", "description": "Sync when back online.", "evidence": []}
]"""


def test_bare_list_is_read_as_feature_mentions():
    obj = ex._ask_json(OneReply(BARE), "p", 100, prompts.FEATURES_SCHEMA)
    assert [f["title"] for f in obj["feature_mentions"]] == ["Conflict review", "Offline sync"]


def test_cut_off_bare_list_is_salvaged():
    obj = ex._ask_json(OneReply(BARE[:BARE.index("Offline") + 20]), "p", 100, prompts.FEATURES_SCHEMA)
    assert obj["feature_mentions"][0]["title"] == "Conflict review"


def test_plain_text_evidence_kept():
    d = ex._normalize_draft({"feature_mentions": [{"title": "X", "evidence": ["said it", {"quote": "q"}]}]})
    assert [e["quote"] for e in d["feature_mentions"][0]["evidence"]] == ["said it", "q"]


def test_object_reply_unchanged():
    obj = ex._ask_json(OneReply('{"summary": "s", "decisions": []}'), "p", 100, prompts.NOTES_SCHEMA)
    assert obj["summary"] == "s"
