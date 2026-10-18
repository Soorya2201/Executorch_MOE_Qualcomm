"""Calibration samples that look like what MeetingGraph actually sends Phi-4-mini on the NPU.

v1 was calibrated on 8 generic "continue this encyclopedia article" samples. The app instead sends chat
instructions with meeting transcripts and expects JSON answers. This builds, with the app's own code:
  - the exact NOTES and compact FEATURES prompts (same chunk sizing, same JSON-shape instruction the NPU adapter
    appends) for 3 held-out fixture meetings + 3 synthetic meetings written by Phi itself,
  - a MERGE_SUMMARY prompt,
  - answers from the current NPU Phi (calibration needs realistic activations, not perfect answers),
  - plus 4 of the original general samples so calibration stays general.
Held out from calibration: atlas_03 and biology_01 (used to evaluate the result), and the 10-prompt suite.

  C:\\et\\Executorch_MOE_Qualcomm\\meetinggraph\\.venv\\Scripts\\python.exe C:\\et\\llm\\build_calib_meeting.py
"""
import json
import sys
import time
from pathlib import Path

APP = Path(r"C:\et\mg_upstream\meetinggraph")
sys.path.insert(0, str(APP))

from app.config import load_config  # noqa: E402
from app.models.llm import _skeleton, make_llm  # noqa: E402
from app.pipeline import extract as ex, prompts  # noqa: E402
from app.pipeline.transcript import chunk_lines, from_text, to_text  # noqa: E402

OUT = Path(r"C:\et\llm\calib_mix_16.json")
GENERAL = Path(r"C:\et\llm\calib_text_8.json")
CALIB_FIXTURES = ("atlas_01_kickoff.txt", "atlas_02_sprint_sync.txt", "website_01_planning.txt")
SYNTHETIC = (
    "a hiring sync where a small startup team decides which of three candidates gets an offer, who calls "
    "references, and when the offer goes out",
    "a university group project meeting about a robotics assignment: splitting the work, the sensor that "
    "is not working, and the demo date",
    "a hardware launch review for a smart speaker: the delayed battery supplier, the firmware freeze, the "
    "marketing date, and an open question about the price",
)


def adapter_text(prompt: str, schema: dict) -> str:
    """Exactly what the NPU adapter sends (ExecuTorchRunnerLLM._qnn_generate)."""
    return prompt + "\n\nReply with ONLY one JSON object in exactly this shape:\n" + \
        json.dumps(_skeleton(schema), separators=(",", ":"))


def sample(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def main():
    cfg = load_config(APP / "config.yaml")
    llm = make_llm(cfg)
    print("model:", llm.label, flush=True)
    t0 = time.time()
    transcripts = [(APP / "tests" / "fixtures" / f).read_text(encoding="utf-8") for f in CALIB_FIXTURES]
    for topic in SYNTHETIC:
        text = llm.generate(
            f"Write a realistic transcript of a 3-minute meeting: {topic}. Use 3 or 4 named speakers, one line "
            "per turn in the form 'Name: what they said'. Include concrete names, dates and numbers, at least one "
            "decision, two action items with owners, and one question left open. Only the transcript.",
            max_tokens=700)
        lines = [ln.strip() for ln in text.splitlines() if ":" in ln and len(ln) > 8]
        transcripts.append("\n".join(lines))
        print(f"synthetic meeting: {len(lines)} lines ({time.time() - t0:.0f} s)", flush=True)

    samples, summaries = [], []
    for i, text in enumerate(transcripts):
        segs = from_text(text)
        chunk_chars, feat_t, feat_s, notes_out, feat_out = ex.plan_requests(llm, 2500, "- (none yet)")
        chunk = chunk_lines(to_text(segs), chunk_chars)[0]
        notes_user = adapter_text(prompts.NOTES.format(transcript=chunk), prompts.NOTES_SCHEMA)
        notes = llm.generate(prompts.NOTES.format(transcript=chunk), max_tokens=notes_out,
                             json_mode=True, schema=prompts.NOTES_SCHEMA)
        samples.append(sample(notes_user, notes))
        try:
            summaries.append(json.loads(notes[notes.index("{"):notes.rindex("}") + 1]).get("summary", ""))
        except (ValueError, AttributeError):
            pass
        feat_prompt = feat_t.format(features="- (none yet)", transcript=chunk, max_items=5)
        feats = llm.generate(feat_prompt, max_tokens=feat_out, json_mode=True, schema=feat_s)
        samples.append(sample(adapter_text(feat_prompt, feat_s), feats))
        print(f"meeting {i + 1}/{len(transcripts)}: notes {len(notes)} chars, features {len(feats)} chars "
              f"({time.time() - t0:.0f} s)", flush=True)
    if summaries:
        parts = "\n\n".join(f"Part {i + 1}: {s}" for i, s in enumerate(summaries[:3]) if s)
        merge_user = prompts.MERGE_SUMMARY.format(parts=parts)
        samples.append(sample(merge_user, llm.generate(merge_user, max_tokens=300, json_mode=True,
                                                       schema=prompts.SUMMARY_SCHEMA)))
    general = json.loads(GENERAL.read_text(encoding="utf-8"))[:4]
    samples += general
    OUT.write_text(json.dumps(samples, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}: {len(samples)} samples ({len(samples) - len(general)} meeting, {len(general)} general) "
          f"in {time.time() - t0:.0f} s", flush=True)
    if getattr(llm, "server", None):
        llm.server.stop()


if __name__ == "__main__":
    main()
