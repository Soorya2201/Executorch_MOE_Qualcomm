"""App's real extraction with a chosen NPU model.  bench_meeting_notes.py {qwen17|phi|config} FIXTURE MAX_CHARS  (phi/config = whatever config.yaml says)"""
import sys, time
from pathlib import Path
APP = Path(__file__).resolve().parents[2] / "meetinggraph"; sys.path.insert(0, str(APP))
from app.config import load_config
from app.models.llm import make_llm
from app.pipeline import extract as ex
from app.pipeline.transcript import from_text

which, fixture, max_chars = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = load_config(APP / "config.yaml"); e = cfg["llm"]["executorch_runner"]
if which == "qwen17":
    R, M = "C:/et/llm/npu-run-246", "C:/et/llm/qwen3_1_7b_ctx8k"
    e.update(runner=f"{R}/qnn_llama_runner.exe", server_runner=f"{R}/qnn_llama_runner_server.exe",
             model_path=f"{M}/sc8380_qwen3-1_7b_ctxt-8000.pte", tokenizer_path=f"{M}/tokenizer.json",
             decoder_model_version="qwen3", max_seq_len=8000, eval_mode=1)
llm = make_llm(cfg)
segs = from_text((APP / "tests" / "fixtures" / fixture).read_text(encoding="utf-8"))
chars = sum(len(s["text"]) for s in segs)
calls = []; orig = llm.generate
def timed(*a, **k):
    t = time.time(); r = orig(*a, **k); calls.append(time.time() - t); return r
llm.generate = timed
t0 = time.time()
d = ex.extract_draft(llm, segs, Path(fixture).stem, [], max_chars)
print(f"\n{which} | {fixture} | {chars} chars | TOTAL {time.time()-t0:.0f} s | requests {[round(c) for c in calls]}")
print("SUMMARY:", d["summary"])
for k in ("decisions", "action_items", "open_questions"):
    print(f"{k}:", [(i["text"], i.get("owner", "")) for i in d[k]])
print("features:", [(f["title"], sum(1 for q in f["evidence"] if q.get("supported")), len(f["evidence"])) for f in d["feature_mentions"]])
print("dropped:", d.get("dropped_features", []))
if llm.server: llm.server.stop()
