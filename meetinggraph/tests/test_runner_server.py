"""The keep-loaded NPU runner client (_RunnerServer), against a fake runner speaking the same stdin protocol."""
import sys
from pathlib import Path

import pytest

from app.models.llm import LLMError, _RunnerServer

FAKE = r'''
import sys
from pathlib import Path
crash_marker = Path(sys.argv[1])
print("READY", flush=True)
for line in sys.stdin:
    line = line.rstrip("\r\n")
    if line == "QUIT":
        break
    prompt_file, seq_len, out_file = line.split("\t")
    text = Path(prompt_file).read_text(encoding="utf-8")
    if text == "crash" and not crash_marker.exists():
        crash_marker.write_text("1")
        sys.exit(3)                                   # die mid-request, once
    Path(out_file).write_text("<|assistant|>echo:" + text + "<|end|>", encoding="utf-8")
    print('PyTorchObserver {"prefill_token_per_sec": 1}', flush=True)
    print(f"OK\t{len(text)}\t3\t10\t20", flush=True)
'''


@pytest.fixture
def server(tmp_path):
    fake = tmp_path / "fake_runner.py"
    fake.write_text(FAKE, encoding="utf-8")
    s = _RunnerServer([sys.executable, str(fake), str(tmp_path / "crashed_once")], load_timeout=30, idle_s=0)
    yield s
    s.stop()


def test_reuses_one_process(server):
    assert "echo:first" in server.generate("first", 64, 30)
    pid = server.proc.pid
    assert "echo:second" in server.generate("second", 64, 30)
    assert server.proc.pid == pid                     # model stayed loaded
    assert server.last_stats.startswith("OK\t6\t")    # the stats line was skipped


def test_restarts_after_crash(server):
    assert "echo:crash" in server.generate("crash", 64, 30)   # first process died; retried on a fresh one


def test_reports_runner_errors(tmp_path):
    bad = tmp_path / "bad_runner.py"
    bad.write_text('print("ERR\\tmodel load failed", flush=True)\n', encoding="utf-8")
    s = _RunnerServer([sys.executable, str(bad)], load_timeout=30, idle_s=0)
    with pytest.raises(LLMError, match="could not load"):
        s.generate("x", 64, 30)
    s.stop()
