"""Language-model adapters. All are local; none calls a cloud service.

- HeuristicLLM: rule-based stand-in so the whole app runs with no model installed.
- OllamaLLM: a local Ollama server on 127.0.0.1 (dev laptop convenience, CPU/GPU).
- ExecuTorchRunnerLLM: runs an ExecuTorch runner executable (the Snapdragon NPU path).
  The command is configured after the model gate on the Dell; nothing is guessed here.
"""
from __future__ import annotations

import atexit
import json
import queue
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


# The app runs under pythonw (no console). Without this flag Windows opens a terminal window for every
# runner .exe it starts (one per live-transcription batch while recording).
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class LLMError(RuntimeError):
    pass


class BaseLLM:
    name = "base"
    heuristic = False

    def __init__(self, label: str = ""):
        self._label = label

    @property
    def label(self) -> str:
        return self._label or self.default_label()

    def default_label(self) -> str:
        return self.name

    def generate(self, prompt: str, max_tokens: int = 1024, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        raise NotImplementedError

    def health(self) -> tuple[bool, str]:
        """(ready, plain-English message) for the status bar and the startup check."""
        return True, self.label


class HeuristicLLM(BaseLLM):
    """No neural network: extraction uses keyword rules (pipeline/heuristic.py)."""
    name = "heuristic"
    heuristic = True

    def default_label(self) -> str:
        return "Rules (no model)"

    def generate(self, prompt: str, max_tokens: int = 1024, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        raise LLMError("Heuristic backend does not generate text.")


class OllamaLLM(BaseLLM):
    name = "ollama"

    def __init__(self, url: str, model: str, timeout_s: int = 300, label: str = "", num_ctx: int = 8192):
        super().__init__(label)
        if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
            raise LLMError("Ollama URL must be local (127.0.0.1/localhost). No remote endpoints.")
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout_s
        self.num_ctx = num_ctx
        self.last_seconds = 0.0

    def default_label(self) -> str:
        return f"{self.model} via Ollama (local CPU/GPU, not NPU)"

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(f"{self.url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def health(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=3) as r:
                names = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
        except Exception:
            return False, ("Ollama is not running. Start the Ollama app (or run 'ollama serve'), "
                           "then restart MeetingGraph.")
        m = self.model
        found = m in names or any(n.startswith(m + ":") for n in names) or \
            (":" not in m and any(n.split(":")[0] == m for n in names))
        if not found:
            return False, f"Model '{m}' is not downloaded yet. In PowerShell run:  ollama pull {m}"
        return True, self.label

    def generate(self, prompt: str, max_tokens: int = 1024, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1, "num_ctx": self.num_ctx}}
        if schema is not None:
            body["format"] = schema          # Ollama structured output: constrains JSON to this schema
        elif json_mode:
            body["format"] = "json"
        t0 = time.perf_counter()
        try:
            data = self._post("/api/generate", body)
        except urllib.error.HTTPError as e:
            if schema is not None and e.code == 400:   # older Ollama: no schema support -> plain JSON mode
                body["format"] = "json"
                data = self._post("/api/generate", body)
            else:
                raise LLMError(f"Ollama error {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach local Ollama at {self.url}: {e}. Is Ollama running?") from e
        self.last_seconds = time.perf_counter() - t0
        return data.get("response", "")


def _skeleton(schema: dict):
    """Compact example of a JSON schema's shape, e.g. {"summary": "", "decisions": [{"text": "", ...}]}."""
    t = schema.get("type")
    if t == "object":
        return {k: _skeleton(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_skeleton(schema.get("items", {}))]
    if t in ("number", "integer"):
        return 0
    return ""


# Chat-template markers written around the reply by qnn_llama_runner (Phi-4, Qwen, Llama 3).
_REPLY_START = ("<|assistant|>", "<|im_start|>assistant", "<|start_header_id|>assistant<|end_header_id|>")
_REPLY_END = ("<|end|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")


class _RunnerServer:
    """One qnn_llama_runner --server process: the .pte is loaded once and reused for every request.

    Protocol (one line each): request ``<prompt_file>\\t<seq_len>\\t<output_file>``; reply
    ``OK\\t<prompt_tokens>\\t<generated>\\t<prefill_ms>\\t<total_ms>`` or ``ERR\\t<message>``. The runner also
    prints a stats line (PyTorchObserver) that is skipped. Unloads itself after ``idle_s`` without requests.
    """

    def __init__(self, args: list[str], load_timeout: float, idle_s: float):
        self.args, self.load_timeout, self.idle_s = args, load_timeout, idle_s
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.workdir = Path(tempfile.mkdtemp(prefix="mg_llm_srv_"))
        self.last_used = 0.0
        self.last_stats = ""
        threading.Thread(target=self._idle_watch, daemon=True, name="llm-idle").start()
        atexit.register(self.stop)

    def _reader(self, proc):
        for line in proc.stdout:
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)  # the process exited

    def _next_line(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise LLMError(f"NPU runner did not answer within {int(timeout)} s") from None
        if line is None:
            raise LLMError("NPU runner exited")
        return line

    def _start(self):
        self.stop()
        self.lines = queue.Queue()
        self.proc = subprocess.Popen(self.args, cwd=self.workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                     creationflags=NO_WINDOW)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True, name="llm-out").start()
        deadline = time.monotonic() + self.load_timeout
        while True:
            line = self._next_line(max(1.0, deadline - time.monotonic()))
            if line == "READY":
                return
            if line.startswith("ERR"):
                raise LLMError(f"NPU runner could not load the model: {line[4:]}")

    def stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            if p.poll() is None:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
                p.wait(10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            p.kill()
        finally:
            for f in (p.stdin, p.stdout):   # close pipes of a dead runner quietly (no "invalid argument" on GC)
                try:
                    f.close()
                except (OSError, ValueError):
                    pass

    def _idle_watch(self):
        while True:
            time.sleep(15)
            if self.proc and self.idle_s > 0 and time.monotonic() - self.last_used > self.idle_s \
                    and self.lock.acquire(blocking=False):
                try:
                    self.stop()
                finally:
                    self.lock.release()

    def _once(self, prompt_file: Path, seq_len: int, out_file: Path, timeout: float) -> str:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(f"{prompt_file}\t{seq_len}\t{out_file}\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as e:     # the runner died between requests
            raise LLMError(f"NPU runner exited: {e}") from e
        while True:
            line = self._next_line(timeout)
            if line.startswith("OK"):
                self.last_stats = line
                return out_file.read_text(encoding="utf-8", errors="replace")
            if line.startswith("ERR"):
                raise LLMError(f"NPU runner: {line[4:]}")

    def generate(self, prompt: str, seq_len: int, timeout: float) -> str:
        with self.lock:
            self.last_used = time.monotonic()
            pf, of = self.workdir / "prompt.txt", self.workdir / "output.txt"
            pf.write_text(prompt, encoding="utf-8")
            of.unlink(missing_ok=True)
            try:
                return self._once(pf, seq_len, of, timeout)
            except LLMError:
                # A crashed, hung or unloaded runner: start a fresh one and try once more.
                self.stop()
                of.unlink(missing_ok=True)
                return self._once(pf, seq_len, of, timeout)
            finally:
                self.last_used = time.monotonic()


class ExecuTorchRunnerLLM(BaseLLM):
    """Runs an ExecuTorch LLM on the NPU. Two modes:

    - qnn_llama_runner (set ``runner`` + ``model_path`` + ``tokenizer_path``): builds the command itself,
      sizes --seq_len to each prompt, adds an explicit JSON-shape instruction (the runner has no
      constrained decoding) and returns only the model's reply. The model runs on the Hexagon NPU;
      the CPU only tokenizes and drives the token loop.
    - generic ``command`` with placeholders {prompt_file} {prompt} {output_file} {max_tokens}.
    """
    name = "executorch_runner"
    CHARS_PER_TOKEN = 3.5  # conservative, so a prompt never overruns the compiled context

    def __init__(self, command: list[str], cwd: str = "", timeout_s: int = 300,
                 verified_on_npu: bool = False, label: str = "", runner: str = "", model_path: str = "",
                 tokenizer_path: str = "", decoder_model_version: str = "phi_4_mini", max_seq_len: int = 2048,
                 eval_mode: int | None = None, server_runner: str = "", keep_loaded_s: float = 300):
        super().__init__(label)
        self.runner = runner
        if runner:
            for p in (runner, model_path, tokenizer_path):
                if not Path(p).exists():
                    raise LLMError(f"ExecuTorch file not found: {p}")
        elif not command:
            raise LLMError("Set llm.executorch_runner.runner/model_path/tokenizer_path (or command) in config.yaml.")
        self.command = command
        self.model_path, self.tokenizer_path = model_path, tokenizer_path
        self.decoder_model_version, self.max_seq_len = decoder_model_version, max_seq_len
        # 0 = kv (one graph), 1 = hybrid (fast prompt-reading graph + kv). Default from the .pte name.
        self.eval_mode = eval_mode if eval_mode is not None else int(Path(model_path).name.startswith("hybrid"))
        self.server = None
        if runner and server_runner:
            if not Path(server_runner).exists():
                raise LLMError(f"ExecuTorch file not found: {server_runner}")
            self.server = _RunnerServer(
                [server_runner, "--server", "--model_path", model_path, "--tokenizer_path", tokenizer_path,
                 "--decoder_model_version", decoder_model_version, "--eval_mode", str(self.eval_mode),
                 "--temperature", "0", "--seq_len", str(max_seq_len), "--output_path", "unused.txt"],
                load_timeout=min(timeout_s, 180), idle_s=keep_loaded_s)
        self.cwd = cwd or None
        self.timeout = timeout_s
        self.verified = verified_on_npu
        self.last_seconds = 0.0

    def default_label(self) -> str:
        what = Path(self.model_path).parent.name if self.runner else "ExecuTorch runner"
        return f"{what} on the NPU (verified)" if self.verified else f"{what} (NPU not yet verified)"

    def _qnn_generate(self, prompt: str, max_tokens: int, json_mode: bool, schema: dict | None) -> str:
        if schema is not None:
            prompt += "\n\nReply with ONLY one JSON object in exactly this shape:\n" + \
                      json.dumps(_skeleton(schema), separators=(",", ":"))
        elif json_mode:
            prompt += "\n\nReply with ONLY one JSON object."
        if self.decoder_model_version.startswith("qwen3"):
            prompt += " /no_think"  # otherwise Qwen3 spends the reply budget inside <think>
        prompt_tokens = int(len(prompt) / self.CHARS_PER_TOKEN) + 64  # 64: chat-template overhead
        room = self.max_seq_len - prompt_tokens
        if room < 64:
            raise LLMError(f"Prompt (~{prompt_tokens} tokens) does not fit the model's {self.max_seq_len}-token "
                           "context. Lower llm.max_transcript_chars_per_chunk in config.yaml.")
        seq_len = prompt_tokens + min(max_tokens, room)
        if self.server is not None:
            t0 = time.perf_counter()
            text = self.server.generate(prompt, seq_len, self.timeout)
            self.last_seconds = time.perf_counter() - t0
            return self._reply_only(text)
        with tempfile.TemporaryDirectory(prefix="mg_llm_") as td:
            out = Path(td) / "output.txt"
            args = [self.runner, "--model_path", self.model_path, "--tokenizer_path", self.tokenizer_path,
                    "--decoder_model_version", self.decoder_model_version, "--eval_mode", str(self.eval_mode),
                    "--seq_len", str(seq_len), "--temperature", "0", "--prompt", prompt,
                    "--output_path", str(out), "--performance_output_path", str(Path(td) / "speed.txt")]
            t0 = time.perf_counter()
            try:
                # cwd=td: the runner writes side files into its working directory.
                p = subprocess.run(args, cwd=td, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=self.timeout,
                                   encoding="utf-8", errors="replace")
            except (OSError, subprocess.TimeoutExpired) as e:
                raise LLMError(f"NPU runner failed to start or timed out: {e}") from e
            self.last_seconds = time.perf_counter() - t0
            if p.returncode != 0 or not out.exists():
                raise LLMError(f"NPU runner exited {p.returncode}: {(p.stderr or p.stdout)[-800:]}")
            text = out.read_text(encoding="utf-8", errors="replace")
        return self._reply_only(text)

    @staticmethod
    def _reply_only(text: str) -> str:
        for marker in _REPLY_START:  # keep only the reply, not the echoed chat template
            if marker in text:
                text = text.rsplit(marker, 1)[1]
                break
        for marker in _REPLY_END:
            text = text.replace(marker, "")
        return text.strip()

    def generate(self, prompt: str, max_tokens: int = 1024, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        if self.runner:
            return self._qnn_generate(prompt, max_tokens, json_mode, schema)
        with tempfile.TemporaryDirectory(prefix="mg_llm_") as td:
            pf = Path(td) / "prompt.txt"
            of = Path(td) / "output.txt"
            pf.write_text(prompt, encoding="utf-8")
            args = [a.replace("{prompt_file}", str(pf)).replace("{output_file}", str(of))
                     .replace("{prompt}", prompt).replace("{max_tokens}", str(max_tokens))
                    for a in self.command]
            t0 = time.perf_counter()
            try:
                p = subprocess.run(args, cwd=self.cwd, capture_output=True, text=True, creationflags=NO_WINDOW,
                                   timeout=self.timeout, encoding="utf-8", errors="replace")
            except (OSError, subprocess.TimeoutExpired) as e:
                raise LLMError(f"Runner failed to start or timed out: {e}") from e
            self.last_seconds = time.perf_counter() - t0
            if p.returncode != 0:
                raise LLMError(f"Runner exited {p.returncode}: {p.stderr[-800:]}")
            out = of.read_text(encoding="utf-8", errors="replace") if of.exists() else p.stdout
            # Many runners echo the prompt; keep only what follows it.
            if prompt and prompt in out:
                out = out.split(prompt, 1)[1]
            return out.strip()


def make_llm(cfg: dict) -> BaseLLM:
    c = cfg["llm"]
    b = c.get("backend", "heuristic")
    label = c.get("label", "")
    if b == "heuristic":
        return HeuristicLLM(label)
    if b == "ollama":
        o = c["ollama"]
        return OllamaLLM(o["url"], o["model"], o.get("timeout_s", 300), label, o.get("num_ctx", 8192))
    if b == "executorch_runner":
        e = c["executorch_runner"]
        return ExecuTorchRunnerLLM(e.get("command") or [], e.get("cwd", ""), e.get("timeout_s", 300),
                                   e.get("verified_on_npu", False), label, e.get("runner", ""),
                                   e.get("model_path", ""), e.get("tokenizer_path", ""),
                                   e.get("decoder_model_version", "phi_4_mini"), e.get("max_seq_len", 2048),
                                   e.get("eval_mode"), e.get("server_runner", ""), e.get("keep_loaded_s", 300))
    raise LLMError(f"Unknown llm.backend {b!r}")
