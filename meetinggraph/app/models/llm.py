"""Language-model adapters. All are local; none calls a cloud service.

- HeuristicLLM: rule-based stand-in so the whole app runs with no model installed.
- OllamaLLM: a local Ollama server on 127.0.0.1 (dev laptop convenience, CPU/GPU).
- ExecuTorchRunnerLLM: runs an ExecuTorch runner executable (the Snapdragon NPU path).
  The command is configured after the model gate on the Dell; nothing is guessed here.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


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


class ExecuTorchRunnerLLM(BaseLLM):
    """Calls a configured ExecuTorch runner. Placeholders in the command:
    {prompt_file} {prompt} {output_file}. If {output_file} is absent, stdout is used.
    """
    name = "executorch_runner"

    def __init__(self, command: list[str], cwd: str = "", timeout_s: int = 300,
                 verified_on_npu: bool = False, label: str = ""):
        super().__init__(label)
        if not command:
            raise LLMError("llm.executorch_runner.command is empty in config.yaml. "
                           "Fill it in after the model gate on the Snapdragon laptop.")
        self.command = command
        self.cwd = cwd or None
        self.timeout = timeout_s
        self.verified = verified_on_npu
        self.last_seconds = 0.0

    def default_label(self) -> str:
        return "ExecuTorch QNN (NPU, verified)" if self.verified else "ExecuTorch runner (NPU not yet verified)"

    def generate(self, prompt: str, max_tokens: int = 1024, json_mode: bool = False,
                 schema: dict | None = None) -> str:
        with tempfile.TemporaryDirectory(prefix="mg_llm_") as td:
            pf = Path(td) / "prompt.txt"
            of = Path(td) / "output.txt"
            pf.write_text(prompt, encoding="utf-8")
            args = [a.replace("{prompt_file}", str(pf)).replace("{output_file}", str(of))
                     .replace("{prompt}", prompt).replace("{max_tokens}", str(max_tokens))
                    for a in self.command]
            t0 = time.perf_counter()
            try:
                p = subprocess.run(args, cwd=self.cwd, capture_output=True, text=True,
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
                                   e.get("verified_on_npu", False), label)
    raise LLMError(f"Unknown llm.backend {b!r}")
