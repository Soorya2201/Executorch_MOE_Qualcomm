# MeetingGraph

A private Windows desktop assistant that turns project meetings into an approved, linked project memory. Record or import a meeting, review the proposed summary and feature changes, approve them, then browse a graph and ask cited questions. Nothing leaves the laptop.

This build runs on any Windows laptop first, then moves to the Snapdragon X Elite Dell by swapping the model backends in `config.yaml`.

---

## 1. Prerequisites

- **Python 3.11 or 3.12, 64-bit**, from python.org (3.13/3.14 also work on a normal Windows laptop). In the installer, tick "Add python.exe to PATH". Avoid 3.13+ for now: some wheels lag behind.
- **VS Code** with the **Python** extension.
- About 1 GB of disk for packages. More if you add local models (step 6).

Check in PowerShell:

```powershell
py -0p          # lists installed Pythons; you want a 3.11 or 3.12 entry
```

## 2. Set up the project (PowerShell in VS Code)

1. Unzip `meetinggraph.zip` somewhere short, for example `C:\dev\meetinggraph`.
2. In VS Code: **File → Open Folder…** → choose `C:\dev\meetinggraph`.
3. Open a terminal: **Terminal → New Terminal** (PowerShell).
4. Create and activate the virtual environment:

```powershell
py -3.11 -m venv .venv          # or: py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked ("running scripts is disabled"), run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

5. Install the packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Point VS Code at the venv: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.\.venv\Scripts\python.exe`.

## 3. Verify

```powershell
python -m pytest -q             # expect: 17 passed
python tools\check_env.py       # Python, PySide6, audio devices, backends, workspace
```

`check_env.py` lists your microphones and output devices. If audio shows a warning, the app still works with imported transcripts.

## 4. Turn on the AI model (Qwen 2.5 3B, runs on your laptop)

1. Install **Ollama for Windows** from https://ollama.com/download and open it once (it then runs in the background).
2. Close and reopen the VS Code terminal, activate the venv again, then run:

```powershell
.\tools\setup_qwen.ps1          # downloads qwen2.5:3b-instruct (~2 GB, one time) and tests it
python tools\check_env.py        # should say: Ollama running with qwen2.5:3b-instruct
```

If Ollama isn't running when you start the app, the app still works with simple rules and the blue banner tells you so.

## 5. Run the app

```powershell
python -m app.main
```

The bottom bar shows which AI model is in use. **Closing the window quits the app.**

## 6. Your first test (about 5 minutes)

The blue bar at the top always tells you what to do next. Follow it.

1. Click **＋ Add a meeting** (top left).
2. Under **1. Which project?** choose **➕ Create a new project…** and type `Atlas`.
3. Under **2. What is the meeting called?** type `Kickoff`.
4. Under **3. How do you want to add it?** pick **Open a transcript file**, click **Choose file…**, and pick `tests\fixtures\atlas_01_kickoff.txt`. Click **Add meeting**.
5. Wait a few seconds (with Qwen: up to a minute) until the blue bar says it's **ready**. Click **Review it**.
6. **Step 1:** read the summary. **Step 2:** each white box is one change. Leave them ticked.
7. **Step 3:** click the green **Save to project**, then **Save to project** again in the confirmation.
8. Add the second meeting: **＋ Add a meeting** → project **Atlas** (already selected) → title `Sprint sync` → file `atlas_02_sprint_sync.txt`.
9. In Step 2, one box has a **yellow question**: the meeting said “The sign-in flow”, is that your existing “User login”? Click **Yes**. Then **Save to project**.
10. Explore: **Meetings**, **Features** (click *Offline sync*: see “Needs first” and “Needed by”), **Map** (click a line to see why it exists), and type a question in **Ask about Atlas** on the right.
11. Try Qwen on natural speech: add `atlas_03_natural_design_review.txt` to Atlas. (The first two files use fixed phrasing so they also work without a model; this one needs Qwen.)

**Speech recognition and live recording (optional):** `pip install -r requirements-asr.txt`, set `asr: backend: faster_whisper` in `config.yaml`, restart. Then **Add a meeting → Open an audio recording**, or **● Record a live meeting**.

## 7. How the code is organised

```text
app/
  main.py                    entry point, tray, startup recovery
  config.py                  loads config.yaml
  workspace/                 durable files (no models here)
    paths.py                 UUID checks, path confinement
    store.py                 read files, front matter, AGENTS.md template
    commit.py                proposal -> validated file writes (cycles, deps, archive rules)
    transaction.py           lock, manifest, atomic replace, MASTER_LOG marker, crash recovery
    index.py, graph.py       disposable SQLite index + evidence-backed edges
  pipeline/
    session.py               staging sessions + approve/verify/cleanup contract
    transcript.py            transcript text format, track merge, evidence matching
    extract.py, prompts.py   LLM extraction (JSON + one repair retry)
    heuristic.py             rule-based extractor (no model)
    match.py                 cards vs existing features (candidates, never merge by title)
    query.py                 cited, project-scoped answers
  models/llm.py, asr.py      backends: heuristic | ollama | faster_whisper | executorch_runner
  capture/recorder.py        mic + loopback WAV tracks (soundcard / WASAPI)
  ui/                        main window, add-meeting dialog, review cards, pages, map, recorder, theme
tests/                       pytest suite + fixture meetings
tools/check_env.py           environment check
```

**Suggested split for 4 people tonight:**

| Person | Area |
|---|---|
| A | Model gate on the Dell; `models/` backends for ExecuTorch |
| B | `pipeline/` prompt quality with Ollama on real transcripts |
| C | `ui/` polish and demo flow |
| D | `capture/` on real devices, fixtures, README, benchmarks |

## 8. Moving to the Snapdragon X Elite Dell

1. Install **native ARM64 Python** 3.11 or 3.12 and confirm: `python -c "import platform; print(platform.machine())"` prints `ARM64`.
2. Copy the project folder (without `.venv` and `workspace`), create a fresh venv, and `pip install -r requirements.txt`. Check each package installs on ARM64, PySide6 first.
3. Run `python -m pytest -q` and the 5-minute test (step 5) before touching models.
4. Do the model gate from the build plan: find the ExecuTorch runners and `.pte` models, run them by hand, and confirm QNN/HTP delegation with profiling.
5. Set the runner commands in `config.yaml`:
   - `llm.backend: executorch_runner` with `command:` using `{prompt_file}` or `{prompt}` (and `{output_file}` if the runner writes a file).
   - `asr.backend: executorch_runner` with `{audio_file}` and `{output_file}`. If the runner's output format differs, write a small wrapper script that prints/writes `[{"start", "end", "text"}]` JSON and point `command` at it.
6. Set `verified_on_npu: true` **only** after profiling shows delegation. Until then the app labels the stage "NPU not yet verified".
7. If a package will not install on ARM64 (faster-whisper is the likely one), use the ExecuTorch ASR runner instead, or keep audio import plus typed transcripts as the fallback.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `pip install PySide6` fails | Use Python 3.11/3.12, 64-bit. Recreate the venv. |
| Recording error mentioning `fromstring` | `pip install "numpy<2"` (older soundcard + NumPy 2 mismatch) |
| Output (remote) level stays at zero | Pick the output device your call actually plays through; Bluetooth headsets often show up as a separate device. |
| "Saved — cleanup failed" in the Inbox | Usually antivirus or an open file. Close other programs, then **Retry cleanup**. |
| "Another save is in progress" | Wait a moment; a stale lock clears itself after 10 minutes or on restart. |
| Chat says "I don't know from this project" | Only saved meetings are searched. Save the meeting first, or check you are in the right project. |
| Banner says the AI model isn't available | Open the Ollama app, run `.\\tools\\setup_qwen.ps1`, then restart MeetingGraph. |
| Qwen is slow | First run loads the model (~10 s). On CPU expect 20–60 s per meeting. `qwen2.5:1.5b-instruct` is faster, `7b` is better. |

All data stays in `workspace\`. Delete that folder to start fresh.
