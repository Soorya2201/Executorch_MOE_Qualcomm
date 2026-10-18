"""Render the main window off-screen and save one PNG per page (for UI review without touching real data).

  python tools\\ui_snapshot.py OUT_DIR

Uses a throwaway workspace, the rule-based LLM and no ASR, so it never calls a model or moves files.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import load_config  # noqa: E402
from app.main import build_ctx  # noqa: E402
from app.ui import theme  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["workspace"] = tempfile.mkdtemp(prefix="mg_snap_")
    cfg["llm"]["backend"] = "heuristic"
    cfg["asr"]["backend"] = "none"
    app = QApplication(sys.argv)
    theme.apply(app)
    ctx = build_ctx(cfg)
    win = MainWindow(ctx)
    win.resize(1440, 900)
    win.show()
    app.processEvents()
    for i, btn in win.nav.items():
        win.go(i)
        app.processEvents()
        name = btn.text().split("  ")[0].lower().replace(" ", "_")
        win.grab().save(str(out / f"{i}_{name}.png"))
        print(out / f"{i}_{name}.png")
    win.close()


if __name__ == "__main__":
    main()
