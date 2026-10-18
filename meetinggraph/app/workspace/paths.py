"""Path confinement and ID validation. Every file path the app touches goes through here."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class PathError(ValueError):
    pass


def new_id() -> str:
    return str(uuid.uuid4())


def check_id(value: str) -> str:
    if not isinstance(value, str) or not UUID_RE.match(value):
        raise PathError(f"Invalid id: {value!r}")
    return value


def confine(root: Path, *parts: str) -> Path:
    """Join parts under root and refuse anything that escapes it."""
    root = root.resolve()
    p = root.joinpath(*parts).resolve()
    if p != root and root not in p.parents:
        raise PathError(f"Path escapes workspace: {p}")
    return p


def clean_name(name: str, limit: int = 80) -> str:
    """Display names are free text, but never used as paths. Trim control chars."""
    name = re.sub(r"[\x00-\x1f\x7f]", " ", name or "").strip()
    return name[:limit] or "Untitled"


class Layout:
    """Knows where everything lives inside the workspace root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def ensure(self) -> None:
        (self.root / "projects").mkdir(parents=True, exist_ok=True)
        (self.root / ".staging").mkdir(parents=True, exist_ok=True)

    # projects
    def projects_dir(self) -> Path:
        return confine(self.root, "projects")

    def project_dir(self, project_id: str) -> Path:
        return confine(self.root, "projects", check_id(project_id))

    def project_file(self, project_id: str, *parts: str) -> Path:
        return confine(self.project_dir(project_id), *parts)

    def feature_path(self, project_id: str, feature_id: str) -> Path:
        return self.project_file(project_id, "features", f"{check_id(feature_id)}.md")

    def meeting_path(self, project_id: str, meeting_id: str) -> Path:
        return self.project_file(project_id, "meetings", f"{check_id(meeting_id)}.md")

    # staging
    def staging_dir(self) -> Path:
        return confine(self.root, ".staging")

    def session_dir(self, session_id: str) -> Path:
        return confine(self.root, ".staging", check_id(session_id))
