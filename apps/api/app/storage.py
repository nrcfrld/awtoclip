import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()
PROJECTS = DATA_ROOT / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)
ID_RE = re.compile(r"^[a-f0-9]{12}$")


def project_dir(project_id: str) -> Path:
    if not ID_RE.fullmatch(project_id):
        raise HTTPException(404, "Project not found")
    path = PROJECTS / project_id
    if not path.is_dir():
        raise HTTPException(404, "Project not found")
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(404, f"Missing or invalid {path.name}") from exc


def atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        temp = Path(out.name)
    os.replace(temp, path)


def project(project_id: str) -> dict:
    return read_json(project_dir(project_id) / "project.json")


def save_project(project_id: str, value: dict) -> None:
    atomic_json(project_dir(project_id) / "project.json", value)


def source_path(project_id: str) -> Path:
    matches = list(project_dir(project_id).glob("source.*"))
    if len(matches) != 1 or matches[0].suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise HTTPException(404, "Source video not found")
    return matches[0]


def stage(project_id: str, name: str, state: str, *, error: str | None = None, progress: int | None = None) -> None:
    data = project(project_id)
    data["state"] = state
    data.setdefault("stages", {})[name] = "failed" if error else "completed" if state == "READY" or progress == 100 else "running"
    data["progress"] = progress
    data["error"] = error
    save_project(project_id, data)
