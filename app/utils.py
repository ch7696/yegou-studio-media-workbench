from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .config import DEFAULT_OUTPUT


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True, timeout=timeout)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_name(value: str, limit: int = 64) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:limit].rstrip(" ._") or "未命名"


def to_wsl_path(value: str) -> Path:
    text = str(value or "").strip()
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match:
        rest = match.group(2).replace("\\", "/")
        return Path(f"/mnt/{match.group(1).lower()}/{rest}")
    return Path(text or str(DEFAULT_OUTPUT))


def to_windows_path(path: Path) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", str(path))
    if not match:
        return str(path)
    rest = (match.group(2) or "").replace("/", "\\")
    return f"{match.group(1).upper()}:\\{rest}" if rest else f"{match.group(1).upper()}:\\"


def probe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        120,
    )
    return json.loads(result.stdout)


def duration(path: Path) -> float:
    try:
        return float((probe(path).get("format") or {}).get("duration") or 0.0)
    except Exception:
        return 0.0


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(manifests_dir: Path, asset_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9-]{12,80}", asset_id):
        raise HTTPException(status_code=400, detail="资产编号无效")
    path = manifests_dir / f"{asset_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="找不到该声音资产")
    return json.loads(path.read_text(encoding="utf-8"))
