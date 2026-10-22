"""Stable project workspace layout shared by CLI and HTTP tasks."""
from __future__ import annotations

import json
from pathlib import Path
import time

DIRECTORIES = {
    "01_config": "任务请求、角色映射、时间轴和可复现配置",
    "02_work": "可重建的任务中间文件，按任务 ID 隔离",
    "03_voices": "本项目的参考音色、参考台词和角色对应关系",
    "04_outputs": "成片、无字幕版、字幕、音轨与质量报告；长期保留",
    "05_logs": "任务状态快照、事件记录和验证证据",
    "99_tmp": "短期暂存；禁止存放唯一一份素材或成片",
}


def initialize(root: Path):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in DIRECTORIES:
        (root / name).mkdir(exist_ok=True)
    manifest = root / "workspace.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({"version": 1, "root": str(root), "created_at": time.time(),
                                       "directories": DIRECTORIES, "sources": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"workspace": str(root), "manifest": str(manifest), "directories": DIRECTORIES}
