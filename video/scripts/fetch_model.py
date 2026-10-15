#!/usr/bin/env python3
"""下载视觉主模型到 WSL 共享模型目录（默认 /home/administrator/models/<名称>）。

用法：
  python3 fetch_model.py --list                        列出注册表条目的下载状态
  python3 fetch_model.py --id qwen3-vl-4b-instruct     下载指定注册表条目
  python3 fetch_model.py --repo Qwen/Qwen3-VL-4B-Instruct --name Qwen3-VL-4B-Instruct
                                                       直接按 HF 仓库下载

通过 hf-mirror 镜像（--endpoint 或 HF_ENDPOINT 覆盖）用标准库 urllib 下载，
支持断点续传；完成后自动在 video/models/ 下建立软链接。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = REPO_ROOT / "config" / "models.json"
MODELS_ROOT = Path(os.environ.get("MODEL_ROOT", "/home/administrator/models"))
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
USER_AGENT = "yegou-fetch-model/1.0"


def load_registry() -> dict:
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"default": "", "models": []}


def http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(endpoint: str, repo: str, filename: str, target: Path) -> None:
    url = f"{endpoint}/{repo}/resolve/main/{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.stat().st_size if target.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            print(f"  [跳过] {filename}（已完整）", flush=True)
            return
        raise
    if existing and response.status != 206:
        mode = "wb"
        existing = 0
    total = int(response.headers.get("Content-Length") or 0) + existing
    done = existing
    last_report = 0.0
    with open(target, mode) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            now = time.time()
            if now - last_report >= 3.0:
                last_report = now
                if total:
                    print(f"  {filename}: {done / 1048576:.0f}/{total / 1048576:.0f} MB", flush=True)
                else:
                    print(f"  {filename}: {done / 1048576:.0f} MB", flush=True)
    print(f"  [完成] {filename}（{done / 1048576:.0f} MB）", flush=True)


def download_model(endpoint: str, repo: str, name: str, link_rel: str | None) -> int:
    target_dir = MODELS_ROOT / name
    print(f"仓库：{repo}\n目标：{target_dir}\n镜像：{endpoint}\n")
    try:
        meta = http_json(f"{endpoint}/api/models/{repo}")
    except Exception as exc:
        print(f"无法获取仓库信息（检查仓库名 / 网络 / 代理）：{exc}", file=sys.stderr)
        return 1
    siblings = [
        item.get("rfilename")
        for item in (meta.get("siblings") or [])
        if item.get("rfilename")
    ]
    if not siblings:
        print("仓库文件列表为空，请检查仓库名。", file=sys.stderr)
        return 1
    for filename in siblings:
        if filename.startswith(".git"):
            continue
        download_file(endpoint, repo, filename, target_dir / filename)
    if link_rel:
        link_path = REPO_ROOT / link_rel
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink() or link_path.exists():
            print(f"\n软链接已存在：{link_path}")
        else:
            link_path.symlink_to(target_dir)
            print(f"\n软链接：{link_path} -> {target_dir}")
    print("\n下载完成。在视频台「模型」页切换到该模型即可。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="下载视频台视觉主模型")
    parser.add_argument("--list", action="store_true", help="列出注册表条目的下载状态")
    parser.add_argument("--id", help="注册表条目 id")
    parser.add_argument("--repo", help="直接指定 HF 仓库（如 Qwen/Qwen3-VL-4B-Instruct）")
    parser.add_argument("--name", help="配合 --repo 使用的目标目录名")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="模型镜像地址")
    args = parser.parse_args()

    registry = load_registry()
    models = registry.get("models") or []

    if args.list or (not args.id and not args.repo):
        print(f"模型根目录：{MODELS_ROOT}\n")
        for entry in models:
            link = str(entry.get("model_link") or "")
            installed = (REPO_ROOT / link).is_dir() if link else False
            state = "已下载" if installed else "未下载"
            print(f"[{state}] {entry.get('id')}  {entry.get('display_name')}  {entry.get('hf_repo') or ''}")
        if not models:
            print("（注册表为空）")
        return 0

    if args.id:
        entry = next((item for item in models if str(item.get("id")) == args.id), None)
        if entry is None:
            print(f"注册表里没有 id={args.id}", file=sys.stderr)
            return 1
        repo = args.repo or str(entry.get("hf_repo") or "")
        if not repo:
            print("注册表条目缺少 hf_repo，可用 --repo 指定", file=sys.stderr)
            return 1
        link_rel = str(entry.get("model_link") or "")
        name = Path(link_rel).name if link_rel else str(entry.get("id"))
        return download_model(args.endpoint, repo, name, link_rel or None)

    name = args.name or args.repo.rsplit("/", 1)[-1]
    return download_model(args.endpoint, args.repo, name, f"models/{name}")


if __name__ == "__main__":
    raise SystemExit(main())
