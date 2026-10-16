"""Manage the video project's dedicated persistent audio container."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = os.environ.get("ASR_RESIDENT_CONTAINER", "vision-audio-resident")
PORT = int(os.environ.get("ASR_RESIDENT_PORT", "8003"))
URL = f"http://127.0.0.1:{PORT}"
JOBS = Path(os.environ.get("ASR_RESIDENT_JOBS", str(Path.home() / ".cache/yegou-resident-audio/jobs"))).expanduser().resolve()


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True, stderr=subprocess.STDOUT).strip()


def session() -> requests.Session:
    client = requests.Session()
    client.trust_env = False
    return client


def stop_service() -> None:
    found = subprocess.run(["docker", "inspect", CONTAINER], capture_output=True, text=True)
    if found.returncode:
        return
    info = json.loads(found.stdout)[0]
    if info["Config"].get("Labels", {}).get("yegou.audio.resident") != "1":
        raise RuntimeError(f"拒绝停止未归本项目管理的容器：{CONTAINER}")
    if info["State"]["Running"]:
        docker("stop", CONTAINER)


def ensure_service() -> dict:
    image = os.environ.get("ASR_IMAGE", "ragflow-qwen-asr:0.0.6")
    model_root = Path(os.environ.get("MODEL_ROOT", str(ROOT / "models"))).resolve()
    asr = (model_root / "Qwen3-ASR-0.6B").resolve()
    aligner = (model_root / "Qwen3-ForcedAligner-0.6B").resolve()
    for model in (asr, aligner):
        if not (model / "config.json").is_file():
            raise RuntimeError(f"模型不存在：{model}")
    image_id = docker("image", "inspect", "-f", "{{.Id}}", image)
    tokens = os.environ.get("ASR_MAX_NEW_TOKENS", "2048")
    worker_code = hashlib.sha256(
        (ROOT / "app/resident_audio_server.py").read_bytes() + (ROOT / "audio_timeline.py").read_bytes()
    ).hexdigest()
    fingerprint = hashlib.sha256(json.dumps([image_id, str(ROOT), str(asr), str(aligner), str(JOBS), PORT, tokens, worker_code]).encode()).hexdigest()
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS.parent / "startup.lock").open("w") as lock, session() as client:
        fcntl.flock(lock, fcntl.LOCK_EX)
        found = subprocess.run(["docker", "inspect", CONTAINER], capture_output=True, text=True)
        if not found.returncode:
            info = json.loads(found.stdout)[0]
            labels = info["Config"].get("Labels", {})
            if labels.get("yegou.audio.resident") != "1":
                raise RuntimeError(f"常驻容器名被占用：{CONTAINER}")
            if labels.get("yegou.audio.config") != fingerprint:
                docker("rm", "-f", CONTAINER)
                found = None
            elif not info["State"]["Running"]:
                docker("start", CONTAINER)
        if found is None or found.returncode:
            docker("run", "-d", "--name", CONTAINER, "--gpus", "all", "--ipc=host", "--shm-size=4g",
                   "--label", "yegou.audio.resident=1", "--label", f"yegou.audio.config={fingerprint}",
                   "-p", f"127.0.0.1:{PORT}:8766", "-e", "TRANSFORMERS_OFFLINE=1", "-e", "HF_HUB_OFFLINE=1",
                   "-e", "PYTHONUNBUFFERED=1", "-e", f"ASR_MAX_NEW_TOKENS={tokens}",
                   "-v", f"{ROOT}:/app:ro", "-v", f"{asr}:/models/asr:ro", "-v", f"{aligner}:/models/aligner:ro",
                   "-v", f"{JOBS}:/jobs", "--entrypoint", "python3", image, "/app/app/resident_audio_server.py")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if docker("inspect", "-f", "{{.State.Running}}", CONTAINER) != "true":
                raise RuntimeError("常驻音频服务退出：\n" + docker("logs", "--tail", "60", CONTAINER))
            try:
                response = client.get(URL + "/health", timeout=2)
                response.raise_for_status()
                if response.json().get("ready"):
                    return response.json()
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("常驻音频服务启动超时")


def analyze(source: Path, output: Path, language: str = "") -> dict:
    health = ensure_service()
    print(f"复用常驻 ASR + ForcedAligner：pid={health['pid']}，已完成={health['completed']}", flush=True)
    job_id = uuid.uuid4().hex
    folder = JOBS / job_id
    input_dir = folder / "input"
    input_dir.mkdir(parents=True)
    request_finished = False
    try:
        shutil.copy2(source, input_dir / source.name)
        with session() as client:
            response = client.post(URL + "/analyze", json={"job_id": job_id, "language": language or None}, timeout=(10, 7200))
            request_finished = True
            if not response.ok:
                raise RuntimeError(f"常驻音频分析失败（{response.status_code}）：{response.text[:2000]}")
            result = response.json()
        shutil.copytree(folder / "output", output, dirs_exist_ok=True)
        print(f"常驻音频完成：{result['elapsed_s']:.2f}秒，pid={result['pid']}", flush=True)
        return result
    finally:
        # Docker writes as root. Remove only our UUID after the worker replies;
        # a timed-out request may still be reading its source and is retained.
        if request_finished:
            try:
                docker("exec", CONTAINER, "python3", "-c",
                       "import shutil,sys; shutil.rmtree(sys.argv[1])", f"/jobs/{job_id}")
            except subprocess.CalledProcessError as exc:
                print(f"音频临时目录保留待清理：{folder}（{exc}）", flush=True)
