"""Host-side durable agent gateway. Python standard library only, loopback by default."""
from __future__ import annotations

import argparse
import array
import contextlib
import hashlib
import http.client
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .agent_workspace import initialize

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("YEGOU_AGENT_STATE", "/home/administrator/ragflow-managed/state/agent-gateway"))
BACKEND = os.environ.get("YEGOU_BACKEND_URL", "http://127.0.0.1:7866")
ASR_HEALTH = os.environ.get("YEGOU_ASR_HEALTH_URL", "http://127.0.0.1:18763/health")
CONTAINERS = ("ragflow-model-speaker-webui", "ragflow-model-asr-api")
TERMINAL = {"done", "failed", "cancelled", "needs_review", "interrupted"}
BACKEND_CONTEXT = threading.local()
JOB_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["input"],
    "anyOf": [{"required": ["voice"]}, {"required": ["voices"]}],
    "properties": {
        "input": {"type": "string", "description": "Existing local Windows or WSL video path"},
        "voice": {"type": "string", "description": "Default registered voice_id for all speakers"},
        "voices": {"type": "object", "additionalProperties": {"type": "string"}},
        "background": {"type": "string", "enum": ["remove", "keep"], "default": "remove"},
        "output": {"type": "string"}, "language": {"type": "string", "default": "Chinese"},
        "workspace": {"type": "string", "description": "Project task workspace for outputs and execution records"},
        "speaker_num": {"type": "integer", "minimum": 0, "maximum": 4, "default": 0},
        "noise_reduction": {"type": "number", "minimum": 0, "maximum": 30, "default": 8},
        "mode": {"type": "string", "enum": ["cross_lingual", "zero_shot", "instruct2"], "default": "cross_lingual"},
        "speed": {"type": "number", "minimum": .5, "maximum": 2, "default": 1},
        "voice_gain_db": {"type": "number", "minimum": -12, "maximum": 18, "default": 6},
        "fit_to_slot": {"type": "boolean", "default": True},
        "adaptive_speed": {"type": "boolean", "default": True},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647, "default": 0},
        "prompt_instruction": {"type": "string", "maxLength": 500},
        "unknown_speaker_id": {"type": "string"},
        "review_after_analysis": {"type": "boolean", "default": False},
        "edits": {"type": "array", "items": {"type": "object", "required": ["segment_id"],
                  "additionalProperties": False, "properties": {
                      "segment_id": {"type": "string"}, "speaker_id": {"type": "string"},
                      "text": {"type": "string", "maxLength": 2000},
                      "dubbing_text": {"type": "string", "maxLength": 2000},
                      "start_sec": {"type": "number", "minimum": 0}, "end_sec": {"type": "number", "exclusiveMinimum": 0}}}},
        "max_retries": {"type": "integer", "minimum": 0, "maximum": 3, "default": 1},
        "timeout_seconds": {"type": "integer", "minimum": 60, "maximum": 172800, "default": 21600},
    },
}


class Problem(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def local_path(value):
    value = str(value)
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return Path(value).expanduser().resolve()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def backend(method, route, payload=None, fields=None, file=None, timeout=120):
    """Stream multipart files; large video uploads never load into RAM."""
    if file is None:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(BACKEND + route, data=data, method=method,
                                         headers={"Content-Type": "application/json", "X-Yegou-Lease": getattr(BACKEND_CONTEXT, "lease", "")})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise Problem(f"处理服务 HTTP {exc.code}: {exc.read().decode(errors='replace')}", 409 if exc.code == 409 else 502) from exc
    name, path = file
    boundary = "yegou" + uuid.uuid4().hex
    chunks = []
    for key, value in (fields or {}).items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    filename = re.sub(r'["\\\r\n]', "_", path.name)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    head = b"".join(chunks)
    tail = f"\r\n--{boundary}--\r\n".encode()
    url = urllib.parse.urlsplit(BACKEND)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        conn.putrequest(method, url.path + route)
        conn.putheader("Content-Type", "multipart/form-data; boundary=" + boundary)
        conn.putheader("Content-Length", str(len(head) + path.stat().st_size + len(tail)))
        conn.putheader("X-Yegou-Lease", getattr(BACKEND_CONTEXT, "lease", ""))
        conn.endheaders()
        conn.send(head)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                conn.send(chunk)
        conn.send(tail)
        response = conn.getresponse()
        data = response.read()
        if response.status >= 400:
            raise Problem(f"处理服务 HTTP {response.status}: {data.decode(errors='replace')}", 502)
        return json.loads(data)
    finally:
        conn.close()


def run(*args, timeout=180):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise Problem(result.stderr.strip() or result.stdout.strip() or f"命令失败: {args[0]}", 503)
    return result.stdout.strip()


def audio_metrics(path):
    """Bounded-memory PCM16 checks; sample every 16th value for long tracks."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise ValueError("质量检查只支持 PCM16 WAV")
        duration = source.getnframes() / source.getframerate()
        peak, clipped, count, square_sum = 0, 0, 0, 0
        while block := source.readframes(65536):
            samples = array.array("h", block)
            if sys.byteorder != "little":
                samples.byteswap()
            for sample in samples[::16]:
                absolute = abs(sample)
                peak = max(peak, absolute)
                clipped += absolute >= 32760
                square_sum += sample * sample
                count += 1
        return {"duration_sec": round(duration, 3), "sampled_peak": round(peak / 32768, 5),
                "sampled_rms": round((square_sum / max(1, count)) ** .5 / 32768, 5),
                "sampled_clip_ratio": round(clipped / max(1, count), 6)}


def media_metrics(source, video):
    """Use the already-running workbench's FFprobe; no host FFmpeg installation required."""
    def probe(path):
        return json.loads(run("docker", "exec", CONTAINERS[0], "ffprobe", "-v", "error",
                              "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,sample_rate",
                              "-of", "json", str(path), timeout=30))
    original, output = probe(source), probe(video)
    delta = abs(float(original["format"]["duration"]) - float(output["format"]["duration"]))
    types = {s.get("codec_type") for s in output.get("streams", [])}
    return {"source_duration_sec": float(original["format"]["duration"]),
            "output_duration_sec": float(output["format"]["duration"]),
            "duration_delta_sec": round(delta, 3), "streams": output.get("streams", []),
            "valid": delta <= .25 and {"audio", "video"} <= types}


class Gateway:
    def __init__(self, state=STATE, idle_seconds=300):
        self.state = state
        self.idle_seconds = idle_seconds
        self.lock = threading.RLock()
        self.lifecycle = threading.RLock()
        self.wake = threading.Event()
        self.closing = threading.Event()
        self.last_use = time.monotonic()
        self.managed = False  # Never auto-stop a manually started WebUI until gateway takes ownership.
        runtime_path = state / "runtime.json"
        if runtime_path.is_file():
            self.managed = bool(json.loads(runtime_path.read_text()).get("managed"))
        self.active = None
        self.jobs = {}
        policy_path = state / "policy.json"
        if policy_path.is_file():
            self.idle_seconds = int(json.loads(policy_path.read_text())["idle_seconds"])
        for folder in ("jobs", "voices"):
            (state / folder).mkdir(parents=True, exist_ok=True)
        for path in sorted((state / "jobs").glob("*.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] not in TERMINAL and job["status"] != "queued":
                job.update(status="interrupted", error="控制服务重启；调用 retry 重新执行，不自动重复提交模型任务")
                atomic_json(path, job)
            self.jobs[job["id"]] = job

    def update(self, job, **values):
        with self.lock:
            job.update(values, updated_at=time.time())
            atomic_json(self.state / "jobs" / (job["id"] + ".json"), job)
            workspace = job.get("request", {}).get("workspace")
            if workspace:
                root = Path(workspace)
                atomic_json(root / "05_logs" / job["id"] / "status.json", job)
                event = {"time": job["updated_at"], "status": job["status"],
                         "stage": job.get("stage"), "phase": job.get("phase"), "error": job.get("error")}
                with (root / "05_logs" / job["id"] / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def get(self, id):
        with self.lock:
            if id not in self.jobs:
                raise Problem("任务不存在", 404)
            return json.loads(json.dumps(self.jobs[id]))

    def voices(self):
        with self.lock:
            return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.state / "voices").glob("*/voice.json"))]

    def voice(self, id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", str(id)):
            raise Problem("voice_id 只能使用字母、数字、下划线或短横线")
        path = self.state / "voices" / id / "voice.json"
        if not path.is_file():
            raise Problem(f"音色不存在: {id}", 404)
        return json.loads(path.read_text(encoding="utf-8"))

    def add_voice(self, data):
        id = data.get("id") or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", str(id)):
            raise Problem("音色编号无效")
        source = local_path(data.get("reference", ""))
        if not source.is_file():
            raise Problem("参考媒体文件不存在")
        start = number(data.get("start", 0), 0, 86400, "start")
        seconds = number(data.get("seconds", 10), 1, 30, "seconds")
        with self.lock:
            root = self.state / "voices" / id
            if root.exists():
                raise Problem("音色编号已存在；请使用新编号以保持已有任务可复现", 409)
            root.mkdir(parents=True)
            try:
                target = root / ("reference" + source.suffix)
                shutil.copy2(source, target)
                voice = dict(id=id, name=str(data.get("name") or id), reference=str(target),
                             start=start, seconds=seconds, text=str(data.get("text", "")), created_at=time.time())
                atomic_json(root / "voice.json", voice)
                return voice
            except Exception:
                shutil.rmtree(root)
                raise

    def submit(self, data, key=None):
        allowed = {"input", "voice", "voices", "background", "output", "workspace", "speaker_num", "language",
                   "noise_reduction", "mode", "speed", "voice_gain_db", "fit_to_slot", "adaptive_speed",
                   "seed", "prompt_instruction", "unknown_speaker_id", "edits", "max_retries", "timeout_seconds", "review_after_analysis"}
        extra = set(data) - allowed
        if extra:
            raise Problem("未知参数: " + ", ".join(sorted(extra)))
        source = local_path(data.get("input", ""))
        if not source.is_file():
            raise Problem("输入视频不存在")
        data = dict(data, input=str(source))
        if data.get("background", "remove") not in ("remove", "keep"):
            raise Problem("background 必须是 remove 或 keep")
        if data.get("mode", "cross_lingual") not in ("cross_lingual", "zero_shot", "instruct2"):
            raise Problem("不支持的 TTS 模式")
        if not data.get("voice") and not data.get("voices"):
            raise Problem("必须提供 voice 或 voices 说话人音色映射")
        if not isinstance(data.get("voices", {}), dict):
            raise Problem("voices 必须是 speaker_id 到 voice_id 的对象")
        for voice_id in [data.get("voice")] + list(data.get("voices", {}).values()):
            if voice_id:
                self.voice(voice_id)
        for name, default, low, high in (("speed", 1, .5, 2), ("voice_gain_db", 6, -12, 18),
                                       ("noise_reduction", 8, 0, 30), ("speaker_num", 0, 0, 4),
                                       ("seed", 0, 0, 2147483647), ("max_retries", 1, 0, 3),
                                       ("timeout_seconds", 21600, 60, 172800)):
            data[name] = number(data.get(name, default), low, high, name)
            if name in {"speaker_num", "seed", "max_retries", "timeout_seconds"}:
                if data[name] != int(data[name]):
                    raise Problem(name + " 必须是整数")
                data[name] = int(data[name])
        for name in ("fit_to_slot", "adaptive_speed", "review_after_analysis"):
            if name in data and not isinstance(data[name], bool):
                raise Problem(name + " 必须是 JSON 布尔值")
        if not isinstance(data.get("edits", []), list):
            raise Problem("edits 必须是数组")
        for field_name in ("input", "voice", "output", "language", "prompt_instruction", "unknown_speaker_id"):
            if field_name in data and not isinstance(data[field_name], str):
                raise Problem(field_name + " 必须是字符串")
        if len(data.get("prompt_instruction", "")) > 500:
            raise Problem("prompt_instruction 最长 500 字")
        for edit in data.get("edits", []):
            if not isinstance(edit, dict) or not edit.get("segment_id"):
                raise Problem("每个 edit 都必须提供 segment_id")
            if set(edit) - {"segment_id", "speaker_id", "text", "dubbing_text", "start_sec", "end_sec"}:
                raise Problem("edit 包含不支持的参数")
        if data.get("output"):
            data["output"] = str(local_path(data["output"]))
        if data.get("workspace"):
            data["workspace"] = str(local_path(data["workspace"]))
            initialize(Path(data["workspace"]))
            if not data.get("output"):
                data["output"] = str(Path(data["workspace"]) / "04_outputs")
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if key:
                for previous in self.jobs.values():
                    if previous.get("idempotency_key") == key:
                        if previous["fingerprint"] != fingerprint:
                            raise Problem("同一 Idempotency-Key 对应的参数不同", 409)
                        return self.get(previous["id"])
            id = uuid.uuid4().hex
            job = dict(id=id, status="queued", stage="queued", request=data, attempts=0,
                       fingerprint=fingerprint, idempotency_key=key, created_at=time.time())
            if data.get("workspace"):
                root = Path(data["workspace"])
                (root / "02_work" / id).mkdir(parents=True, exist_ok=True)
                (root / "99_tmp" / id).mkdir(parents=True, exist_ok=True)
                atomic_json(root / "01_config" / (id + ".json"), data)
            self.jobs[id] = job
            self.update(job)
            self.wake.set()
            return self.get(id)

    def service_status(self):
        containers = []
        for name in CONTAINERS:
            try:
                status = run("docker", "inspect", "--format", "{{.State.Status}}", name, timeout=10)
            except Problem:
                status = "unavailable"
            containers.append(dict(name=name, status=status))
        with self.lock:
            return dict(containers=containers, active_job=self.active, managed=self.managed,
                        idle_seconds=self.idle_seconds, idle_for=round(time.monotonic() - self.last_use),
                        queued=sum(j["status"] == "queued" for j in self.jobs.values()), gateway_pid=os.getpid())

    def start(self):
        with self.lifecycle:
            self.last_use = time.monotonic()
            run("systemctl", "start", "docker", timeout=30) if os.geteuid() == 0 else None
            for name, script in zip(reversed(CONTAINERS), ("run_asr_api.sh", "run_webui.sh")):
                try:
                    state = run("docker", "inspect", "--format", "{{.State.Status}}", name, timeout=10)
                except Problem:
                    state = "missing"
                if state == "missing":
                    run("bash", str(ROOT / "scripts" / script))
                elif state != "running":
                    run("docker", "start", name)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    backend("GET", "/api/health", timeout=3)
                    with urllib.request.urlopen(ASR_HEALTH, timeout=3) as response:
                        if not json.load(response).get("ok"):
                            raise Problem("ASR 尚未就绪", 503)
                    backend("POST", "/api/system/undrain", timeout=3)
                    self.managed = True
                    atomic_json(self.state / "runtime.json", {"managed": True})
                    return {"status": "running"}
                except Exception:
                    time.sleep(1)
            raise Problem("工作台启动超时，请检查 docker logs ragflow-model-speaker-webui", 503)

    def stop(self):
        with self.lifecycle, self.lock:
            if self.active or any(j["status"] == "queued" for j in self.jobs.values()):
                raise Problem("仍有运行或排队任务，暂不能关闭", 409)
            running = []
            for name in CONTAINERS:
                try:
                    if run("docker", "inspect", "--format", "{{.State.Running}}", name, timeout=10) == "true":
                        running.append(name)
                except Problem:
                    pass
            if not running:
                self.managed = False
                atomic_json(self.state / "runtime.json", {"managed": False})
                return {"status": "stopped"}
            if CONTAINERS[0] in running:
                try:
                    drain = backend("POST", "/api/system/drain", timeout=10)
                except Exception as exc:
                    raise Problem(f"不能确认工作台空闲，拒绝关闭: {exc}", 409) from exc
                if not drain.get("idle"):
                    raise Problem("WebUI 仍有任务，暂不能关闭", 409)
            try:
                run("docker", "stop", "--time", "30", *running, timeout=90)
            except Exception:
                with contextlib.suppress(Exception):
                    backend("POST", "/api/system/undrain", timeout=5)
                raise
            self.managed = False
            atomic_json(self.state / "runtime.json", {"managed": False})
            return {"status": "stopped", "containers": running}

    def poll(self, job, child):
        deadline = time.monotonic() + job["request"]["timeout_seconds"]
        while time.monotonic() < deadline:
            if self.closing.is_set():
                raise Problem("控制服务正在关闭")
            state = backend("GET", f"/api/dubbing/jobs/{child}")
            self.update(job, child_status=state.get("status"), progress=state.get("progress", 0),
                        phase=state.get("phase"))
            if state["status"] == "failed":
                raise Problem(state.get("error", "模型任务失败"), 502)
            if state["status"] == "done":
                return state
            time.sleep(2)
        # A timed out child may still run: never automatically duplicate it.
        raise Problem("处理超时；子任务可能仍在运行，请检查后 retry", 408)

    def execute(self, job):
        data = job["request"]
        self.update(job, status="running", stage="starting", error=None)
        self.start()
        self.update(job, stage="waiting_for_workbench")
        deadline = time.monotonic() + data["timeout_seconds"]
        while True:
            try:
                BACKEND_CONTEXT.lease = backend("POST", "/api/system/lease")["token"]
                break
            except Problem as exc:
                if exc.status != 409 or time.monotonic() > deadline:
                    raise
                time.sleep(2)
        child = job.get("child_id")
        if not child:
            self.update(job, stage="analysis")
            result = backend("POST", "/api/dubbing/analyze", fields={
                "output_dir": data.get("output", ""), "speaker_num": data["speaker_num"],
                "language": data.get("language", "Chinese"), "noise_reduction": data["noise_reduction"],
            }, file=("media", Path(data["input"])))
            child = result["job_id"]
            self.update(job, child_id=child)
            self.poll(job, child)
        else:
            # Retry after interruption reuses only a completed analysis, never a live child.
            state = backend("GET", f"/api/dubbing/jobs/{child}")
            if state["status"] not in ("done", "failed"):
                raise Problem("子任务仍可能运行，不能重复生成；请查询子任务或重启工作台后 retry", 409)
        self.update(job, stage="references")
        timeline = backend("GET", f"/api/dubbing/jobs/{child}/timeline")
        if data.get("review_after_analysis") and not job.get("analysis_reviewed"):
            self.update(job, status="needs_review", stage="review", timeline=timeline,
                        error="已按 review_after_analysis 暂停；agent 检查时间轴后调用 resume")
            return
        edits = list(data.get("edits", []))
        unknown = [s for s in timeline["segments"] if s.get("speaker_id") == "unknown"]
        # A global voice explicitly makes every utterance use the same voice.
        replacement = data.get("unknown_speaker_id")
        if data.get("voice") and not data.get("voices") and not replacement and timeline["speakers"]:
            replacement = timeline["speakers"][0]["speaker_id"]
        if replacement:
            edits = [dict(segment_id=s["segment_id"], speaker_id=replacement) for s in unknown] + edits
        if edits:
            backend("POST", f"/api/dubbing/jobs/{child}/timeline", {"edits": edits})
            timeline = backend("GET", f"/api/dubbing/jobs/{child}/timeline")
        speaker_ids = {s["speaker_id"] for s in timeline["segments"] if (s.get("dubbing_text") or s.get("text", "")).strip()}
        known = {s["speaker_id"] for s in timeline["speakers"]}
        mapping = {s: data.get("voices", {}).get(s) or data.get("voice") for s in speaker_ids}
        missing = sorted(s for s in speaker_ids if s not in known or not mapping[s])
        if missing or not speaker_ids:
            self.update(job, status="needs_review", stage="review", missing_speakers=missing,
                        timeline=timeline, error="存在未映射说话人或没有可配音台词；使用 resume 补充音色映射/台词编辑")
            return
        for speaker, voice_id in sorted(mapping.items()):
            voice = self.voice(voice_id)
            backend("POST", f"/api/dubbing/jobs/{child}/speakers/{urllib.parse.quote(speaker, safe='')}/reference",
                    fields={"reference_start": voice["start"], "reference_seconds": voice["seconds"],
                            "reference_text": voice["text"]}, file=("reference_media", Path(voice["reference"])))
        self.update(job, stage="generation", resolved_voices=mapping)
        controls = {k: data[k] for k in ("speed", "voice_gain_db", "seed", "fit_to_slot", "adaptive_speed", "prompt_instruction") if k in data}
        controls.update(mode=data.get("mode", "cross_lingual"), target_speaker_id="all", export_video=True,
                        target_original_volume=0, other_speaker_volume=0,
                        background_volume=1 if data.get("background", "remove") == "keep" else 0)
        backend("POST", f"/api/dubbing/jobs/{child}/generate", controls)
        result = self.poll(job, child)
        quality = result.get("quality", {})
        expected = sum(bool((s.get("dubbing_text") or s.get("text", "")).strip()) for s in timeline["segments"])
        generated = quality.get("segment_count", 0)
        urls = {k: f'/api/agent/jobs/{job["id"]}/files/{k}' for k in result.get("generation_urls", {})}
        checks, warnings = {}, []
        for key in ("voice_track", "mix_track", "video"):
            target = result.get("generation", {}).get(key)
            path = local_path(target) if target else None
            if not path or not path.is_file() or path.stat().st_size == 0:
                warnings.append(key + " 输出不存在或为空")
            elif key != "video":
                try:
                    checks[key] = audio_metrics(path)
                    if checks[key]["sampled_peak"] < .001:
                        warnings.append(key + " 接近静音")
                    if checks[key]["sampled_clip_ratio"] > .01:
                        warnings.append(key + " 疑似削波")
                    if abs(checks[key]["duration_sec"] - float(timeline.get("duration_sec", 0))) > .25:
                        warnings.append(key + " 与视频时长不匹配")
                except (OSError, ValueError, wave.Error) as exc:
                    warnings.append(key + " 质量检测失败: " + str(exc))
        try:
            checks["video"] = media_metrics(result.get("source_path", data["input"]),
                                            result.get("generation", {}).get("video", ""))
            if not checks["video"]["valid"]:
                warnings.append("导出视频缺少音视频流或改变了原片时长")
        except Exception as exc:
            warnings.append("视频格式检测失败: " + str(exc))
        status = "done" if generated == expected and "video" in urls and not warnings else "needs_review"
        self.update(job, status=status, stage="complete", progress=100, result=result,
                    files=urls, quality=dict(quality, expected_segments=expected, generated_segments=generated,
                                            unknown_reassigned=len(unknown) if replacement else 0,
                                            audio_checks=checks, warnings=warnings),
                    error=None if status == "done" else "片段数量、导出或音频质量需复核")

    def action(self, id, action, data):
        with self.lock:
            job = self.jobs.get(id)
            if not job:
                raise Problem("任务不存在", 404)
            if action == "cancel":
                if job["status"] != "queued":
                    raise Problem("只能取消排队任务；运行任务不会被强制中断", 409)
                self.update(job, status="cancelled")
            elif action in ("retry", "resume"):
                if job["status"] not in ("failed", "interrupted", "needs_review"):
                    raise Problem("当前状态不能重试或恢复", 409)
                allowed = {"voice", "voices", "unknown_speaker_id", "edits"}
                if set(data) - allowed:
                    raise Problem("resume 仅允许 voice、voices、unknown_speaker_id、edits")
                if not isinstance(data.get("voices", {}), dict) or not isinstance(data.get("edits", []), list):
                    raise Problem("voices 必须为对象，edits 必须为数组")
                for voice_id in [data.get("voice")] + list(data.get("voices", {}).values()):
                    if voice_id:
                        self.voice(voice_id)
                job["request"].update(data)
                if job.get("stage") == "analysis" and job.get("child_id"):
                    child = backend("GET", f'/api/dubbing/jobs/{job["child_id"]}')
                    if child["status"] != "failed":
                        raise Problem("原分析任务未确认失败，不能重复提交", 409)
                    job.pop("child_id", None)
                self.update(job, status="queued", error=None, attempts=0, analysis_reviewed=True)
                self.wake.set()
            else:
                raise Problem("未知操作", 404)
            return self.get(id)

    def worker(self):
        while not self.closing.is_set():
            with self.lock:
                job = next((j for j in self.jobs.values() if j["status"] == "queued"), None)
                if job:
                    self.active = job["id"]
                    self.update(job, status="running", attempts=job["attempts"] + 1)
            if job:
                try:
                    self.execute(job)
                except Exception as exc:
                    # Retry only failures known to have finished. Transport errors/timeouts
                    # have uncertain outcomes and must not create duplicate GPU work.
                    retry = isinstance(exc, Problem) and exc.status == 502 and "child_id" in job
                    if retry:
                        try:
                            retry = backend("GET", f'/api/dubbing/jobs/{job["child_id"]}')["status"] == "failed"
                        except Exception:
                            retry = False
                    if retry and job.get("stage") == "analysis":
                        job.pop("child_id", None)
                    self.update(job, status="queued" if retry and job["attempts"] <= job["request"]["max_retries"] else "failed",
                                error=f"{type(exc).__name__}: {exc}")
                finally:
                    if getattr(BACKEND_CONTEXT, "lease", ""):
                        with contextlib.suppress(Exception):
                            backend("POST", "/api/system/lease/release")
                        BACKEND_CONTEXT.lease = ""
                    with self.lock:
                        self.active = None
                        self.last_use = time.monotonic()
            elif self.managed and self.idle_seconds and time.monotonic() - self.last_use >= self.idle_seconds:
                try:
                    self.stop()
                except Exception:
                    self.last_use = time.monotonic()  # Retry after another full idle interval.
            self.wake.wait(1)
            self.wake.clear()


def number(value, low, high, name):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise Problem(name + " 必须是数值")
    if not low <= result <= high or isinstance(value, bool):
        raise Problem(f"{name} 必须在 {low} 到 {high} 之间")
    return result


class Handler(BaseHTTPRequestHandler):
    server_version = "YegouAgent/1.0"

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def dispatch(self, method):
        gateway = self.server.gateway
        try:
            # No browser-origin mutation; gateway intentionally has no CORS support.
            if method == "POST" and self.headers.get("Origin"):
                raise Problem("控制接口只接受 CLI/API 调用", 403)
            request_path = urllib.parse.urlsplit(self.path).path
            if self.server.server_address[0] in {"127.0.0.1", "::1"}:
                hostname = urllib.parse.urlsplit("http://" + self.headers.get("Host", "")).hostname
                if hostname not in {"127.0.0.1", "localhost", "::1"}:
                    raise Problem("控制服务 Host 不匹配", 403)
            if request_path != "/api/agent/health" and self.server.token and self.headers.get("Authorization") != "Bearer " + self.server.token:
                raise Problem("需要 Bearer token", 401)
            route = urllib.parse.urlsplit(self.path).path.strip("/").split("/")
            data = {}
            if method == "POST":
                length = int(self.headers.get("Content-Length", 0))
                if length < 0 or length > 2 * 1024 * 1024:
                    raise Problem("JSON 请求过大", 413)
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(data, dict):
                    raise Problem("请求必须是 JSON 对象")
            if route == ["api", "agent", "health"] and method == "GET":
                result = {"ok": True, "service": "yegou-agent", "version": 1}
            elif route == ["api", "agent", "capabilities"] and method == "GET":
                result = {"workflow": "video-redub", "background": ["remove", "keep"],
                          "modes": ["cross_lingual", "zero_shot", "instruct2"],
                          "voices": "GET/POST /api/agent/voices", "jobs": "GET/POST /api/agent/jobs",
                          "job_actions": ["retry", "resume", "cancel"],
                          "service_actions": ["start", "stop", "release"],
                          "job_schema": JOB_SCHEMA}
            elif route == ["api", "agent", "voices"]:
                result = gateway.voices() if method == "GET" else gateway.add_voice(data)
            elif route == ["api", "agent", "jobs"]:
                result = [gateway.get(id) for id in reversed(list(gateway.jobs))] if method == "GET" else gateway.submit(data, self.headers.get("Idempotency-Key"))
            elif route[:3] == ["api", "agent", "jobs"] and len(route) == 4 and method == "GET":
                result = gateway.get(route[3])
            elif route[:3] == ["api", "agent", "jobs"] and len(route) == 6 and route[4] == "files" and method == "GET":
                job = gateway.get(route[3])
                key = route[5]
                if key == "quality":
                    return self.respond(200, job.get("quality", {}))
                if key not in {"video", "voice_track", "mix_track"} or not job.get("result", {}).get("generation", {}).get(key):
                    raise Problem("输出文件尚未生成", 404)
                return self.send_file(local_path(job["result"]["generation"][key]))
            elif route[:3] == ["api", "agent", "voices"] and len(route) == 5 and route[4] == "reference" and method == "GET":
                return self.send_file(Path(gateway.voice(route[3])["reference"]))
            elif route[:3] == ["api", "agent", "jobs"] and len(route) == 5 and method == "POST":
                result = gateway.action(route[3], route[4], data)
            elif route == ["api", "agent", "service"] and method == "GET":
                result = gateway.service_status()
            elif route == ["api", "agent", "policy"]:
                with gateway.lock:
                    if method == "POST":
                        if set(data) != {"idle_seconds"}:
                            raise Problem("仅接受 idle_seconds（0 关闭自动停机）")
                        value = number(data["idle_seconds"], 0, 86400, "idle_seconds")
                        if value != int(value):
                            raise Problem("idle_seconds 必须是整数")
                        gateway.idle_seconds = int(value)
                        atomic_json(gateway.state / "policy.json", {"idle_seconds": int(value)})
                    result = {"idle_seconds": gateway.idle_seconds}
            elif route[:3] == ["api", "agent", "service"] and len(route) == 4 and method == "POST":
                action = route[3]
                if action == "start":
                    result = gateway.start()
                elif action == "stop":
                    result = gateway.stop()
                elif action == "release":
                    with gateway.lifecycle, gateway.lock:
                        if gateway.active:
                            raise Problem("任务运行中，不能释放模型", 409)
                        result = backend("POST", "/api/system/release-memory")
                else:
                    raise Problem("未知服务操作", 404)
            else:
                raise Problem("接口不存在", 404)
            self.respond(200, result)
        except Problem as exc:
            self.respond(exc.status, {"error": str(exc)})
        except (ValueError, TypeError, AttributeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception as exc:
            self.respond(500, {"error": f"{type(exc).__name__}: {exc}"})

    def respond(self, status, result):
        raw = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(raw)

    def send_file(self, path):
        if not path.is_file():
            raise Problem("输出文件不存在或已被移动", 404)
        with path.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(path.name))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                shutil.copyfileobj(stream, self.wfile, 1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--idle-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("YEGOU_AGENT_TOKEN"):
        parser.error("非本机监听必须设置 YEGOU_AGENT_TOKEN")
    if args.idle_seconds < 0:
        parser.error("idle-seconds 不能为负数")
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "gateway.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("相同状态目录已有控制服务运行")
        gateway = Gateway(idle_seconds=args.idle_seconds)
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        server.gateway = gateway
        server.token = os.environ.get("YEGOU_AGENT_TOKEN", "")
        worker = threading.Thread(target=gateway.worker, daemon=True, name="agent-worker")
        worker.start()
        try:
            server.serve_forever()
        finally:
            gateway.closing.set()
            server.server_close()


if __name__ == "__main__":
    main()
