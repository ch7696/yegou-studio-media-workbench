#!/usr/bin/env python3
"""Single GPU audio worker; ASR and aligner remain loaded between jobs."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/app")
import audio_timeline

LOCK = threading.Lock()
MAX_TOKENS = int(os.environ.get("ASR_MAX_NEW_TOKENS", "2048"))
MODEL = None
COMPLETED = 0
STARTED = time.time()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json({"error": "Not Found"}, 404)
            return
        self.send_json({"ready": MODEL is not None, "busy": LOCK.locked(),
                        "completed": COMPLETED, "pid": os.getpid(),
                        "started_at": STARTED, "models": ["Qwen3-ASR-0.6B", "Qwen3-ForcedAligner-0.6B"]})

    def do_POST(self) -> None:
        global COMPLETED
        if self.path != "/analyze":
            self.send_json({"error": "Not Found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request length")
            payload = json.loads(self.rfile.read(length))
            job = str(payload.get("job_id", ""))
            if not re.fullmatch(r"[a-f0-9]{32}", job):
                raise ValueError("Invalid job_id")
            folder = (Path("/jobs") / job).resolve()
            if folder.parent != Path("/jobs").resolve():
                raise ValueError("Invalid job directory")
            inputs = list((folder / "input").glob("*"))
            if len(inputs) != 1 or not inputs[0].is_file():
                raise ValueError("Exactly one source is required")
            source = inputs[0].resolve()
            if source.parent != folder / "input":
                raise ValueError("Invalid source")
            language = payload.get("language")
            if language is not None and (not isinstance(language, str) or len(language) > 100):
                raise ValueError("Invalid language")
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
            return
        if not LOCK.acquire(blocking=False):
            self.send_json({"error": "Audio worker is busy"}, 409)
            return
        started = time.perf_counter()
        try:
            output = folder / "output"
            args = ["--source-root", str(source.parent), "--source-file", str(source),
                    "--output-root", str(output), "--force"]
            if language:
                args += ["--language", language]
            audio_timeline.main(args, model=MODEL)
            records = audio_timeline.load_jsonl(output / audio_timeline.RECORDS_FILE_NAME)
            if not records or any(r.get("status") == "failed" for r in records.values()):
                raise RuntimeError(str([r.get("error") for r in records.values()]))
            COMPLETED += 1
            # Keep weights, free transient allocator blocks after each job.
            import torch
            torch.cuda.empty_cache()
            self.send_json({"ok": True, "elapsed_s": round(time.perf_counter() - started, 3),
                            "completed": COMPLETED, "pid": os.getpid()})
        except Exception as exc:
            import torch
            torch.cuda.empty_cache()
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        finally:
            LOCK.release()


if __name__ == "__main__":
    MODEL = audio_timeline.load_asr("/models/asr", "/models/aligner", MAX_TOKENS)
    print("常驻 ASR + ForcedAligner 已就绪", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8766), Handler).serve_forever()
