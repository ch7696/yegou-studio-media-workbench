#!/usr/bin/env python3
"""Agent client for the existing WebUI job queue; stdout is always JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import quote

import requests


TERMINAL = {"done", "failed", "cancelled"}


def emit(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--url", default=os.environ.get("YEGOU_URL", "http://127.0.0.1:7877"))
    commands = result.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="upload one video and wait for results")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--mode", choices=("director", "subtitle"), default="director")
    analyze.add_argument("--start-sec", type=float, default=0)
    analyze.add_argument("--end-sec", type=float)
    analyze.add_argument("--release-gpu", action="store_true")
    analyze.add_argument("--no-wait", action="store_true")
    status = commands.add_parser("status", help="fetch one job")
    status.add_argument("job_id")
    wait = commands.add_parser("wait", help="wait for an existing job")
    wait.add_argument("job_id")
    for command in (analyze, wait):
        command.add_argument("--timeout", type=positive, default=7200, help="wait deadline in seconds")
        command.add_argument("--poll-interval", type=positive, default=1)
    return result


def main() -> int:
    args_parser = parser()
    args = args_parser.parse_args()
    base = args.url.rstrip("/")
    session = requests.Session()
    # Local job traffic should not pass through a shell's HTTP proxy.
    session.trust_env = False
    job = None

    def request(method: str, path: str, **kwargs) -> dict:
        response = session.request(method, base + path, timeout=60, **kwargs)
        try:
            value = response.json()
        except ValueError:
            raise RuntimeError(f"HTTP {response.status_code}: response is not JSON") from None
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {value.get('error', value)}")
        return value

    def output(value: dict) -> dict:
        value = dict(value)
        value["files"] = [dict(item, url=base + item["url"]) for item in value.get("files", [])]
        return value

    try:
        if args.command == "analyze":
            source = args.source.expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"视频不存在：{source}")
            if not math.isfinite(args.start_sec) or args.start_sec < 0:
                raise ValueError("start-sec must be finite and >= 0")
            if args.end_sec is not None and (not math.isfinite(args.end_sec) or args.end_sec <= args.start_sec):
                raise ValueError("end-sec must be finite and > start-sec")
            params = {"mode": args.mode, "start_sec": args.start_sec, "release_gpu": int(args.release_gpu)}
            if args.end_sec is not None:
                params["end_sec"] = args.end_sec
            with source.open("rb") as stream:
                job = request("POST", "/api/jobs", params=params, files={"video": (source.name, stream, "application/octet-stream")})
            print(f"submitted job_id={job['id']}", file=sys.stderr, flush=True)
            if args.no_wait:
                emit(output(job))
                return 0
        else:
            job = request("GET", "/api/jobs/" + quote(args.job_id, safe=""))

        if args.command != "status":
            deadline = time.monotonic() + args.timeout
            last_phase = None
            while job["status"] not in TERMINAL:
                phase = (job["status"], job.get("phase"))
                if phase != last_phase:
                    print(f"job_id={job['id']} {phase[0]}: {phase[1]}", file=sys.stderr, flush=True)
                    last_phase = phase
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    emit(dict(output(job), cli_error="wait timeout; job continues on server"))
                    return 124
                time.sleep(min(args.poll_interval, remaining))
                job = request("GET", "/api/jobs/" + quote(job["id"], safe=""))
        emit(output(job))
        return 1 if job["status"] in {"failed", "cancelled"} else 0
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        emit({"error": str(exc), "job_id": job.get("id") if job else None})
        return 2
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
