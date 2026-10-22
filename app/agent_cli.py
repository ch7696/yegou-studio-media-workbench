"""JSON CLI for the Yegou agent gateway (standard library only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from .agent_service import ROOT, TERMINAL, local_path, backend, BACKEND_CONTEXT
from .agent_workspace import initialize


def call(base, method, route, data=None, key=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("YEGOU_AGENT_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["YEGOU_AGENT_TOKEN"]
    if key:
        headers["Idempotency-Key"] = key
    req = urllib.request.Request(base + "/api/agent/" + route, method=method,
                                 data=json.dumps(data).encode() if data is not None else None, headers=headers)
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default=os.environ.get("YEGOU_AGENT_URL", "http://127.0.0.1:7868"))
    p.add_argument("--no-autostart", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    workspace = sub.add_parser("workspace")
    workspace.add_argument("action", choices=["init"])
    workspace.add_argument("--root", required=True)
    media = sub.add_parser('media')
    ms = media.add_subparsers(dest='action', required=True)
    extract = ms.add_parser('extract-reference')
    extract.add_argument('--input', required=True)
    extract.add_argument('--output', required=True)
    extract.add_argument('--start', type=float, required=True)
    extract.add_argument('--seconds', type=float, required=True)
    mx = ms.add_parser('mux')
    mx.add_argument('--input', required=True)
    mx.add_argument('--audio', required=True)
    mx.add_argument('--output', required=True)
    render = ms.add_parser('render')
    render.add_argument('--config', required=True)
    render.add_argument('--output', required=True)
    tr = ms.add_parser('transcribe')
    tr.add_argument('--input', required=True)
    tr.add_argument('--output', required=True)
    captions = ms.add_parser('caption')
    captions.add_argument('--input', required=True)
    captions.add_argument('--timeline', required=True)
    captions.add_argument('--output', required=True)
    captions.add_argument('--srt', required=True)
    engine = sub.add_parser("engine", help="Call existing media engine APIs through CLI, including file upload")
    engine.add_argument("--method", choices=["GET", "POST"], default="GET")
    engine.add_argument("--route", required=True)
    engine.add_argument("--config", help="JSON request body or multipart form fields")
    engine.add_argument("--file")
    engine.add_argument("--file-field", default="media")
    engine.add_argument("--save", help="Save the JSON response in the project workspace")
    service = sub.add_parser("service")
    service.add_argument("action", choices=["status", "start", "stop", "release"])
    voices = sub.add_parser("voices")
    vs = voices.add_subparsers(dest="action", required=True)
    vs.add_parser("list")
    add = vs.add_parser("add")
    add.add_argument("--id")
    add.add_argument("--name")
    add.add_argument("--reference", required=True)
    add.add_argument("--text", default="")
    add.add_argument("--start", type=float, default=0)
    add.add_argument("--seconds", type=float, default=10)
    dub = sub.add_parser("dub")
    dub.add_argument("--input")
    dub.add_argument("--voice")
    dub.add_argument("--voices", help="JSON speaker_id → voice_id mapping")
    dub.add_argument("--background", choices=["remove", "keep"])
    dub.add_argument("--output")
    dub.add_argument("--workspace", help="Project task workspace; outputs and task records stay here")
    dub.add_argument("--config", help="JSON request file; other arguments override it")
    dub.add_argument("--idempotency-key")
    dub.add_argument("--wait", action="store_true")
    dub.add_argument("--stop-after", action="store_true", help="Stop model containers after waiting, only if idle")
    dub.add_argument("--wait-timeout", type=int, default=86400)
    jobs = sub.add_parser("jobs")
    js = jobs.add_subparsers(dest="action", required=True)
    js.add_parser("list")
    for action in ("get", "wait", "retry", "resume", "cancel", "download"):
        sp = js.add_parser(action)
        sp.add_argument("id")
        if action in ("retry", "resume"):
            sp.add_argument("--config", help="JSON resume patch file")
        if action == "wait":
            sp.add_argument("--timeout", type=int, default=86400)
        if action == "download":
            sp.add_argument("--file", choices=["video", "voice_track", "mix_track", "quality"], default="video")
            sp.add_argument("--output", required=True)
    sub.add_parser("capabilities")
    policy = sub.add_parser("policy")
    policy.add_argument("--idle-seconds", type=int)
    batch = sub.add_parser("batch")
    batch.add_argument("--config", required=True, help="JSON array of dub request objects")
    batch.add_argument("--wait", action="store_true")
    batch.add_argument("--idempotency-prefix", required=True, help="Stable batch key; item index is appended")
    batch.add_argument("--stop-after", action="store_true")
    args = p.parse_args()
    try:
        if args.command == 'media':
            from . import agent_media
            ensure_gateway(args)
            call(args.url, 'POST', 'service/start', {})
            token = backend('POST','/api/system/lease')['token']
            BACKEND_CONTEXT.lease = token
            finished = threading.Event()
            def heartbeat():
                BACKEND_CONTEXT.lease = token
                while not finished.wait(60):
                    try:
                        backend('GET','/api/health',timeout=5)
                    except Exception:
                        pass
            keeper=threading.Thread(target=heartbeat,daemon=True,name='media-lease-heartbeat')
            keeper.start()
            try:
                if args.action == 'extract-reference':
                    result = agent_media.extract(args.input,args.output,args.start,args.seconds)
                elif args.action == 'mux':
                    result = agent_media.mux(args.input,args.audio,args.output)
                elif args.action == 'render':
                    result = agent_media.render(args.config,args.output)
                elif args.action == 'transcribe':
                    result = agent_media.transcribe(args.input,args.output)
                else:
                    result = agent_media.caption(args.input,args.timeline,args.output,args.srt)
            finally:
                finished.set()
                keeper.join(timeout=6)
                try:
                    backend('POST','/api/system/lease/release')
                finally:
                    BACKEND_CONTEXT.lease = ''
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0
        if args.command == "workspace":
            print(json.dumps(initialize(local_path(args.root)), ensure_ascii=False, indent=2))
            return 0
        if args.command == "engine":
            if not args.route.startswith("/api/") or ".." in args.route:
                raise ValueError("engine route 必须以 /api/ 开头")
            data = json.loads(local_path(args.config).read_text(encoding="utf-8-sig")) if args.config else None
            result = backend(args.method, args.route,
                             payload=data if not args.file else None,
                             fields=data if args.file else None,
                             file=(args.file_field, local_path(args.file)) if args.file else None,
                             timeout=1200)
            if args.save:
                target = local_path(args.save)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        ensure_gateway(args)
        if args.command == "service":
            result = call(args.url, "GET" if args.action == "status" else "POST",
                          "service" + ("" if args.action == "status" else "/" + args.action), {})
        elif args.command == "capabilities":
            result = call(args.url, "GET", "capabilities")
        elif args.command == "policy":
            result = call(args.url, "GET" if args.idle_seconds is None else "POST", "policy",
                          None if args.idle_seconds is None else {"idle_seconds": args.idle_seconds})
        elif args.command == "batch":
            requests = json.loads(local_path(args.config).read_text(encoding="utf-8-sig"))
            if not isinstance(requests, list) or not requests:
                raise ValueError("批量配置必须是非空 JSON 数组")
            if args.stop_after and not args.wait:
                raise ValueError("--stop-after 必须与 --wait 一起使用")
            items = []
            for index, data in enumerate(requests):
                try:
                    items.append(call(args.url, "POST", "jobs", data, args.idempotency_prefix + ":" + str(index)))
                except urllib.error.HTTPError as exc:
                    items.append({"status": "failed", "index": index, "error": exc.read().decode(errors="replace")})
            if args.wait:
                items = [wait(args.url, item["id"], 172800) if item.get("id") else item for item in items]
            result = {"items": items, "status": "failed" if any(j["status"] in {"failed", "interrupted", "cancelled"} for j in items)
                      else "needs_review" if any(j["status"] == "needs_review" for j in items) else "done" if args.wait else "queued"}
            if args.stop_after:
                try:
                    result["shutdown"] = call(args.url, "POST", "service/stop", {})
                except urllib.error.HTTPError as exc:
                    result["shutdown"] = {"error": exc.read().decode(), "status": exc.code}
        elif args.command == "voices":
            data = {k: v for k, v in vars(args).items() if k in {"id", "name", "reference", "text", "start", "seconds"} and v is not None}
            result = call(args.url, "GET" if args.action == "list" else "POST", "voices", data if args.action == "add" else None)
        elif args.command == "dub":
            data = json.loads(local_path(args.config).read_text(encoding="utf-8-sig")) if args.config else {}
            for k in ("input", "voice", "background", "output", "workspace"):
                if getattr(args, k) is not None:
                    data[k] = getattr(args, k)
            if args.voices:
                data["voices"] = json.loads(args.voices)
            if args.stop_after and not args.wait:
                p.error("--stop-after 必须与 --wait 一起使用")
            result = call(args.url, "POST", "jobs", data, args.idempotency_key)
            if args.wait:
                result = wait(args.url, result["id"], args.wait_timeout)
                if args.stop_after:
                    try:
                        result["shutdown"] = call(args.url, "POST", "service/stop", {})
                    except urllib.error.HTTPError as exc:
                        result["shutdown"] = {"error": exc.read().decode(), "status": exc.code}
        else:
            if args.action == "download":
                headers = {}
                if os.environ.get("YEGOU_AGENT_TOKEN"):
                    headers["Authorization"] = "Bearer " + os.environ["YEGOU_AGENT_TOKEN"]
                request = urllib.request.Request(f"{args.url}/api/agent/jobs/{args.id}/files/{args.file}", headers=headers)
                target = local_path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".part")
                try:
                    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
                        while chunk := response.read(1024 * 1024):
                            out.write(chunk)
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
                result = {"file": str(target)}
            elif args.action == "wait":
                result = wait(args.url, args.id, args.timeout)
            else:
                action = args.action
                route = "jobs" if action == "list" else "jobs/" + args.id
                data = None
                if action in ("retry", "resume", "cancel"):
                    route += "/" + action
                    data = json.loads(local_path(args.config).read_text(encoding="utf-8-sig")) if getattr(args, "config", None) else {}
                result = call(args.url, "POST" if data is not None else "GET", route, data)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if isinstance(result, dict) and result.get("status") in {"failed", "interrupted", "cancelled"}:
            return 1
        if isinstance(result, dict) and result.get("status") == "needs_review":
            return 2
        return 0
    except urllib.error.HTTPError as exc:
        print(json.dumps({"error": exc.read().decode(errors="replace"), "http_status": exc.code}, ensure_ascii=False))
        return 1
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


def wait(base, id, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        result = call(base, "GET", "jobs/" + id)
        progress = (result.get("stage"), result.get("phase"), result.get("progress"))
        if progress != previous:
            print(json.dumps({"id": id, "status": result["status"], "progress": progress}, ensure_ascii=False), file=sys.stderr)
            previous = progress
        if result["status"] in TERMINAL:
            return result
        time.sleep(2)
    raise TimeoutError("等待超时；后台任务继续运行，可使用 jobs get/wait 查询")


def ensure_gateway(args):
    try:
        call(args.url, 'GET', 'health')
    except urllib.error.URLError:
        if args.no_autostart or args.url != 'http://127.0.0.1:7868':
            raise
        subprocess.run(['bash',str(ROOT/'scripts/start_agent_gateway.sh')],check=True,
                       stdout=sys.stderr,timeout=30)
        call(args.url,'GET','health')


if __name__ == "__main__":
    sys.exit(main())
