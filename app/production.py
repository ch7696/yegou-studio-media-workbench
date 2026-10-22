"""Process-based production CLI: original HDEMUCS separation + Fish S2-Pro."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / 'configs/production-profile.json').read_text(encoding='utf-8'))


def host_path(value):
    value = str(value)
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', value):
        value = '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    return Path(value).expanduser().resolve()


def native_path(path):
    path = host_path(path)
    if os.name == 'nt':
        return str(path)
    return subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()


def validate_job(data, action):
    if not isinstance(data, dict):
        raise ValueError('Job configuration must be a JSON object')
    if data.get('separation_model', PROFILE['separation_model']) != PROFILE['separation_model']:
        raise ValueError('Production uses HDEMUCS_HIGH_MUSDB; BandIt is not a production backend')
    if not data.get('workspace'):
        raise ValueError('workspace is required')
    segments = data.get('segments')
    if not isinstance(segments, list) or not segments:
        raise ValueError('segments must be a nonempty array')
    refs = data.get('references', {})
    previous_end = 0.0
    for segment in segments:
        start, end = float(segment['start_sec']), float(segment['end_sec'])
        if not math.isfinite(start + end) or start < previous_end or start < 0 or end <= start:
            raise ValueError('Segment timing must be finite, ordered and non-overlapping')
        previous_end = end
        if not isinstance(segment.get('text'), str) or not segment['text'].strip():
            raise ValueError('Every segment requires subtitle text')
        if len(segment.get('tts_text', segment['text'])) > 2000:
            raise ValueError('Split long text into shorter segments')
        if action == 'render' and not segment.get('audio_path'):
            raise ValueError('render requires audio_path for every segment')
        if action in {'run', 'tts'} and not segment.get('audio_path'):
            ref = refs.get(segment.get('speaker_id'))
            if not ref or not ref.get('audio') or not str(ref.get('text', '')).strip():
                raise ValueError('Every generated speaker requires reference audio and its exact transcript')
        for key, low, high in [('temperature', .1, 1), ('top_p', .1, 1),
                               ('repetition_penalty', .9, 2), ('gain', .01, 10)]:
            if key in segment and not low <= float(segment[key]) <= high:
                raise ValueError(f'{key} outside [{low}, {high}]')
        if not 1 <= int(segment.get('max_new_tokens', PROFILE['max_new_tokens'])) <= 2048:
            raise ValueError('max_new_tokens must be within [1, 2048]')
    volume = float(data.get('background_volume', PROFILE['background_volume']))
    if not math.isfinite(volume) or not 0 <= volume <= 2:
        raise ValueError('background_volume must be within [0, 2]')
    if action in {'run', 'render'} and not data.get('input'):
        raise ValueError('input video is required')
    if action == 'render' and not data.get('background_audio'):
        raise ValueError('render requires the original HDEMUCS background_audio')
    if data.get('background_audio') and data.get('background_method') != PROFILE['separation_model']:
        raise ValueError('A cached background must declare background_method=HDEMUCS_HIGH_MUSDB')
    return data


def execute(action, config=None, runtime=None):
    runtime_path = host_path(runtime or os.environ.get('YEGOU_PRODUCTION_RUNTIME', ROOT / 'configs/production.local.json'))
    settings = json.loads(runtime_path.read_text(encoding='utf-8-sig')) if runtime_path.exists() else {}
    plugin = host_path(settings['comfy_root']) / 'custom_nodes/ComfyUI-FishAudioS2' if settings.get('comfy_root') else None
    model = host_path(settings['model_root']) / 's2-pro' if settings.get('model_root') else None
    checks = {
        'runtime_config': runtime_path.is_file(),
        'python': bool(settings.get('python')) and host_path(settings['python']).is_file(),
        'comfy_node': plugin is not None and (plugin / 'nodes/voice_clone_node.py').is_file(),
        's2_config': model is not None and (model / 'config.json').is_file(),
        's2_weights': model is not None and all((model / name).is_file() for name in (
            'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'codec.pth')),
    }
    if action == 'status':
        return {'profile': PROFILE, 'runtime': str(runtime_path), 'ready': all(checks.values()), 'checks': checks,
                'scope': 'native production CLI; legacy gateway/WebUI continue to use CosyVoice',
                'lifecycle': 'one command per process; GPU lock, model release on exit; no resident Fish service',
                'check_scope': 'files only; CUDA and model loading are checked during execution'}
    if not all(checks.values()):
        raise ValueError(f'Production runtime is incomplete: {checks}; see docs/PRODUCTION_AUDIO.md')
    data = validate_job(json.loads(host_path(config).read_text(encoding='utf-8-sig')), action)
    workspace = host_path(data['workspace'])
    from .agent_workspace import initialize
    initialize(workspace)
    job_id = 'fish_s2_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    record = workspace / '01_config' / (job_id + '.json')
    logs = workspace / '05_logs' / job_id
    logs.mkdir(parents=True, exist_ok=True)
    # Native worker receives paths valid in its own operating system. No shell interpolation.
    data = dict(data, workspace=native_path(workspace), job_id=job_id)
    for key in ('input', 'background_audio'):
        if data.get(key):
            data[key] = native_path(data[key])
    data['references'] = {key: dict(ref, audio=native_path(ref['audio'])) for key, ref in data.get('references', {}).items()}
    data['segments'] = [dict(seg, **({'audio_path': native_path(seg['audio_path'])} if seg.get('audio_path') else {}))
                        for seg in data['segments']]
    record.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    worker = ROOT / 'app/production_worker.py'
    env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    if settings.get('torch_home'):
        env['TORCH_HOME'] = settings['torch_home']
    python_executable = settings['python'] if os.name == 'nt' else str(host_path(settings['python']))
    command = [python_executable, native_path(worker), action, native_path(record), native_path(runtime_path)]
    report_path = logs / 'result.json'
    try:
        with (logs / 'worker.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                           check=True, timeout=int(settings.get('timeout_seconds', 21600)))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'Production task failed; inspect {logs / "worker.log"}') from exc
    return json.loads(report_path.read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'run', 'tts', 'render'])
    parser.add_argument('--runtime')
    parser.add_argument('--config')
    args = parser.parse_args(argv)
    try:
        if args.action != 'status' and not args.config:
            raise ValueError('--config is required')
        print(json.dumps(execute(args.action, args.config, args.runtime), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
