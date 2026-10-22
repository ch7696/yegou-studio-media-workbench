"""Native worker; heavy dependencies stay inside the configured ComfyUI Python."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.production import PROFILE, validate_job


@contextlib.contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def synthesize(data, settings, work):
    import numpy as np
    import soundfile as sf
    import torch

    plugin = Path(settings['comfy_root']) / 'custom_nodes/ComfyUI-FishAudioS2'
    # Use TensorBoard's built-in stub in this worker; TensorFlow is not used.
    sys.modules['tensorboard.compat.notf'] = types.ModuleType('tensorboard.compat.notf')
    sys.path.insert(0, str(plugin / 'fish_speech_src'))
    package = types.ModuleType('yegou_fish_node')
    package.__path__ = [str(plugin)]
    sys.modules[package.__name__] = package
    from yegou_fish_node.nodes.voice_clone_node import FishS2VoiceCloneTTS
    from yegou_fish_node.nodes import loader
    from yegou_fish_node.nodes.model_cache import unload_engine
    loader._get_models_base = lambda: Path(settings['model_root'])
    results = []
    try:
        for index, segment in enumerate(data['segments']):
            if segment.get('audio_path'):
                results.append({'index': index, 'audio_path': segment['audio_path'], 'reused': True})
                continue
            ref = data['references'][segment['speaker_id']]
            reference, rate = sf.read(ref['audio'], dtype='float32', always_2d=True)
            if not len(reference) or not np.isfinite(reference).all():
                raise ValueError('Invalid reference audio')
            before = time.monotonic()
            result = FishS2VoiceCloneTTS().generate(
                model_path=PROFILE['tts_model'], text=segment.get('tts_text', segment['text']),
                reference_audio={'waveform': torch.from_numpy(reference.T.copy()).unsqueeze(0), 'sample_rate': rate},
                reference_text=ref['text'], language=segment.get('language', 'auto'), device='cuda',
                precision=PROFILE['precision'], attention=PROFILE['attention'],
                max_new_tokens=int(segment.get('max_new_tokens', PROFILE['max_new_tokens'])), chunk_length=200,
                temperature=float(segment.get('temperature', PROFILE['temperature'])),
                top_p=float(segment.get('top_p', PROFILE['top_p'])),
                repetition_penalty=float(segment.get('repetition_penalty', PROFILE['repetition_penalty'])),
                seed=int(segment.get('seed', PROFILE['seed'])), keep_model_loaded=True,
                offload_to_cpu=False, compile_model=PROFILE['compile_model'])[0]
            audio = result['waveform'][0].cpu().numpy().T
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError('Fish produced empty or non-finite audio')
            peak = float(np.abs(audio).max())
            if peak > .98:
                audio *= .96 / peak
            output = work / f'{index + 1:04}_natural.wav'
            sf.write(output, audio, result['sample_rate'], subtype='PCM_16')
            segment['audio_path'] = str(output)
            results.append({'index': index, 'audio_path': str(output), 'text': segment.get('tts_text', segment['text']),
                            'duration_seconds': len(audio) / result['sample_rate'],
                            'generation_seconds': time.monotonic() - before})
    finally:
        unload_engine()
    return results


def render(data, work, out, ffmpeg, run):
    import math
    import numpy as np
    import soundfile as sf
    from app.agent_media import stamp
    # The render worker deliberately uses native FFmpeg, never starts Docker.
    source = Path(data['input'])
    extracted = work / 'source_48k.wav'
    run(['-i', str(source), '-vn', '-ar', '48000', '-ac', '2', '-c:a', 'pcm_f32le', str(extracted)], 'extract')
    length = sf.info(extracted).frames
    rate = 48000
    background = work / 'background_48k.wav'
    run(['-i', data['background_audio'], '-ar', str(rate), '-ac', '2', '-c:a', 'pcm_f32le', str(background)], 'resample-background')
    bg, _ = sf.read(background, dtype='float32', always_2d=True)
    if abs(len(bg) - length) > rate * .1:
        raise ValueError('Background duration differs from source by more than 100ms')
    bg = np.pad(bg[:length], ((0, max(0, length - len(bg))), (0, 0)))
    voice = np.zeros_like(bg)
    alignment, cues = [], []
    for index, seg in enumerate(data['segments']):
        start = float(seg['start_sec'])
        end = float(seg['end_sec'])
        next_start = data['segments'][index + 1]['start_sec'] if index + 1 < len(data['segments']) else length / rate
        if end > length / rate + .02:
            raise ValueError('Timeline exceeds source duration')
        audio, sr = sf.read(seg['audio_path'], dtype='float32', always_2d=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError('Invalid generated clip')
        # Leave timing decisions visible; never truncate speech to force a fit.
        available = min(next_start - start - .08, max(end - start + .28, (end - start) * 1.25))
        tempo = max(1., (len(audio) / sr) / available) if available > 0 else math.inf
        if tempo > PROFILE['max_alignment_tempo']:
            raise ValueError(f'Segment {index + 1} needs a retake or timeline edit: tempo {tempo:.3f}')
        aligned = work / f'{index + 1:04}_aligned.wav'
        run(['-i', seg['audio_path'], '-af', f'atempo={tempo:.8f}', '-ar', str(rate), '-ac', '2',
             '-c:a', 'pcm_f32le', str(aligned)], f'align-{index + 1}')
        audio, _ = sf.read(aligned, dtype='float32', always_2d=True)
        audio *= float(seg.get('gain', 1))
        fade = min(round(.008 * rate), len(audio) // 2)
        if fade:
            audio[:fade] *= np.linspace(0, 1, fade)[:, None]
            audio[-fade:] *= np.linspace(1, 0, fade)[:, None]
        first = round(start * rate)
        last = first + len(audio)
        if last > length or last / rate > next_start - .03:
            raise ValueError(f'Segment {index + 1} overlaps the next segment')
        voice[first:last] += audio
        subtitle_end = min(next_start - .04, max(end, last / rate + .05))
        cues.append(f'{index + 1}\n{stamp(start)} --> {stamp(subtitle_end)}\n{seg["text"]}\n')
        alignment.append({'index': index, 'tempo': tempo, 'start_sec': start, 'end_sec': last / rate,
                          'audio_path': str(aligned)})
    mix = voice + bg * float(data.get('background_volume', PROFILE['background_volume']))
    if not np.isfinite(mix).all():
        raise ValueError('Non-finite mix')
    peak = float(np.abs(mix).max())
    master = min(1., .96 / max(peak, 1e-8))
    mix *= master
    voice *= master
    voice_path, mix_path = out / 'voice.wav', out / 'mix.wav'
    sf.write(voice_path, voice, rate, subtype='PCM_24')
    sf.write(mix_path, mix, rate, subtype='PCM_24')
    srt = out / 'subtitles.srt'
    srt.write_text('\n'.join(cues), encoding='utf-8-sig')
    plain, final = out / 'video.mp4', out / 'video_subtitled.mp4'
    run(['-i', str(source), '-i', str(mix_path), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy',
         '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', str(plain)], 'mux')
    if data.get('burn_subtitles', True):
        # Relative ASCII SRT filename prevents Windows drive/AVFilter escaping bugs.
        style = 'FontName=Microsoft YaHei,FontSize=18,Outline=1.4,Shadow=0.4,MarginV=26'
        run(['-i', str(plain), '-vf', f"subtitles=subtitles.srt:force_style='{style}'", '-c:v', 'libx264',
             '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
             '-movflags', '+faststart', str(final)], 'caption', cwd=out)
    else:
        final = plain
    run(['-i', str(final), '-f', 'null', '-'], 'decode-verification')
    return {'video': str(final), 'voice_track': str(voice_path), 'mix_track': str(mix_path), 'srt': str(srt),
            'alignment': alignment, 'master_gain': master, 'mix_peak': peak * master,
            'duration_seconds': length / rate, 'verification': 'finite audio, no overlap, full output decode passed'}


def main():
    import subprocess
    action, config_path, runtime_path = sys.argv[1:]
    data = validate_job(json.loads(Path(config_path).read_text(encoding='utf-8-sig')), action)
    settings = json.loads(Path(runtime_path).read_text(encoding='utf-8-sig'))
    # WSL interop does not forward arbitrary environment variables to Windows.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if settings.get('torch_home'):
        os.environ['TORCH_HOME'] = settings['torch_home']
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    # A worker deadline releases its own CUDA process even if an inference thread stalls.
    deadline = threading.Timer(int(settings.get('timeout_seconds', 21600)), lambda: os._exit(124))
    deadline.daemon = True
    deadline.start()
    import torch
    import soundfile as sf
    import imageio_ffmpeg
    if not torch.cuda.is_available():
        raise RuntimeError('This production profile requires CUDA')
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    root = Path(data['workspace'])
    job_id = data['job_id']
    work, logs, out = root / '02_work' / job_id, root / '05_logs' / job_id, root / '04_outputs' / job_id
    for path in (work, logs, out):
        path.mkdir(parents=True, exist_ok=True)
    for path in [data.get('input'), data.get('background_audio'),
                 *[ref['audio'] for ref in data.get('references', {}).values()],
                 *[seg['audio_path'] for seg in data['segments'] if seg.get('audio_path')]]:
        if path and not Path(path).is_file():
            raise ValueError(f'Input file does not exist: {path}')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    def run(args, label, cwd=None):
        result = subprocess.run([ffmpeg, '-v', 'error', '-y', *args], cwd=cwd, capture_output=True, timeout=1800)
        (logs / (label + '.log')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'FFmpeg {label}: {result.stderr.decode("utf-8", errors="replace")[-2000:]}')
    begun = time.monotonic()
    with gpu_lock(Path(settings['model_root']).parent / 'yegou-production-gpu.lock'):
        separation = {'model': PROFILE['separation_model'], 'cached': True, 'background_audio': data.get('background_audio')}
        if action == 'run' and not data.get('background_audio'):
            os.environ['VOICE_SEPARATION_MODEL'] = PROFILE['separation_model']
            from app.separation import separate_audio, unload_separation
            source = work / 'source_24k.wav'
            run(['-i', data['input'], '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', str(source)], 'extract-for-separation')
            def load(path):
                audio, rate = sf.read(path, dtype='float32', always_2d=True)
                return torch.from_numpy(audio.T.copy()), rate
            def save(path, tensor, rate):
                sf.write(path, tensor.cpu().numpy().T, rate, subtype='PCM_16')
            try:
                separation = separate_audio(source, work / 'original_vocals.wav', work / 'background.wav',
                                            load_audio=load, save_audio=save)
            finally:
                unload_separation()
            data['background_audio'] = str(work / 'background.wav')
            data['background_method'] = PROFILE['separation_model']
        synthesis = synthesize(data, settings, work) if action in {'run', 'tts'} else []
        (logs / 'resolved-job.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        result = render(data, work, out, ffmpeg, run) if action in {'run', 'render'} else {}
        output = dict(status='done', job_id=job_id, profile=PROFILE, separation=separation, synthesis=synthesis,
                      elapsed_seconds=time.monotonic() - begun, peak_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      **result)
        (logs / 'result.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'production-report.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    deadline.cancel()


if __name__ == '__main__':
    main()
