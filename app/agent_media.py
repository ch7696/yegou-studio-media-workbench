"""CLI media operations using the deployed engine's FFmpeg and fonts."""
import json
from pathlib import Path
import re

from .agent_service import CONTAINERS, local_path, run


def extract(source, output, start, seconds):
    source, output = str(local_path(source)) if not str(source).startswith('/state/') else str(source), local_path(output)
    if start < 0 or not 0 < seconds <= 30:
        raise ValueError('Reference start must be >= 0 and duration within (0, 30]')
    output.parent.mkdir(parents=True, exist_ok=True)
    run('docker', 'exec', CONTAINERS[0], 'ffmpeg', '-v', 'error', '-y', '-ss', str(start), '-i', source,
        '-t', str(seconds), '-vn', '-ac', '1', '-ar', '24000', '-c:a', 'pcm_s16le', str(output))
    return {'reference': str(output), 'source': source, 'start': start, 'seconds': seconds}


def stamp(seconds):
    millis = round(float(seconds)*1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02},{millis:03}'


def mux(source, audio, output):
    source, audio, output = local_path(source), local_path(audio), local_path(output)
    if source == output:
        raise ValueError('Cannot overwrite input video')
    output.parent.mkdir(parents=True, exist_ok=True)
    code = 'from pathlib import Path; import sys; from app.dubbing import _mux_video; assert _mux_video(*map(Path,sys.argv[1:]))'
    run('docker','exec',CONTAINERS[0],'python3','-c',code,str(source),str(audio),str(output),timeout=1800)
    return {'video':str(output),'audio':str(audio)}


def render(config, output):
    config, output = local_path(config), local_path(output)
    output.mkdir(parents=True, exist_ok=True)
    code = 'import sys; from app.agent_media import render_in_engine; render_in_engine(sys.argv[1],sys.argv[2])'
    return json.loads(run('docker','exec',CONTAINERS[0],'python3','-c',code,str(config),str(output),timeout=1800))


def transcribe(source, output):
    source, output=local_path(source),local_path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    code=('from pathlib import Path; import sys,json; from app.asr import transcribe_detailed; '
          'result=transcribe_detailed([Path(sys.argv[1])],return_time_stamps=True,raise_on_error=True); '
          'Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")')
    run('docker','exec',CONTAINERS[0],'python3','-c',code,str(source),str(output),timeout=1800)
    return json.loads(output.read_text(encoding='utf-8'))


def render_in_engine(config, output):
    """Reassemble validated per-line revisions without re-synthesizing other lines."""
    from .dubbing import _render_voice_track, _render_mixed_track, _mux_video
    data=json.loads(Path(config).read_text(encoding='utf-8-sig'))
    timeline, segments=data['timeline'],data['segments']
    root=Path(output)
    root.mkdir(parents=True,exist_ok=True)
    if not segments or any(not Path(s['audio_path']).is_file() for s in segments):
        raise ValueError('Missing generated audio clips')
    for segment in segments:
        if not 0 <= segment['start_sec'] < segment['end_sec'] <= timeline['duration_sec'] + .1:
            raise ValueError('Invalid generated segment timing')
    source=Path(data['source'])
    names=data.get('outputs',{})
    filenames=[names.get('voice_track','新配音人声.wav'),names.get('mix_track','配音混音.wav'),names.get('video','配音视频.mp4')]
    if any(not isinstance(n,str) or Path(n).name!=n or n in {'','.', '..'} for n in filenames):
        raise ValueError('Output filenames must be simple basenames')
    voice,mix,video=[root/n for n in filenames]
    if source.resolve()==video.resolve():
        raise ValueError('Cannot overwrite source')
    _render_voice_track(segments,timeline['duration_sec'],voice)
    _render_mixed_track(segments,timeline,target_speaker_id='all',target_original_volume=0,
                        other_speaker_volume=0,background_volume=float(data.get('background_volume',1)),target=mix)
    _mux_video(source,mix,video)
    print(json.dumps({'voice_track':str(voice),'mix_track':str(mix),'video':str(video)},ensure_ascii=False))


def caption(source, timeline_path, output, srt):
    source, output, srt = local_path(source), local_path(output), local_path(srt)
    if source == output:
        raise ValueError('Cannot overwrite input video')
    timeline = json.loads(local_path(timeline_path).read_text(encoding='utf-8-sig'))
    cues = timeline.get('segments', [])
    if not cues:
        raise ValueError('Timeline has no segments')
    for cue in cues:
        if not 0 <= float(cue['start_sec']) < float(cue['end_sec']):
            raise ValueError('Invalid subtitle timing')
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text('\n\n'.join(f'{i}\n{stamp(c["start_sec"])} --> {stamp(c["end_sec"])}\n{c["text"]}'
                               for i,c in enumerate(cues,1))+'\n', encoding='utf-8')
    info = json.loads(run('docker','exec',CONTAINERS[0], 'ffprobe','-v','error','-show_entries','stream=codec_type,width,height',
                          '-of','json',str(source)))
    stream = next(s for s in info['streams'] if s['codec_type']=='video')
    # libass font size uses a virtual 384x288 canvas, not source pixel dimensions.
    font_size = 14 if stream['height'] > stream['width'] else 18
    style = f'FontName=Microsoft YaHei,FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00141414,BorderStyle=1,Outline=1.2,Shadow=0,Alignment=2,MarginV=24'
    escaped = str(srt).replace('\\','\\\\').replace(':','\\:').replace("'", "'\\''").replace(',', '\\,')
    vf = f"subtitles=filename='{escaped}':fontsdir=/mnt/c/Windows/Fonts:force_style='{style}'"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem+'.part.mp4')
    try:
        run('docker','exec',CONTAINERS[0], 'ffmpeg','-v','error','-y','-i',str(source),'-vf',vf,
            '-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
            '-c:a','copy','-movflags','+faststart',str(temporary),timeout=1800)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'video':str(output),'srt':str(srt),'cue_count':len(cues)}
