from __future__ import annotations

from pathlib import Path

from .config import RATE
from .utils import duration, probe, run


def extract_reference(
    source: Path,
    target: Path,
    *,
    start_seconds: float,
    max_seconds: float,
    noise_reduction: float,
    trim_leading_silence: bool,
) -> float:
    """Extract one prompt clip without diarization or transcription.

    CosyVoice's official frontend accepts prompt audio up to 30 seconds. The
    caller caps max_seconds before reaching ffmpeg so a long video remains a
    valid direct-TTS input.
    """

    streams = probe(source).get("streams", [])
    if not source.stat().st_size or not any(item.get("codec_type") == "audio" for item in streams):
        raise RuntimeError("参考文件没有可用音轨")
    total = duration(source)
    start = max(0.0, float(start_seconds))
    if total > 0 and start >= total:
        raise RuntimeError(f"参考起点 {start:g} 秒超过媒体时长 {total:.2f} 秒")
    clip_seconds = min(30.0, max(1.0, float(max_seconds)))
    if total > 0:
        clip_seconds = min(clip_seconds, max(0.1, total - start))

    filters: list[str] = []
    if noise_reduction > 0:
        filters.append(f"afftdn=nr={max(0.0, min(30.0, noise_reduction)):.2f}:nf=-35")
    if trim_leading_silence:
        filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-38dB")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-t",
        f"{clip_seconds:.3f}",
        "-vn",
        "-sn",
        "-dn",
    ]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(["-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(target)])
    target.parent.mkdir(parents=True, exist_ok=True)
    run(command, 300)
    actual = duration(target)
    if actual < 0.5:
        raise RuntimeError("截取的参考音频太短，无法用于 TTS")
    return actual
