from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import RATE
from .utils import duration, run


def extract_clean(
    source: Path,
    target: Path,
    noise_reduction: float = 12.0,
    trim_leading_silence: bool = False,
) -> None:
    """Extract a full mono track while preserving pauses inside a long recording."""

    filters = [
        "pan=mono|c0=0.5*c0+0.5*c1",
        "highpass=f=50",
        "lowpass=f=12000",
    ]
    if noise_reduction > 0:
        filters.append(f"afftdn=nr={noise_reduction:.2f}:nf=-35")
    if trim_leading_silence:
        # Do not use stop_periods: that can cut off the rest of a long video
        # after the first internal pause.
        filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-38dB")

    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-af",
        ",".join(filters),
        "-ac",
        "1",
        "-ar",
        str(RATE),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        run(command)
    except Exception:
        fallback_filters = ["pan=mono|c0=0.5*c0+0.5*c1"]
        if trim_leading_silence:
            fallback_filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-38dB")
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-af",
                ",".join(fallback_filters),
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-c:a",
                "pcm_s16le",
                str(target),
            ]
        )
    if not target.exists() or duration(target) < 0.5:
        raise RuntimeError("没有检测到足够的人声")


def merge_segments(raw: list[list[Any]]) -> list[list[float | int]]:
    merged: list[list[float | int]] = []
    for start, end, speaker in sorted(raw, key=lambda item: (float(item[0]), float(item[1]))):
        start, end, speaker = float(start), float(end), int(speaker)
        if end <= start:
            continue
        if merged and merged[-1][2] == speaker and float(merged[-1][1]) >= start - 0.05:
            merged[-1][1] = max(float(merged[-1][1]), end)
        else:
            merged.append([start, end, speaker])
    return merged


def make_sample(
    wav: Path,
    segments: list[list[float | int]],
    speaker: int,
    max_seconds: float,
    target: Path,
) -> tuple[float, list[list[float]]]:
    data, rate = sf.read(str(wav), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    total = len(data) / rate
    pieces: list[np.ndarray] = []
    ranges: list[list[float]] = []
    used = 0.0

    for start, end, label in segments:
        if int(label) != speaker:
            continue
        start = max(0.0, float(start) - 0.10)
        end = min(total, float(end) + 0.10)
        gap = 0.08 if pieces else 0.0
        available = max_seconds - used - gap
        if available <= 0:
            break
        take = min(end - start, available)
        if take <= 0.15:
            continue
        begin = int(round(start * rate))
        stop = min(len(data), begin + int(round(take * rate)))
        if stop <= begin:
            continue
        if pieces:
            pieces.append(np.zeros(int(round(0.08 * rate)), dtype=np.float32))
        pieces.append(data[begin:stop])
        ranges.append([round(start, 3), round(start + (stop - begin) / rate, 3)])
        used += gap + (stop - begin) / rate

    if not pieces:
        raise RuntimeError(f"说话人 {speaker} 没有可用片段")
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.concatenate(pieces), rate, subtype="PCM_16")
    return duration(target), ranges


def atempo_chain(factor: float) -> str:
    filters: list[str] = []
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={max(0.05, factor):.8f}")
    return ",".join(filters)


def fit_audio(source: Path, target: Path, seconds: float) -> tuple[float, float]:
    raw = duration(source)
    if raw <= 0:
        raise RuntimeError("生成结果为空")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter:a",
            f"{atempo_chain(raw / seconds)},apad=whole_dur={seconds:.6f},atrim=duration={seconds:.6f}",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        300,
    )
    return raw, duration(target)
