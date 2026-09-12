from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import RATE
from .utils import duration, run


def _extract_scan_audio(
    source: Path,
    target: Path,
    *,
    start_seconds: float,
    scan_seconds: float,
    noise_reduction: float,
    trim_leading_silence: bool,
) -> float:
    """Create a small, mono scan track from an arbitrary audio or video file."""

    total = duration(source)
    start = max(0.0, float(start_seconds))
    if total > 0 and start >= total:
        raise RuntimeError(f"参考起点 {start:g} 秒超过媒体时长 {total:.2f} 秒")
    length = max(1.0, min(300.0, float(scan_seconds)))
    if total > 0:
        length = min(length, max(0.1, total - start))

    filters = ["highpass=f=50", "lowpass=f=12000"]
    if noise_reduction > 0:
        filters.append(f"afftdn=nr={max(0.0, min(30.0, noise_reduction)):.2f}:nf=-35")
    if trim_leading_silence:
        filters.append("silenceremove=start_periods=1:start_duration=0.2:start_threshold=-38dB")

    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
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
            f"{length:.3f}",
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
        ],
        600,
    )
    actual = duration(target)
    if actual < 0.5:
        raise RuntimeError("参考媒体中没有足够长的音频")
    return actual


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, int(rate)


def _db(value: float) -> float:
    return float(20.0 * math.log10(max(1.0e-8, float(value))))


def _frame_rms(data: np.ndarray, rate: int) -> tuple[np.ndarray, int]:
    frame = max(1, int(round(rate * 0.030)))
    hop = max(1, int(round(rate * 0.010)))
    if data.size < frame:
        padded = np.pad(data, (0, frame - data.size))
    else:
        frame_count = 1 + (data.size - frame) // hop
        required = frame + max(0, frame_count - 1) * hop
        padded = data[:required]
    frame_count = max(1, 1 + (padded.size - frame) // hop)
    starts = np.arange(frame_count, dtype=np.int64) * hop
    ends = starts + frame
    squared = np.square(padded.astype(np.float64, copy=False))
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    means = (cumulative[ends] - cumulative[starts]) / float(frame)
    return np.sqrt(np.maximum(means, 1.0e-12)).astype(np.float32), hop


def _fill_short_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    output = active.copy()
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        end = index
        while end < len(output) and not output[end]:
            end += 1
        if index > 0 and end < len(output) and end - index <= max_gap:
            output[index:end] = True
        index = end
    return output


def _remove_short_runs(active: np.ndarray, min_length: int) -> np.ndarray:
    output = active.copy()
    index = 0
    while index < len(output):
        if not output[index]:
            index += 1
            continue
        end = index
        while end < len(output) and output[end]:
            end += 1
        if end - index < min_length:
            output[index:end] = False
        index = end
    return output


def _energy_intervals(data: np.ndarray, rate: int) -> tuple[list[dict[str, float]], dict[str, float]]:
    rms, hop = _frame_rms(data, rate)
    rms_db = np.asarray([_db(value) for value in rms], dtype=np.float32)
    noise_floor = float(np.percentile(rms_db, 20)) if rms_db.size else -80.0
    if not rms_db.size or float(np.max(rms_db)) < -55.0:
        return [], {
            "noise_floor_db": round(noise_floor, 2),
            "vad_threshold_db": -55.0,
            "speech_detected_sec": 0.0,
            "speech_ratio": 0.0,
        }
    threshold = max(-52.0, min(-28.0, noise_floor + 7.0))
    active = rms_db >= threshold
    if not active.any() and rms_db.size:
        threshold = float(np.percentile(rms_db, 60))
        active = rms_db >= threshold
    active = _fill_short_gaps(active, max_gap=max(1, int(round(0.25 / 0.010))))
    active = _remove_short_runs(active, min_length=max(1, int(round(0.35 / 0.010))))

    intervals: list[dict[str, float]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and active[end]:
            end += 1
        start_sec = max(0.0, index * hop / rate - 0.06)
        end_sec = min(len(data) / rate, (end * hop + max(1, int(round(rate * 0.030)))) / rate + 0.06)
        if end_sec - start_sec >= 0.45:
            intervals.append({"start_sec": start_sec, "end_sec": end_sec})
        index = end

    return intervals, {
        "noise_floor_db": round(noise_floor, 2),
        "vad_threshold_db": round(threshold, 2),
        "speech_detected_sec": round(float(active.sum()) * 0.010, 3),
        "speech_ratio": round(float(active.mean()) if active.size else 0.0, 4),
    }


def _candidate_rows(
    data: np.ndarray,
    rate: int,
    intervals: list[dict[str, float]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for interval in intervals:
        start = float(interval["start_sec"]) + 0.06
        end = float(interval["end_sec"]) - 0.06
        cursor = start
        while cursor < end:
            piece_end = min(end, cursor + 4.5)
            if piece_end - cursor >= 0.65:
                begin = max(0, int(round(cursor * rate)))
                stop = min(len(data), int(round(piece_end * rate)))
                clip = data[begin:stop]
                if len(clip) >= int(rate * 0.6):
                    rms = float(np.sqrt(np.mean(np.square(clip), dtype=np.float64)))
                    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
                    clipping = float(np.mean(np.abs(clip) >= 0.995)) if len(clip) else 0.0
                    loudness_score = max(0.0, min(1.0, (_db(rms) + 45.0) / 35.0))
                    length_score = max(0.0, min(1.0, (piece_end - cursor) / 3.0))
                    score = 0.55 * length_score + 0.30 * loudness_score + 0.15 * (1.0 - min(1.0, clipping * 100.0))
                    candidates.append(
                        {
                            "start_sec": round(cursor, 6),
                            "end_sec": round(piece_end, 6),
                            "duration_sec": round(piece_end - cursor, 6),
                            "rms_db": round(_db(rms), 2),
                            "peak_db": round(_db(peak), 2),
                            "clipping_ratio": round(clipping, 6),
                            "quality_score": round(score, 4),
                        }
                    )
            cursor = piece_end
    return candidates


def _attach_speaker_posteriors(candidates: list[dict[str, Any]], result: dict[str, Any]) -> None:
    frames = list(result.get("frames") or [])
    for candidate in candidates:
        start = float(candidate["start_sec"])
        end = float(candidate["end_sec"])
        weighted: dict[str, float] = {}
        overlap_seconds = 0.0
        covered = 0.0
        for frame in frames:
            frame_start = float(frame.get("start_sec", 0.0))
            frame_end = float(frame.get("end_sec", frame_start))
            overlap = max(0.0, min(end, frame_end) - max(start, frame_start))
            if overlap <= 0:
                continue
            covered += overlap
            if bool(frame.get("overlap")):
                overlap_seconds += overlap
            for speaker, score in (frame.get("scores") or {}).items():
                weighted[str(speaker)] = weighted.get(str(speaker), 0.0) + overlap * max(0.0, float(score))
        if not weighted or covered <= 0:
            candidate.update(
                {
                    "speaker_id": "unknown",
                    "speaker_confidence": 0.0,
                    "speaker_ambiguous": True,
                    "overlap": False,
                }
            )
            continue
        ranked = sorted(weighted.items(), key=lambda item: item[1], reverse=True)
        primary, value = ranked[0]
        confidence = value / max(1.0e-8, covered)
        second = ranked[1][1] / max(1.0e-8, covered) if len(ranked) > 1 else 0.0
        ambiguous = confidence < 0.45 or confidence - second < 0.12 or overlap_seconds / max(covered, 1.0e-8) > 0.25
        candidate.update(
            {
                "speaker_id": primary if not ambiguous else "unknown",
                "speaker_confidence": round(max(0.0, min(1.0, confidence)), 4),
                "speaker_ambiguous": bool(ambiguous),
                "overlap": bool(overlap_seconds / max(covered, 1.0e-8) > 0.25),
                "speaker_scores": {key: round(value / max(1.0e-8, covered), 4) for key, value in ranked},
            }
        )


def _select_candidates(
    candidates: list[dict[str, Any]],
    target_seconds: float,
    *,
    diarization_result: dict[str, Any] | None,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    if not candidates:
        return [], None
    selected_pool = list(candidates)
    selected_speaker: str | None = None
    if diarization_result:
        _attach_speaker_posteriors(selected_pool, diarization_result)
        usable = [
            item
            for item in selected_pool
            if item.get("speaker_id") not in {None, "unknown"}
            and not item.get("speaker_ambiguous")
            and not item.get("overlap")
        ]
        totals: dict[str, float] = {}
        for item in usable:
            speaker = str(item["speaker_id"])
            totals[speaker] = totals.get(speaker, 0.0) + float(item["duration_sec"]) * float(item.get("speaker_confidence", 0.0))
        if totals:
            selected_speaker = max(totals, key=totals.get)
            selected_pool = [item for item in usable if item.get("speaker_id") == selected_speaker]
            if sum(float(item["duration_sec"]) for item in selected_pool) < target_seconds * 0.65:
                warnings.append("自动筛选到的单一说话人素材不足 65%，已保守使用可用片段")
        else:
            warnings.append("说话人置信度不足，未强制指定单一说话人")

    selected: list[dict[str, Any]] = []
    remaining = max(0.8, float(target_seconds))
    for item in sorted(selected_pool, key=lambda value: (-float(value.get("quality_score", 0.0)), float(value["start_sec"]))):
        if remaining <= 0:
            break
        length = min(float(item["duration_sec"]), remaining)
        if length < 0.6:
            continue
        chosen = dict(item)
        chosen["end_sec"] = round(float(chosen["start_sec"]) + length, 6)
        chosen["duration_sec"] = round(length, 6)
        selected.append(chosen)
        remaining -= length
    selected.sort(key=lambda value: float(value["start_sec"]))
    if remaining > 0.5:
        warnings.append(f"有效单人语音只有约 {target_seconds - remaining:.1f} 秒，低于目标 {target_seconds:g} 秒")
    return selected, selected_speaker


def _assemble(data: np.ndarray, rate: int, selected: list[dict[str, Any]], target: Path) -> tuple[float, dict[str, float]]:
    pieces: list[np.ndarray] = []
    for item in selected:
        begin = max(0, int(round(float(item["start_sec"]) * rate)))
        end = min(len(data), int(round(float(item["end_sec"]) * rate)))
        if end <= begin:
            continue
        clip = np.asarray(data[begin:end], dtype=np.float32).copy()
        fade = min(len(clip) // 2, max(1, int(round(rate * 0.015))))
        if fade > 1:
            clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if pieces:
            pieces.append(np.zeros(max(1, int(round(rate * 0.08))), dtype=np.float32))
        pieces.append(clip)
    if not pieces:
        raise RuntimeError("没有可拼接的清晰人声片段")
    audio = np.concatenate(pieces).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        audio *= 0.96 / peak
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, rate, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.995)) if audio.size else 0.0
    return duration(target), {
        "rms_db": round(_db(rms), 2),
        "peak_db": round(_db(peak), 2),
        "clipping_ratio": round(clipping, 6),
    }


def prepare_reference(
    source: Path,
    target: Path,
    *,
    work_root: Path,
    start_seconds: float,
    scan_seconds: float,
    target_seconds: float,
    noise_reduction: float,
    trim_leading_silence: bool,
    auto_select: bool = True,
    detect_speakers: bool = True,
) -> dict[str, Any]:
    """Prepare a clean CosyVoice prompt from an audio or video source.

    The fast path uses a lightweight energy VAD. Long or multi-speaker inputs
    can additionally use the configured diarization backend; any diarization
    failure is recorded and the VAD result remains available as a fallback.
    """

    scan_path = work_root / "扫描音频.wav"
    actual_scan_seconds = _extract_scan_audio(
        source,
        scan_path,
        start_seconds=start_seconds,
        scan_seconds=scan_seconds,
        noise_reduction=noise_reduction,
        trim_leading_silence=False if auto_select else trim_leading_silence,
    )
    data, rate = _read_mono(scan_path)
    intervals, vad_info = _energy_intervals(data, rate)
    candidates = _candidate_rows(data, rate, intervals)
    warnings: list[str] = []
    if auto_select and not candidates:
        raise RuntimeError("没有检测到足够清晰的人声，请扩大扫描范围或换一段参考媒体")

    diarization_result: dict[str, Any] | None = None
    speaker_detection: dict[str, Any] = {
        "requested": bool(detect_speakers and auto_select),
        "attempted": False,
        "backend": None,
        "fallback": False,
        "reason": "",
    }
    if detect_speakers and auto_select and actual_scan_seconds >= 12.0 and len(candidates) >= 2:
        speaker_detection["attempted"] = True
        try:
            from .diarization import run_diarization

            diarization_result = run_diarization(scan_path, None, include_overlap=True)
            speaker_detection["backend"] = diarization_result.get("actual_backend", diarization_result.get("backend"))
            speaker_detection["fallback"] = bool(diarization_result.get("fallback"))
            speaker_detection["reason"] = str(diarization_result.get("fallback_reason") or "")
            if speaker_detection["fallback"]:
                warnings.append(
                    f"说话人筛选使用了回退后端 {speaker_detection['backend'] or 'unknown'}"
                )
        except Exception as exc:
            speaker_detection["reason"] = f"{type(exc).__name__}: {exc}"
            warnings.append("自动说话人筛选不可用，已退回语音活动检测结果")

    if auto_select:
        selected, selected_speaker = _select_candidates(
            candidates,
            target_seconds=max(1.0, min(30.0, float(target_seconds))),
            diarization_result=diarization_result,
            warnings=warnings,
        )
    else:
        # Compatibility path: keep the explicitly requested clip, but still
        # pass it through the same mono, band-limited extraction stage.
        explicit_end = min(len(data) / rate, 30.0, max(1.0, float(scan_seconds)))
        selected = [
            {
                "start_sec": 0.0,
                "end_sec": round(explicit_end, 6),
                "duration_sec": round(explicit_end, 6),
                "quality_score": 0.0,
            }
        ]
        selected_speaker = None
        if explicit_end < float(target_seconds) * 0.65:
            warnings.append("手动参考片段短于建议时长")
    if not selected:
        raise RuntimeError("没有找到可用于音色复刻的单人语音片段")

    actual_reference_seconds, selected_metrics = _assemble(data, rate, selected, target)
    if actual_reference_seconds < 0.8:
        raise RuntimeError("生成的参考音频太短，无法用于音色复刻")
    if selected_metrics["rms_db"] < -55.0:
        raise RuntimeError("参考片段几乎没有有效声音，请换一段人声更清楚的媒体")
    if selected_metrics["clipping_ratio"] > 0.005:
        warnings.append("参考音频存在削波，建议换用更清晰的原始音频")
    if selected_metrics["rms_db"] < -35:
        warnings.append("参考音频平均响度偏低，可能影响音色稳定性")
    if vad_info["speech_ratio"] > 0.98:
        warnings.append("整段音频几乎没有静音间隔，无法仅靠 VAD 完全排除背景声")

    absolute_ranges = [
        [
            round(float(start_seconds) + float(item["start_sec"]), 3),
            round(float(start_seconds) + float(item["end_sec"]), 3),
        ]
        for item in selected
    ]
    quality_score = float(np.mean([float(item.get("quality_score", 0.0)) for item in selected])) if selected else 0.0
    return {
        "method": "energy_vad" if not diarization_result else "energy_vad+diarization",
        "scan_start_sec": round(max(0.0, float(start_seconds)), 3),
        "scan_duration_sec": round(actual_scan_seconds, 3),
        "target_duration_sec": round(max(1.0, min(30.0, float(target_seconds))), 3),
        "reference_duration_sec": round(actual_reference_seconds, 3),
        "selected_speaker_id": selected_speaker,
        "selected_ranges_sec": absolute_ranges,
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "speaker_detection": speaker_detection,
        "quality": {
            **vad_info,
            **selected_metrics,
            "score": round(max(0.0, min(1.0, quality_score)), 4),
            "warnings": warnings,
        },
        "candidates": candidates,
        "selected_candidates": selected,
        "reference_audio": str(target),
    }
