"""Import and normalize external subtitle / visual subtitle timelines.

The visual analysis project already exports a useful JSON schema.  This module
keeps that schema at the boundary and turns SRT, VTT, and JSON into one small
internal representation for the silent-video dubbing branch.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any


_TIME_ARROW = re.compile(r"^\s*(.*?)\s*-->\s*(.*?)\s*$")
_TAG_RE = re.compile(r"<[^>]+>")
_ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")


def _parse_time(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("时间不能是布尔值")
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("时间为空")
        text = text.replace(",", ".")
        parts = text.split(":")
        try:
            if len(parts) == 1:
                result = float(parts[0])
            elif len(parts) == 2:
                result = float(parts[0]) * 60.0 + float(parts[1])
            elif len(parts) == 3:
                result = float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"无法解析时间：{value}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"时间无效：{value}")
    return result


def _clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = html.unescape(str(value or ""))
    text = _ASS_TAG_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line).strip()


def _first(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _normalise_row(item: Any, index: int, source_format: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    start_raw = _first(item, "start_sec", "start", "start_time", "begin", "from")
    end_raw = _first(item, "end_sec", "end", "end_time", "stop", "to")
    if end_raw is None:
        duration_raw = _first(item, "duration_sec", "duration")
        if duration_raw is not None and start_raw is not None:
            try:
                end_raw = _parse_time(start_raw) + _parse_time(duration_raw)
            except ValueError:
                return None
    text = _clean_text(_first(item, "dubbing_text", "text", "content", "caption", "subtitle"))
    if start_raw is None or end_raw is None or not text:
        return None
    try:
        start = _parse_time(start_raw)
        end = _parse_time(end_raw)
    except ValueError:
        return None
    if end <= start:
        return None
    source_id = _first(item, "id", "segment_id", "cue_id")
    segment_id = str(_first(item, "segment_id") or f"T{index + 1:04d}")
    speaker = _first(item, "speaker_id", "speaker", "speaker_label")
    confidence_raw = _first(item, "confidence", "score")
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw))) if confidence_raw is not None else 1.0
    except (TypeError, ValueError):
        confidence = 1.0
    return {
        "segment_id": segment_id,
        "source_id": source_id,
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "text": text,
        "dubbing_text": text,
        "speaker_id": str(speaker).strip() if speaker is not None else "default",
        "confidence": round(confidence, 3),
        "source": str(_first(item, "source") or source_format),
        "text_type": str(_first(item, "text_type", "type") or "subtitle"),
        "dubbing_candidate": _as_bool(item.get("dubbing_candidate"), True),
    }


def _json_rows(payload: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError("JSON 时间线必须是数组或对象")
    for key in ("segments", "items", "entries", "subtitles", "captions"):
        value = payload.get(key)
        if isinstance(value, list):
            metadata = {
                key: payload.get(key)
                for key in ("schema_version", "mode", "model", "timing", "media_metadata")
                if key in payload
            }
            return value, metadata
    if _first(payload, "start_sec", "start", "start_time") is not None:
        return [payload], {}
    raise ValueError("JSON 中没有找到 segments / items / entries 时间线数组")


def _parse_json(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 时间线格式错误：{exc}") from exc
    rows, metadata = _json_rows(payload)
    output: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        normalized = _normalise_row(item, index, "json")
        if normalized is not None:
            output.append(normalized)
    return output, metadata


def _parse_srt_vtt(path: Path, source_format: str) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n"))
    output: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.split("\n")]
        arrow_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if arrow_index is None:
            continue
        match = _TIME_ARROW.match(lines[arrow_index])
        if match is None:
            continue
        start_text = match.group(1).split()[0]
        end_text = match.group(2).split()[0]
        text = _clean_text("\n".join(lines[arrow_index + 1 :]))
        if not text:
            continue
        try:
            start = _parse_time(start_text)
            end = _parse_time(end_text)
        except ValueError:
            continue
        if end <= start:
            continue
        output.append(
            {
                "segment_id": f"T{len(output) + 1:04d}",
                "source_id": len(output) + 1,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "text": text,
                "dubbing_text": text,
                "speaker_id": "default",
                "confidence": 1.0,
                "source": source_format,
                "text_type": "subtitle",
            }
        )
    return output


def parse_timeline(path: Path) -> dict[str, Any]:
    """Parse SRT, VTT, or JSON into the workbench timeline schema."""

    if not path.exists() or not path.is_file():
        raise ValueError("时间线文件不存在")
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}
    if suffix == ".json":
        segments, metadata = _parse_json(path)
        source_format = "json"
    elif suffix == ".srt":
        segments = _parse_srt_vtt(path, "srt")
        source_format = "srt"
    elif suffix == ".vtt":
        segments = _parse_srt_vtt(path, "vtt")
        source_format = "vtt"
    else:
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:200]
        if "WEBVTT" in sample.upper() or "-->" in sample:
            segments = _parse_srt_vtt(path, "text")
            source_format = "text"
        else:
            raise ValueError("时间线只支持 .srt、.vtt 或 .json 文件")

    segments.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"]), item["segment_id"]))
    used_ids: set[str] = set()
    for index, item in enumerate(segments):
        base_id = str(item.get("segment_id") or f"T{index + 1:04d}")
        segment_id = base_id
        suffix = 2
        while segment_id in used_ids:
            segment_id = f"{base_id}_{suffix}"
            suffix += 1
        item["segment_id"] = segment_id
        used_ids.add(segment_id)
    warnings: list[str] = []
    if not segments:
        raise ValueError("时间线中没有有效的字幕片段")
    timing = metadata.get("timing") if isinstance(metadata, dict) else None
    if isinstance(timing, dict) and timing.get("timestamp_precision_sec"):
        try:
            precision = float(timing["timestamp_precision_sec"])
            if precision >= 0.5:
                warnings.append(f"导入时间线的估计精度约为 {precision:g} 秒，导出前建议抽查边界")
        except (TypeError, ValueError):
            pass
    if any(float(right["start_sec"]) < float(left["end_sec"]) for left, right in zip(segments, segments[1:])):
        warnings.append("时间线存在相互覆盖的字幕片段；配音会叠加在同一时间轴上")
    return {
        "schema_version": "voice-workbench-timeline/v1",
        "source_path": str(path),
        "source_format": source_format,
        "source_metadata": metadata,
        "segment_count": len(segments),
        "warnings": warnings,
        "segments": segments,
    }
