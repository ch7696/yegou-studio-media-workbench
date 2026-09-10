from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .asr import transcribe
from .assets import save_card, speaker_card
from .audio import extract_clean, make_sample
from .config import MANIFESTS, WORK
from .diarization import diarize
from .utils import digest, duration, probe, safe_name, save_json, to_windows_path, to_wsl_path


def analyze_file(
    source: Path,
    *,
    source_id: str,
    original_filename: str,
    output_dir: str,
    speaker_num: int,
    sample_seconds: float,
    noise_reduction: float,
    trim_leading_silence: bool,
) -> dict[str, Any]:
    streams = probe(source).get("streams", [])
    if not source.stat().st_size or not any(item.get("codec_type") == "audio" for item in streams):
        raise RuntimeError("文件没有可用音轨")

    clean_wav = WORK / source_id / "清洁音轨.wav"
    extract_clean(
        source,
        clean_wav,
        noise_reduction=max(0.0, min(30.0, float(noise_reduction))),
        trim_leading_silence=trim_leading_silence,
    )
    segments = diarize(clean_wav, speaker_num if speaker_num > 0 else None)
    if not segments:
        raise RuntimeError("没有识别到说话人")

    labels = sorted({int(item[2]) for item in segments})
    target_root = to_wsl_path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    root = target_root / f"{safe_name(Path(original_filename).stem, 60)}__说话人资产__{source_id[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    # CosyVoice's official frontend rejects prompt audio longer than 30 s.
    limit = max(1.0, min(30.0, float(sample_seconds)))
    source_sha256 = digest(source)
    drafts: list[dict[str, Any]] = []
    samples: list[Path] = []

    for index, label in enumerate(labels):
        asset_id = f"{source_id}-{index:02d}"
        speaker_label = f"说话人{chr(65 + index) if index < 26 else index + 1}"
        sample = root / f"{speaker_label}__清洁参考音色__约{limit:g}秒.wav"
        sample_seconds_actual, selected = make_sample(clean_wav, segments, label, limit, sample)
        speech_total = sum(
            max(0.0, float(end) - float(start))
            for start, end, item_label in segments
            if int(item_label) == label
        )
        drafts.append(
            {
                "asset_id": asset_id,
                "speaker_label": speaker_label,
                "speaker_index": index,
                "cluster_label": label,
                "sample_audio": str(sample),
                "sample_duration_sec": sample_seconds_actual,
                "speaker_speech_duration_sec": speech_total,
                "selected_ranges": selected,
            }
        )
        samples.append(sample)

    texts = transcribe(samples)
    cards: list[dict[str, Any]] = []
    created_at = datetime.now().isoformat(timespec="seconds")
    for draft, reference_text in zip(drafts, texts):
        manifest = {
            "schema_version": "speaker-reference-asset/v1",
            "asset_id": draft["asset_id"],
            "source_asset_id": source_id,
            "source_filename": original_filename,
            "source_path": str(source),
            "source_sha256": source_sha256,
            "clean_audio": str(clean_wav),
            "speaker_label": draft["speaker_label"],
            "speaker_index": draft["speaker_index"],
            "cluster_label": draft["cluster_label"],
            "speaker_speech_duration_sec": round(draft["speaker_speech_duration_sec"], 3),
            "sample_audio": draft["sample_audio"],
            "sample_duration_sec": round(draft["sample_duration_sec"], 3),
            "selected_ranges": draft["selected_ranges"],
            "reference_text": reference_text,
            "asset_root": str(root),
            "generations": [],
            "created_at": created_at,
        }
        save_json(MANIFESTS / f"{draft['asset_id']}.json", manifest)
        save_card(manifest)
        cards.append(speaker_card(manifest))

    source_manifest = {
        "schema_version": "multi-speaker-asset/v1",
        "source_asset_id": source_id,
        "source_filename": original_filename,
        "source_path": str(source),
        "source_sha256": source_sha256,
        "clean_audio": str(clean_wav),
        "clean_duration_sec": round(duration(clean_wav), 3),
        "output_dir": str(target_root),
        "asset_root": str(root),
        "audio_processing": {
            "noise_reduction": max(0.0, min(30.0, float(noise_reduction))),
            "trim_leading_silence": trim_leading_silence,
            "preserve_internal_silence": True,
        },
        "diarization_model": "3D-Speaker CAMPPlus + FSMN-VAD",
        "diarization_segments": [
            {"start_sec": round(float(start), 3), "end_sec": round(float(end), 3), "cluster_label": int(label)}
            for start, end, label in segments
        ],
        "speakers": [item["asset_id"] for item in drafts],
        "created_at": created_at,
    }
    save_json(MANIFESTS / f"{source_id}.json", source_manifest)
    save_json(root / "说话人分析.json", source_manifest)
    return {
        "ok": True,
        "speaker_count": len(cards),
        "source_filename": original_filename,
        "clean_duration_sec": source_manifest["clean_duration_sec"],
        "asset_root": to_windows_path(root),
        "speakers": cards,
        "message": f"识别完成：{len(cards)} 个说话人，资产已保存到 {to_windows_path(root)}",
    }
