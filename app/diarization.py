"""Speaker diarization backends and a shared frame-level result schema.

The workbench keeps the old 3D-Speaker pipeline available for A/B tests while
adding NVIDIA NeMo's Streaming Sortformer. A caller that needs the old
``[[start, end, label], ...]`` shape can continue to use :func:`diarize`; the
new dubbing pipeline uses :func:`run_diarization` and keeps frame posteriors.
"""

from __future__ import annotations

import gc
import math
import re
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from .audio import merge_segments
from .config import (
    DIAR_REPO,
    DIARIZATION_BACKEND,
    HF_ACCESS_TOKEN,
    MODEL_CACHE,
    SORTFORMER_ACTIVITY_THRESHOLD,
    SORTFORMER_CHUNK_LEN,
    SORTFORMER_CHUNK_RIGHT_CONTEXT,
    SORTFORMER_FIFO_LEN,
    SORTFORMER_MODEL,
    SORTFORMER_OVERLAP_THRESHOLD,
    SORTFORMER_SPKCACHE_UPDATE_PERIOD,
)
from .utils import duration


# Different web jobs must not load two large diarization models onto a 16GB
# card simultaneously.
DIAR_LOCK = threading.RLock()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _canonical_speaker(index: int) -> str:
    return f"S{int(index):02d}"


def _speaker_index(value: Any) -> int | None:
    text = str(value or "")
    match = re.search(r"(?:speaker[_-]?)?(\d+)$", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _normalise_intervals(raw: Any) -> list[dict[str, Any]]:
    """Convert RTTM/list/dict-like segments to internal intervals."""

    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("segments", raw.get("diarization", []))
    output: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start", 0.0))
            end = item.get("end_sec", item.get("end", start))
            label = item.get("cluster_label", item.get("speaker", item.get("label", 0)))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end, label = item[0], item[1], item[2]
        else:
            continue
        start_value = max(0.0, _float(start))
        end_value = max(start_value, _float(end, start_value))
        index = _speaker_index(label)
        if index is None:
            index = int(_float(label, 0.0))
        if end_value - start_value < 0.02:
            continue
        output.append(
            {
                "start_sec": round(start_value, 6),
                "end_sec": round(end_value, 6),
                "cluster_label": int(index),
            }
        )
    return sorted(output, key=lambda value: (value["start_sec"], value["end_sec"], value["cluster_label"]))


def _frames_from_intervals(
    intervals: list[dict[str, Any]],
    duration_sec: float,
    *,
    frame_resolution: float,
    include_overlap: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    labels = sorted({int(item["cluster_label"]) for item in intervals})
    speakers = [_canonical_speaker(index) for index in range(len(labels))]
    label_to_speaker = {label: speakers[pos] for pos, label in enumerate(labels)}
    end_of_audio = max(
        max((_float(item.get("end_sec")) for item in intervals), default=0.0),
        _float(duration_sec),
    )
    step = max(0.01, float(frame_resolution))
    frame_count = max(1, int(math.ceil(end_of_audio / step)))
    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        start = frame_index * step
        end = min(end_of_audio, start + step)
        if end <= start:
            continue
        scores = {speaker: 0.0 for speaker in speakers}
        for item in intervals:
            overlap = max(0.0, min(end, item["end_sec"]) - max(start, item["start_sec"]))
            if overlap <= 0:
                continue
            speaker = label_to_speaker[int(item["cluster_label"])]
            scores[speaker] = 1.0
        active = [speaker for speaker in speakers if scores[speaker] >= 0.5]
        if active and not include_overlap:
            active = [max(active, key=lambda speaker: scores[speaker])]
        frames.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "scores": {key: round(value, 6) for key, value in scores.items()},
                "active_speakers": active,
                "overlap": bool(include_overlap and len(active) > 1),
            }
        )
    return speakers, frames


def _frames_from_scores(
    scores_rows: list[list[float]],
    duration_sec: float,
    *,
    frame_resolution: float,
    include_overlap: bool,
    activity_threshold: float,
    overlap_threshold: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    if not scores_rows:
        return [], []
    speaker_count = max(len(row) for row in scores_rows)
    speakers = [_canonical_speaker(index) for index in range(speaker_count)]
    step = max(0.01, float(frame_resolution))
    end_of_audio = _float(duration_sec, 0.0)
    if end_of_audio <= 0:
        end_of_audio = len(scores_rows) * step
    frames: list[dict[str, Any]] = []
    for index, row in enumerate(scores_rows):
        start = index * step
        if start >= end_of_audio:
            break
        end = min(end_of_audio, start + step)
        values = [max(0.0, min(1.0, _float(row[pos] if pos < len(row) else 0.0))) for pos in range(speaker_count)]
        scores = {speaker: round(values[pos], 6) for pos, speaker in enumerate(speakers)}
        active = [speaker for pos, speaker in enumerate(speakers) if values[pos] >= overlap_threshold]
        if not active:
            # Keep a single weak-but-best candidate for token attribution; it
            # will still be marked low confidence downstream.
            best = max(range(len(values)), key=values.__getitem__, default=-1)
            if best >= 0 and values[best] >= activity_threshold:
                active = [speakers[best]]
        if active and not include_overlap:
            active = [max(active, key=lambda speaker: scores[speaker])]
        frames.append(
            {
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "scores": scores,
                "active_speakers": active,
                "overlap": bool(include_overlap and len(active) > 1),
            }
        )
    # Drop columns that never carry meaningful activity. Keeping them in every
    # frame creates fake choices and unnecessarily enlarges long JSON files.
    active_columns = {
        speaker
        for frame in frames
        for speaker, score in frame["scores"].items()
        if score >= max(0.05, activity_threshold * 0.25)
    }
    if active_columns:
        speakers = [speaker for speaker in speakers if speaker in active_columns]
        for frame in frames:
            frame["scores"] = {speaker: frame["scores"].get(speaker, 0.0) for speaker in speakers}
            frame["active_speakers"] = [speaker for speaker in frame["active_speakers"] if speaker in speakers]
            frame["overlap"] = bool(include_overlap and len(frame["active_speakers"]) > 1)
    return speakers, frames


def _frames_to_segments(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create compatibility intervals from frame activity without losing frames."""

    speakers = sorted({str(speaker) for frame in frames for speaker in frame.get("scores", {})})
    output: list[dict[str, Any]] = []
    sentinel = {"start_sec": 0.0, "end_sec": 0.0, "scores": {}, "active_speakers": []}
    for speaker in speakers:
        start: float | None = None
        end = 0.0
        confidences: list[float] = []
        overlap_seen = False
        for frame in frames + [sentinel]:
            active = speaker in (frame.get("active_speakers") or [])
            if active and start is None:
                start = _float(frame.get("start_sec"))
                confidences = []
                overlap_seen = False
            if active:
                end = _float(frame.get("end_sec"), end)
                confidences.append(_float((frame.get("scores") or {}).get(speaker), 0.0))
                overlap_seen = overlap_seen or bool(frame.get("overlap"))
            elif start is not None:
                cluster = _speaker_index(speaker) or 0
                output.append(
                    {
                        "start_sec": round(start, 6),
                        "end_sec": round(end, 6),
                        "cluster_label": cluster,
                        "speaker_id": speaker,
                        "confidence": round(sum(confidences) / max(1, len(confidences)), 6),
                        "overlap": overlap_seen,
                    }
                )
                start = None
                end = 0.0
                confidences = []
                overlap_seen = False
    return sorted(output, key=lambda value: (value["start_sec"], value["end_sec"], value["cluster_label"]))


def _parse_rttm_lines(raw: Any) -> list[dict[str, Any]]:
    lines: list[str] = []
    if isinstance(raw, str):
        lines = raw.splitlines()
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                lines.extend(item.splitlines())
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                lines.append(" ".join(str(value) for value in item))
    intervals: list[dict[str, Any]] = []
    for line in lines:
        fields = line.strip().split()
        if not fields or fields[0].upper() != "SPEAKER" or len(fields) < 8:
            continue
        start = _float(fields[3], -1.0)
        length = _float(fields[4], -1.0)
        label = fields[7]
        if start < 0 or length <= 0:
            continue
        intervals.append({"start_sec": start, "end_sec": start + length, "cluster_label": _speaker_index(label) or 0})
    return intervals


def _base_result(
    *,
    backend: str,
    model: str,
    speakers: list[str],
    frames: list[dict[str, Any]],
    frame_resolution: float,
    include_overlap: bool,
    overlap_enabled: bool,
    posterior_available: bool,
    fallback: bool = False,
    fallback_reason: str = "",
) -> dict[str, Any]:
    metadata = {
        "backend": backend,
        "model": model,
        "frame_resolution_sec": round(float(frame_resolution), 6),
        "posterior_available": bool(posterior_available),
        "overlap_requested": bool(include_overlap),
        "overlap_enabled": bool(overlap_enabled),
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
    }
    return {
        "backend": backend,
        "model": model,
        "frame_resolution_sec": round(float(frame_resolution), 6),
        "speakers": speakers,
        "frames": frames,
        "segments": _frames_to_segments(frames),
        "overlap_requested": bool(include_overlap),
        "overlap_enabled": bool(overlap_enabled),
        "fallback": bool(fallback),
        "fallback_reason": fallback_reason,
        "posterior_available": bool(posterior_available),
        "metadata": metadata,
    }


class DiarizationBackend(ABC):
    """Common interface for frame-level speaker activity backends."""

    name = "unknown"

    @abstractmethod
    def analyze(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None,
        include_overlap: bool,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        """Release model resources. Backends may override this."""


class ThreeDSpeakerBackend(DiarizationBackend):
    name = "3dspeaker"
    model_name = "3D-Speaker CAMPPlus + FSMN-VAD"

    @staticmethod
    def _load_pipeline(cls: Any, *, device: torch.device, include_overlap: bool, speaker_num: int | None) -> Any:
        return cls(
            device=device,
            include_overlap=include_overlap,
            # The official 3D-Speaker implementation requires this exact
            # argument whenever include_overlap=True.
            hf_access_token=HF_ACCESS_TOKEN or None,
            speaker_num=speaker_num,
            model_cache_dir=MODEL_CACHE,
        )

    def analyze(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None,
        include_overlap: bool,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del config
        if DIAR_REPO not in sys.path:
            sys.path.insert(0, DIAR_REPO)
        from speakerlab.bin.infer_diarization import Diarization3Dspeaker

        pipeline: Any = None
        actual_overlap = bool(include_overlap)
        fallback = False
        fallback_reason = ""
        raw: Any
        try:
            try:
                pipeline = self._load_pipeline(
                    Diarization3Dspeaker,
                    device=device,
                    include_overlap=bool(include_overlap),
                    speaker_num=num_speakers,
                )
                raw = pipeline(str(audio_path))
            except Exception as exc:
                if not include_overlap:
                    raise
                fallback = True
                actual_overlap = False
                fallback_reason = f"{type(exc).__name__}: {exc}"
                print(
                    "[Diarization] 3D-Speaker overlap requested but failed; "
                    f"falling back to normal pipeline. reason={fallback_reason}",
                    flush=True,
                )
                if pipeline is not None:
                    del pipeline
                    pipeline = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                pipeline = self._load_pipeline(
                    Diarization3Dspeaker,
                    device=device,
                    include_overlap=False,
                    speaker_num=num_speakers,
                )
                raw = pipeline(str(audio_path))
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        intervals = _normalise_intervals(merge_segments(raw or []))
        speakers, frames = _frames_from_intervals(
            intervals,
            duration(audio_path),
            frame_resolution=0.1,
            include_overlap=actual_overlap,
        )
        result = _base_result(
            backend=self.name,
            model=self.model_name,
            speakers=speakers,
            frames=frames,
            frame_resolution=0.1,
            include_overlap=bool(include_overlap),
            overlap_enabled=actual_overlap,
            posterior_available=False,
            fallback=fallback,
            fallback_reason=fallback_reason,
        )
        print(
            "[Diarization] actual_pipeline=3dspeaker "
            f"overlap_requested={bool(include_overlap)} overlap_enabled={actual_overlap} "
            f"fallback={fallback} reason={fallback_reason or '-'}",
            flush=True,
        )
        return result


class SortformerBackend(DiarizationBackend):
    name = "sortformer"

    def __init__(self, model_name: str = SORTFORMER_MODEL) -> None:
        self.model_name = model_name
        self.model: Any = None

    def _release_model(self) -> None:
        model = self.model
        self.model = None
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        self._release_model()

    def _load_model(self, device: torch.device) -> Any:
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except Exception as exc:
            raise RuntimeError(
                "NeMo ASR 未安装或导入失败；请重建 V3 镜像安装 nemo_toolkit[asr]。"
            ) from exc
        local_model = Path(self.model_name).expanduser()
        if local_model.is_dir():
            candidates = sorted(local_model.glob("*.nemo"))
            local_model = candidates[0] if candidates else local_model
        if local_model.is_file() and local_model.suffix == ".nemo":
            # NeMo's official local-checkpoint path. This keeps runtime fully
            # offline after the mirror download and avoids a second HF cache.
            model = SortformerEncLabelModel.restore_from(
                restore_path=str(local_model),
                map_location=device,
            )
        else:
            model = SortformerEncLabelModel.from_pretrained(self.model_name)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _configure_streaming(model: Any, config: dict[str, Any]) -> None:
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            return
        values = {
            "chunk_len": config.get("chunk_len", SORTFORMER_CHUNK_LEN),
            "chunk_right_context": config.get("chunk_right_context", SORTFORMER_CHUNK_RIGHT_CONTEXT),
            "fifo_len": config.get("fifo_len", SORTFORMER_FIFO_LEN),
            "spkcache_update_period": config.get("spkcache_update_period", SORTFORMER_SPKCACHE_UPDATE_PERIOD),
        }
        for key, value in values.items():
            if value is not None and hasattr(modules, key):
                setattr(modules, key, int(value))

    def analyze(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None,
        include_overlap: bool,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(config or {})
        if num_speakers and int(num_speakers) > 4:
            print(
                f"[Diarization] Sortformer model supports up to 4 speakers; requested={num_speakers}.",
                flush=True,
            )
        try:
            self.model = self._load_model(device)
            self._configure_streaming(self.model, config)
            print(
                f"[Diarization] actual_pipeline=sortformer model={self.model_name} "
                f"device={device} overlap_requested={bool(include_overlap)}",
                flush=True,
            )
            output = self.model.diarize(
                audio=str(audio_path),
                batch_size=1,
                include_tensor_outputs=True,
                num_workers=0,
                verbose=False,
            )
            predicted_segments = output[0] if isinstance(output, tuple) and len(output) >= 1 else output
            tensor_outputs = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
            raw_tensor: Any = tensor_outputs[0] if tensor_outputs else None
            if isinstance(raw_tensor, torch.Tensor):
                raw_tensor = raw_tensor.detach().float().cpu()
                if raw_tensor.ndim == 3:
                    raw_tensor = raw_tensor[0]
                scores_rows = raw_tensor.tolist()
            else:
                scores_rows = []

            output_factor = max(1, int(getattr(self.model, "output_subsampling_factor", 8)))
            frame_resolution = output_factor * 0.01
            audio_duration = duration(audio_path)
            if scores_rows:
                speakers, frames = _frames_from_scores(
                    scores_rows,
                    audio_duration,
                    frame_resolution=frame_resolution,
                    include_overlap=bool(include_overlap),
                    activity_threshold=_float(
                        config.get("activity_threshold", SORTFORMER_ACTIVITY_THRESHOLD),
                        SORTFORMER_ACTIVITY_THRESHOLD,
                    ),
                    overlap_threshold=_float(
                        config.get("overlap_threshold", SORTFORMER_OVERLAP_THRESHOLD),
                        SORTFORMER_OVERLAP_THRESHOLD,
                    ),
                )
                posterior_available = True
            else:
                # Older NeMo builds can omit tensor outputs even though the
                # diarize() call succeeds. Keep the run usable and explicit.
                intervals = _parse_rttm_lines(predicted_segments)
                if not intervals:
                    intervals = _normalise_intervals(predicted_segments)
                speakers, frames = _frames_from_intervals(
                    intervals,
                    audio_duration,
                    frame_resolution=frame_resolution,
                    include_overlap=bool(include_overlap),
                )
                posterior_available = False
            result = _base_result(
                backend=self.name,
                model=self.model_name,
                speakers=speakers,
                frames=frames,
                frame_resolution=frame_resolution,
                include_overlap=bool(include_overlap),
                overlap_enabled=bool(include_overlap),
                posterior_available=posterior_available,
            )
            result["metadata"]["num_speakers_requested"] = num_speakers
            result["metadata"]["num_speakers_model_limit"] = 4
            result["metadata"]["streaming_config"] = {
                "chunk_len": int(config.get("chunk_len", SORTFORMER_CHUNK_LEN)),
                "chunk_right_context": int(config.get("chunk_right_context", SORTFORMER_CHUNK_RIGHT_CONTEXT)),
                "fifo_len": int(config.get("fifo_len", SORTFORMER_FIFO_LEN)),
                "spkcache_update_period": int(config.get("spkcache_update_period", SORTFORMER_SPKCACHE_UPDATE_PERIOD)),
            }
            return result
        finally:
            self._release_model()


def _with_run_metadata(result: dict[str, Any], *, requested_backend: str, fallback_reason: str = "") -> dict[str, Any]:
    result["requested_backend"] = requested_backend
    result["actual_backend"] = result.get("backend", requested_backend)
    if fallback_reason:
        result["fallback"] = True
        result["fallback_reason"] = fallback_reason
        result.setdefault("metadata", {})["fallback"] = True
        result["metadata"]["fallback_reason"] = fallback_reason
    result.setdefault("metadata", {})["requested_backend"] = requested_backend
    result["metadata"]["actual_backend"] = result.get("backend", requested_backend)
    return result


def run_diarization(
    path: Path,
    speaker_num: int | None,
    *,
    include_overlap: bool = False,
    backend: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a configured backend and return frame-level speaker activity."""

    requested = str(backend or DIARIZATION_BACKEND or "sortformer").strip().lower()
    aliases = {"3d-speaker": "3dspeaker", "3d_speaker": "3dspeaker", "legacy": "3dspeaker"}
    requested = aliases.get(requested, requested)
    if requested not in {"sortformer", "3dspeaker"}:
        raise ValueError(f"不支持的说话人识别后端：{requested}（可选 sortformer 或 3dspeaker）")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with DIAR_LOCK:
        try:
            backend_instance: DiarizationBackend
            if requested == "sortformer":
                backend_instance = SortformerBackend()
            else:
                backend_instance = ThreeDSpeakerBackend()
            result = backend_instance.analyze(
                path,
                num_speakers=speaker_num,
                include_overlap=bool(include_overlap),
                device=device,
                config=config,
            )
            return _with_run_metadata(result, requested_backend=requested)
        except Exception as exc:
            if requested != "sortformer":
                raise
            reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[Diarization] sortformer failed; fallback=3dspeaker reason={reason}",
                flush=True,
            )
            fallback_backend = ThreeDSpeakerBackend()
            result = fallback_backend.analyze(
                path,
                num_speakers=speaker_num,
                include_overlap=bool(include_overlap),
                device=device,
                config=config,
            )
            return _with_run_metadata(
                result,
                requested_backend=requested,
                fallback_reason=f"sortformer unavailable: {reason}",
            )


def diarize(
    path: Path,
    speaker_num: int | None,
    *,
    include_overlap: bool = False,
    backend: str | None = None,
    return_result: bool = False,
) -> list[list[float | int]] | dict[str, Any]:
    """Compatibility wrapper for the original interval-only callers."""

    result = run_diarization(
        path,
        speaker_num,
        include_overlap=include_overlap,
        backend=backend,
    )
    if return_result:
        return result
    return [
        [item["start_sec"], item["end_sec"], item["cluster_label"]]
        for item in result.get("segments", [])
    ]


def backend_status() -> list[dict[str, Any]]:
    """Return cheap capability information for the WebUI health endpoint."""

    try:
        import importlib.util

        nemo_available = importlib.util.find_spec("nemo") is not None
    except Exception:
        nemo_available = False
    return [
        {
            "name": "sortformer",
            "model": SORTFORMER_MODEL,
            "available": nemo_available,
            "recommended": True,
        },
        {
            "name": "3dspeaker",
            "model": "3D-Speaker CAMPPlus + FSMN-VAD",
            "available": True,
            "recommended": False,
            "overlap_token_configured": bool(HF_ACCESS_TOKEN),
        },
    ]
