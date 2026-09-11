from __future__ import annotations

import gc
import sys
import threading
from pathlib import Path

import torch

from .audio import merge_segments
from .config import DIAR_REPO, MODEL_CACHE


DIAR_LOCK = threading.Lock()


def diarize(
    path: Path,
    speaker_num: int | None,
    *,
    include_overlap: bool = False,
) -> list[list[float | int]]:
    """Run the official 3D-Speaker diarization entry point.

    The overlap detector is optional in the official pipeline and can require
    an additional pyannote model/token.  The full dubbing path requests it but
    falls back to the regular VAD/embedding/clustering path when unavailable.
    """

    with DIAR_LOCK:
        if DIAR_REPO not in sys.path:
            sys.path.insert(0, DIAR_REPO)
        from speakerlab.bin.infer_diarization import Diarization3Dspeaker

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            pipeline = Diarization3Dspeaker(
                device=device,
                include_overlap=include_overlap,
                speaker_num=speaker_num,
                model_cache_dir=MODEL_CACHE,
            )
        except Exception:
            if not include_overlap:
                raise
            print("3D-Speaker 重叠语音模块不可用，回退到基础说话人分段。", flush=True)
            pipeline = Diarization3Dspeaker(
                device=device,
                include_overlap=False,
                speaker_num=speaker_num,
                model_cache_dir=MODEL_CACHE,
            )
        try:
            return merge_segments(pipeline(str(path)))
        finally:
            del pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
