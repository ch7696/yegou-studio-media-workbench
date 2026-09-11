from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TTSGenerateRequest(BaseModel):
    speaker_asset_id: str
    text: str = Field(min_length=1, max_length=2000)
    reference_text: str = Field(default="", max_length=2000)
    mode: Literal["zero_shot", "cross_lingual", "instruct2"] = "zero_shot"
    prompt_instruction: str = Field(default="You are a helpful assistant.", max_length=500)
    target_duration: float = Field(default=5.0, ge=0.5, le=120.0)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    text_frontend: bool = True
    fit_duration: bool = False
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class DubbingSegmentEdit(BaseModel):
    """Editable fields for one item in the generated dubbing timeline."""

    segment_id: str = Field(min_length=1, max_length=80)
    text: str | None = Field(default=None, max_length=2000)
    dubbing_text: str | None = Field(default=None, max_length=2000)
    speaker_id: str | None = Field(default=None, min_length=1, max_length=80)
    start_sec: float | None = Field(default=None, ge=0.0)
    end_sec: float | None = Field(default=None, gt=0.0)


class DubbingGenerateRequest(BaseModel):
    """Generation controls kept in the API so the WebUI does not hard-code them."""

    edits: list[DubbingSegmentEdit] = Field(default_factory=list)
    selected_ids: list[str] | None = None
    mode: Literal["zero_shot", "cross_lingual", "instruct2"] = "zero_shot"
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    adaptive_speed: bool = True
    fit_to_slot: bool = True
    prompt_instruction: str = Field(default="You are a helpful assistant.", max_length=500)
    text_frontend: bool = True
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    export_video: bool = True
    target_speaker_id: str = Field(default="all", min_length=1, max_length=80)
    target_original_volume: float = Field(default=0.0, ge=0.0, le=1.5)
    other_speaker_volume: float = Field(default=1.0, ge=0.0, le=1.5)
    background_volume: float = Field(default=1.0, ge=0.0, le=2.0)


class DubbingTimelineSaveRequest(BaseModel):
    edits: list[DubbingSegmentEdit] = Field(default_factory=list)
