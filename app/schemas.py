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
