from __future__ import annotations

from pydantic import BaseModel, Field


class AblationDirectAnswerResponse(BaseModel):
    """Minimal ablation response: direct answer only."""

    direct_answer: str = Field(..., description="Primary answer text.")
