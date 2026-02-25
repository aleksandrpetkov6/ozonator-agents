from typing import Any

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    external_task_id: str | None = Field(default=None)
    source_agent: str = Field(default="AA")
    target_agent: str
    task_type: str
    priority: int = Field(default=100, ge=1, le=1000)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: int | None = Field(default=None)
