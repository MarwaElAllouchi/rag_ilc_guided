from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatSuggestion(BaseModel):
    label: str
    value: str


class ChatSession(BaseModel):
    child_age: int | None = None
    child_birth_year: int | None = None
    program: str | None = None


class ChatRequest(BaseModel):
    question: str | None = None
    choice: str | None = None
    context: str | None = None
    session: ChatSession | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    intent: str | None = None
    route: str | None = None
    documents: list[dict[str, Any]] = Field(default_factory=list)
    suggestions: list[ChatSuggestion] = Field(default_factory=list)