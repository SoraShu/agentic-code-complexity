"""Pydantic schemas for task records, repo context, and judge output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


class Hunk(BaseModel):
    """A single hunk in a unified-diff patch."""

    old_start: int
    old_len: int
    new_start: int
    new_len: int


class FilePatch(BaseModel):
    """Per-file portion of a diff."""

    path: str
    hunks: list[Hunk]


class TaskRecord(BaseModel):
    """Everything we extract from a single harbor task directory."""

    instance_id: str
    repo: str
    base_commit: str
    environment_setup_commit: str
    difficulty: Difficulty | str
    instruction_md: str
    gold_patch: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    touched_files: list[FilePatch]


class FileSnippet(BaseModel):
    """A windowed snippet from a touched file in the repo."""

    path: str
    content: str
    truncated: bool = False
    note: str = ""


class RepoContext(BaseModel):
    """Pre-rendered repo context handed to the prompt builder."""

    repo: str
    commit: str
    file_snippets: list[FileSnippet] = Field(default_factory=list)
    repo_tree: str = ""
    error: str | None = None


class Subfactors(BaseModel):
    reasoning: int = Field(ge=1, le=5)
    scope: int = Field(ge=1, le=5)
    ambiguity: int = Field(ge=1, le=5)
    domain_knowledge: int = Field(ge=1, le=5)
    test_strictness: int = Field(ge=1, le=5)


class ReasoningNotes(BaseModel):
    why_easy: list[str] = Field(default_factory=list)
    why_hard: list[str] = Field(default_factory=list)
    coverage_check: str = ""
    weigh: str = ""


class JudgeOutput(BaseModel):
    """Validated JSON returned by the LLM judge."""

    p_success: float = Field(ge=0.0, le=1.0)
    complexity_score: int = Field(ge=1, le=10)
    subfactors: Subfactors
    rationale: str
    reasoning_notes: ReasoningNotes | None = None
