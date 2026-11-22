"""Unit tests for the prompt builder."""
from __future__ import annotations

from llm_judge.prompt import SYSTEM_PROMPT, build_user_prompt, estimate_tokens
from llm_judge.schema import FilePatch, Hunk, TaskRecord


def _make_record() -> TaskRecord:
    patch = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -1,2 +1,3 @@
 a
+b
 c
"""
    return TaskRecord(
        instance_id="dummy__1",
        repo="acme/widget",
        base_commit="abc",
        environment_setup_commit="abc",
        difficulty="<15 min fix",
        instruction_md="Fix the off-by-one in x.py.",
        gold_patch=patch,
        test_patch="",
        fail_to_pass=["tests/test_x.py::test_ok"],
        pass_to_pass=[],
        touched_files=[FilePatch(path="x.py", hunks=[Hunk(old_start=1, old_len=2, new_start=1, new_len=3)])],
    )


def test_system_prompt_has_schema() -> None:
    assert "p_success" in SYSTEM_PROMPT
    assert "complexity_score" in SYSTEM_PROMPT
    assert "subfactors" in SYSTEM_PROMPT


def test_build_user_prompt_contains_sections() -> None:
    rec = _make_record()
    text = build_user_prompt(rec, ctx=None)
    assert "# Task: dummy__1" in text
    assert "Fix the off-by-one" in text
    assert "x.py" in text
    assert "diff --git" in text
    assert "Reference (gold) patch" in text
    # The built-in difficulty label must NOT leak into the prompt — it would
    # anchor the judge and contaminate calibration against that same label.
    assert "<15 min fix" not in text
    assert "Built-in difficulty" not in text


def test_estimate_tokens_monotone() -> None:
    a = "x" * 100
    b = "x" * 1000
    assert estimate_tokens(b) > estimate_tokens(a)
