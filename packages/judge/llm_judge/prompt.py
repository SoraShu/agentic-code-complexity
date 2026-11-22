"""Prompt builder for the LLM judge."""
from __future__ import annotations

from llm_judge.schema import RepoContext, TaskRecord

# Recovered shared three-step rubric used for the thesis's primary results.
from pathlib import Path
SYSTEM_PROMPT = Path(__file__).with_name("three_step_rubric.txt").read_text(encoding="utf-8")

# Soft per-section budgets in *characters* (cheap proxy for tokens — ~4 chars/tok).
_MAX_INSTRUCTION_CHARS = 6_000
_MAX_PATCH_CHARS = 12_000
_MAX_PER_SNIPPET_CHARS = 8_000
_MAX_TOTAL_SNIPPET_CHARS = 40_000
_MAX_TREE_CHARS = 3_000
_MAX_TEST_PATCH_CHARS = 10_000


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars of {label}]"


def build_user_prompt(
    record: TaskRecord,
    ctx: RepoContext | None,
    show_tests: bool = False,
    fewshot_block: str | None = None,
) -> str:
    parts: list[str] = []

    if fewshot_block:
        parts.append(fewshot_block)

    parts.append(f"# Task: {record.instance_id}")
    parts.append(f"Repo: {record.repo}")
    parts.append(
        f"FAIL_TO_PASS: {len(record.fail_to_pass)} test(s); "
        f"PASS_TO_PASS: {len(record.pass_to_pass)} test(s)"
    )
    parts.append("")

    parts.append("## Instruction (what the agent sees)")
    parts.append(_truncate(record.instruction_md.strip(), _MAX_INSTRUCTION_CHARS, "instruction"))
    parts.append("")

    parts.append("## Touched files")
    if record.touched_files:
        for fp in record.touched_files:
            parts.append(f"- {fp.path} ({len(fp.hunks)} hunk(s))")
    else:
        parts.append("(patch did not parse into per-file hunks)")
    parts.append("")

    if ctx is not None and ctx.file_snippets:
        parts.append("## Relevant source (at base_commit — the buggy state)")
        total = 0
        for snip in ctx.file_snippets:
            body = _truncate(snip.content, _MAX_PER_SNIPPET_CHARS, snip.path)
            if total + len(body) > _MAX_TOTAL_SNIPPET_CHARS:
                parts.append(
                    f"### {snip.path}\n[omitted to stay within context budget]"
                )
                continue
            header = f"### {snip.path}"
            if snip.note:
                header += f"  ({snip.note})"
            parts.append(header)
            parts.append("```")
            parts.append(body)
            parts.append("```")
            total += len(body)
        parts.append("")
    elif ctx is not None and ctx.error:
        parts.append(f"## Relevant source\n(unavailable: {ctx.error})\n")

    parts.append("## Reference (gold) patch")
    parts.append("```diff")
    parts.append(_truncate(record.gold_patch, _MAX_PATCH_CHARS, "patch"))
    parts.append("```")
    parts.append("")

    if show_tests:
        parts.append("## Evaluation tests")
        parts.append(
            "The fix is graded by these tests. FAIL_TO_PASS must go from failing "
            "to passing; PASS_TO_PASS must stay passing."
        )
        if record.fail_to_pass:
            parts.append("FAIL_TO_PASS:")
            for t in record.fail_to_pass:
                parts.append(f"  - {t}")
        if record.pass_to_pass:
            shown = record.pass_to_pass[:40]
            parts.append("PASS_TO_PASS:")
            for t in shown:
                parts.append(f"  - {t}")
            if len(record.pass_to_pass) > len(shown):
                parts.append(f"  ... (+{len(record.pass_to_pass) - len(shown)} more)")
        if record.test_patch.strip():
            parts.append("")
            parts.append("### Test patch (the new/changed tests added by the eval)")
            parts.append("```diff")
            parts.append(_truncate(record.test_patch, _MAX_TEST_PATCH_CHARS, "test_patch"))
            parts.append("```")
        parts.append("")

    if ctx is not None and ctx.repo_tree:
        parts.append("## Repo tree (truncated)")
        parts.append("```")
        parts.append(_truncate(ctx.repo_tree, _MAX_TREE_CHARS, "tree"))
        parts.append("```")
        parts.append("")

    parts.append("Return ONLY the JSON object specified in the system prompt.")
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate (~4 chars/token)."""
    return len(text) // 4
