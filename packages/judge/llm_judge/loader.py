"""Dataset loader and unified-diff patch parser."""
from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

from llm_judge.schema import FilePatch, Hunk, TaskRecord

_DIFF_HEADER = re.compile(r"^diff --git a/(?P<a>\S+) b/(?P<b>\S+)\s*$", re.MULTILINE)
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<os>\d+)(?:,(?P<ol>\d+))? \+(?P<ns>\d+)(?:,(?P<nl>\d+))? @@",
    re.MULTILINE,
)


def parse_patch(patch_text: str) -> list[FilePatch]:
    """Parse a unified diff into a list of FilePatch (path + hunk ranges).

    Skips pure-delete files (path b == /dev/null) by preferring the b-side path
    and falling back to the a-side path.
    """
    if not patch_text.strip():
        return []

    # Split text into per-file sections using the diff header positions.
    headers = list(_DIFF_HEADER.finditer(patch_text))
    if not headers:
        return []

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(patch_text)
        a_path, b_path = m.group("a"), m.group("b")
        path = b_path if b_path != "/dev/null" else a_path
        sections.append((path, patch_text[m.end() : end]))

    out: list[FilePatch] = []
    for path, body in sections:
        hunks: list[Hunk] = []
        for hm in _HUNK_HEADER.finditer(body):
            hunks.append(
                Hunk(
                    old_start=int(hm.group("os")),
                    old_len=int(hm.group("ol") or 1),
                    new_start=int(hm.group("ns")),
                    new_len=int(hm.group("nl") or 1),
                )
            )
        out.append(FilePatch(path=path, hunks=hunks))
    return out


def load_task(task_dir: Path) -> TaskRecord:
    """Build a TaskRecord from a single harbor task directory."""
    config_path = task_dir / "tests" / "config.json"
    toml_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with toml_path.open("rb") as f:
        toml_data = tomllib.load(f)

    # instruction.md is the agent-facing prompt; fall back to the SWE-bench
    # problem_statement if the file is missing (schema varies across benchmarks).
    if instruction_path.exists():
        instruction = instruction_path.read_text(encoding="utf-8")
    else:
        instruction = config.get("problem_statement", "")
    gold_patch = config["patch"]
    test_patch = config.get("test_patch", "")

    # swe-bench-verified uses FAIL_TO_PASS/PASS_TO_PASS (often JSON strings);
    # swe-bench-pro uses lowercase fail_to_pass/pass_to_pass (native lists).
    fail_to_pass = _maybe_json_list(
        config.get("FAIL_TO_PASS", config.get("fail_to_pass", []))
    )
    pass_to_pass = _maybe_json_list(
        config.get("PASS_TO_PASS", config.get("pass_to_pass", []))
    )

    difficulty = (
        toml_data.get("metadata", {}).get("difficulty")
        or config.get("difficulty")
        or "unknown"
    )

    return TaskRecord(
        instance_id=config["instance_id"],
        repo=config["repo"],
        base_commit=config["base_commit"],
        environment_setup_commit=config.get("environment_setup_commit", config["base_commit"]),
        difficulty=difficulty,
        instruction_md=instruction,
        gold_patch=gold_patch,
        test_patch=test_patch,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        touched_files=parse_patch(gold_patch),
    )


def _maybe_json_list(val: object) -> list[str]:
    """SWE-bench sometimes encodes these as JSON strings, sometimes as lists."""
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            return [val]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return []


def iter_task_dirs(dataset_root: Path) -> Iterator[Path]:
    """Yield each task directory under the dataset root, sorted by instance id."""
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "tests" / "config.json").exists():
            yield child


def load_all(dataset_root: Path) -> Iterator[TaskRecord]:
    """Yield TaskRecord for every task under dataset_root."""
    for task_dir in iter_task_dirs(dataset_root):
        yield load_task(task_dir)
