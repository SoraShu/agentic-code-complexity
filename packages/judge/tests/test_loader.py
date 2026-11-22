"""Unit tests for the dataset loader and patch parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.loader import iter_task_dirs, load_task, parse_patch

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "swe-bench-verified"


def test_parse_patch_simple() -> None:
    patch = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -10,3 +10,4 @@ def hello():
     return 1
+    # extra
@@ -100,2 +101,2 @@ def goodbye():
-old
+new
"""
    files = parse_patch(patch)
    assert len(files) == 1
    f = files[0]
    assert f.path == "foo.py"
    assert len(f.hunks) == 2
    assert f.hunks[0].old_start == 10 and f.hunks[0].old_len == 3
    assert f.hunks[1].new_start == 101


def test_parse_patch_multifile() -> None:
    patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x
+y
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -5,2 +5,3 @@
 y
+z
"""
    files = parse_patch(patch)
    assert [f.path for f in files] == ["a.py", "b.py"]
    assert files[0].hunks[0].old_len == 1  # default-1 elision
    assert files[1].hunks[0].new_len == 3


def test_parse_patch_empty() -> None:
    assert parse_patch("") == []
    assert parse_patch("not a diff") == []


@pytest.mark.parametrize(
    "instance_id, expected_files, expected_min_hunks",
    [
        ("astropy__astropy-7166", ["astropy/utils/misc.py"], 1),
    ],
)
def test_load_sample_task(instance_id: str, expected_files: list[str], expected_min_hunks: int) -> None:
    task_dir = DATASET / instance_id
    if not task_dir.exists():
        pytest.skip(f"sample task {instance_id} not present")
    rec = load_task(task_dir)
    assert rec.instance_id == instance_id
    assert rec.gold_patch.strip().startswith("diff --git")
    assert len(rec.touched_files) >= 1
    paths = [f.path for f in rec.touched_files]
    for ef in expected_files:
        assert ef in paths
    total_hunks = sum(len(f.hunks) for f in rec.touched_files)
    assert total_hunks >= expected_min_hunks
    assert rec.difficulty in {"<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours", "unknown"}


def test_iter_task_dirs_finds_many() -> None:
    if not DATASET.exists():
        pytest.skip("dataset not present")
    count = sum(1 for _ in iter_task_dirs(DATASET))
    assert count > 100  # we expect ~504
