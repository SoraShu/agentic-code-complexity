"""Fetch repo file context at the right commit for each task."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from llm_judge.schema import FilePatch, FileSnippet, RepoContext

DEFAULT_CACHE = Path(
    os.environ.get("LLM_JUDGE_CACHE", str(Path.home() / ".cache" / "llm-judge"))
)
WINDOW_LINES = 60
FULL_FILE_THRESHOLD = 300
REPO_TREE_MAX_ENTRIES = 120


def _repo_dir(cache: Path, repo: str, commit: str) -> Path:
    return cache / "repos" / f"{repo.replace('/', '__')}@{commit[:12]}"


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ensure_repo(repo: str, commit: str, cache: Path = DEFAULT_CACHE) -> Path | None:
    """Clone (if needed) and checkout the given commit. Returns the repo path, or None on failure."""
    target = _repo_dir(cache, repo, commit)
    if (target / ".git").exists():
        # Already cloned; ensure desired commit is checked out.
        co = _run(["git", "checkout", "--quiet", commit], cwd=target)
        if co.returncode == 0:
            return target
        # Try to fetch the specific commit then checkout.
        _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=target)
        co = _run(["git", "checkout", "--quiet", commit], cwd=target)
        return target if co.returncode == 0 else None

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    # Try shallow clone of the specific commit first; fall back to full clone.
    target.mkdir(parents=True, exist_ok=True)
    init = _run(["git", "init", "--quiet"], cwd=target)
    if init.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        return None
    _run(["git", "remote", "add", "origin", url], cwd=target)
    fetch = _run(["git", "fetch", "--depth", "1", "origin", commit], cwd=target, timeout=900)
    if fetch.returncode != 0:
        # Fall back to a full fetch of the default branch and try to checkout.
        shutil.rmtree(target, ignore_errors=True)
        clone = _run(["git", "clone", "--quiet", url, str(target)], timeout=1800)
        if clone.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            return None
    co = _run(["git", "checkout", "--quiet", commit], cwd=target)
    if co.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        return None
    return target


def extract_file_window(repo_path: Path, file_patch: FilePatch, window: int = WINDOW_LINES) -> FileSnippet:
    """Read the file at the commit and return either the full file (small) or hunk windows."""
    full_path = repo_path / file_patch.path
    if not full_path.exists():
        return FileSnippet(path=file_patch.path, content="", note="file not found at this commit")

    try:
        lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return FileSnippet(path=file_patch.path, content="", note=f"read error: {e}")

    n = len(lines)
    if n <= FULL_FILE_THRESHOLD:
        return FileSnippet(
            path=file_patch.path,
            content=_with_line_numbers(lines, 1),
            truncated=False,
        )

    if not file_patch.hunks:
        # Whole-file replacement or rename without hunks; just take the head.
        head = lines[:FULL_FILE_THRESHOLD]
        return FileSnippet(
            path=file_patch.path,
            content=_with_line_numbers(head, 1),
            truncated=True,
            note=f"head of {n}-line file",
        )

    # Merge overlapping windows around each hunk (old-side coordinates).
    ranges: list[tuple[int, int]] = []
    for h in file_patch.hunks:
        start = max(1, h.old_start - window)
        end = min(n, h.old_start + h.old_len + window)
        ranges.append((start, end))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    for start, end in merged:
        chunk = lines[start - 1 : end]
        parts.append(f"# --- lines {start}-{end} of {n} ---")
        parts.append(_with_line_numbers(chunk, start))
    return FileSnippet(
        path=file_patch.path,
        content="\n".join(parts),
        truncated=True,
        note=f"windowed view ({len(merged)} regions, total file is {n} lines)",
    )


def _with_line_numbers(lines: list[str], start_lineno: int) -> str:
    width = max(4, len(str(start_lineno + len(lines))))
    return "\n".join(f"{i + start_lineno:>{width}}  {line}" for i, line in enumerate(lines))


def build_repo_tree(repo_path: Path, max_entries: int = REPO_TREE_MAX_ENTRIES) -> str:
    """List the top-2 levels of the repo, capped to max_entries."""
    entries: list[str] = []
    try:
        for top in sorted(repo_path.iterdir()):
            if top.name.startswith("."):
                continue
            if top.is_dir():
                entries.append(f"{top.name}/")
                if len(entries) >= max_entries:
                    break
                children: list[str] = []
                try:
                    for sub in sorted(top.iterdir()):
                        if sub.name.startswith("."):
                            continue
                        suffix = "/" if sub.is_dir() else ""
                        children.append(f"  {sub.name}{suffix}")
                        if len(children) >= 30:
                            children.append("  ...")
                            break
                except OSError:
                    pass
                entries.extend(children)
                if len(entries) >= max_entries:
                    entries.append("...")
                    break
            else:
                entries.append(top.name)
    except OSError as e:
        return f"(repo tree unavailable: {e})"
    return "\n".join(entries[:max_entries])


def build_context(
    repo: str,
    commit: str,
    touched_files: list[FilePatch],
    cache: Path = DEFAULT_CACHE,
) -> RepoContext:
    """Top-level: ensure the repo is cloned, then build per-file snippets + repo tree."""
    repo_path = ensure_repo(repo, commit, cache=cache)
    if repo_path is None:
        return RepoContext(repo=repo, commit=commit, error="clone/checkout failed")

    snippets = [extract_file_window(repo_path, fp) for fp in touched_files]
    tree = build_repo_tree(repo_path)
    return RepoContext(repo=repo, commit=commit, file_snippets=snippets, repo_tree=tree)
