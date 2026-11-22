"""LLM judge invocation via LiteLLM, with disk cache and retry."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from pydantic import ValidationError

from llm_judge.prompt import SYSTEM_PROMPT, build_user_prompt
from llm_judge.repo_context import DEFAULT_CACHE, build_context
from llm_judge.schema import JudgeOutput, RepoContext, TaskRecord

DEFAULT_MODEL = os.environ.get("LLM_JUDGE_MODEL", "github_copilot/claude-opus-4.7")
CACHE_DIR = DEFAULT_CACHE / "judge"


def _cache_key(instance_id: str, model: str, prompt: str) -> Path:
    # Hash both system + user prompt so any prompt revision invalidates the cache.
    payload = SYSTEM_PROMPT + "\n\n" + prompt
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    safe_model = model.replace("/", "_").replace(":", "_")
    return CACHE_DIR / safe_model / f"{instance_id}__{h}.json"


def _iter_balanced_objects(s: str):
    """Yield every top-level balanced ``{...}`` substring in ``s``."""
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        end_idx = -1
        for j in range(i, n):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = j
                        break
        if end_idx == -1:
            return  # unbalanced; nothing more to find
        yield s[i : end_idx + 1]
        i = end_idx + 1


def _strip_json_fence(text: str) -> str:
    """Extract the judge's JSON object from a model response.

    The model sometimes wraps the answer in a preamble/postamble or in a
    ```json fence``` and may quote OTHER ``{...}`` blocks (schema snippets,
    diff fragments). Strategy:

      1. Prefer a ```json``` fenced block if present.
      2. Otherwise, scan all top-level balanced ``{...}`` substrings and return
         the LAST one whose top-level keys include ``p_success`` — the answer.
      3. Fallback: return the last balanced block, or the stripped text.
    """
    s = text.strip()

    # 1. Fenced JSON.
    if "```" in s:
        # Find a ```json ... ``` or ``` ... ``` block.
        for marker in ("```json", "```JSON", "```"):
            start = s.find(marker)
            if start == -1:
                continue
            after = start + len(marker)
            end = s.find("```", after)
            if end != -1:
                inner = s[after:end].strip()
                # The inner may itself need balanced-object extraction.
                for obj in _iter_balanced_objects(inner):
                    if '"p_success"' in obj:
                        return obj.strip()
                return inner

    # 2. Pick the last top-level object that looks like the answer.
    best: str | None = None
    last: str | None = None
    for obj in _iter_balanced_objects(s):
        last = obj
        if '"p_success"' in obj:
            best = obj
    if best is not None:
        return best.strip()
    if last is not None:
        return last.strip()

    # 3. Nothing balanced — return the stripped text and let pydantic complain.
    return s


def _call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_retries: int = 5,
    timeout: int = 180,
) -> tuple[str, dict]:
    """Call LiteLLM with retry; returns (content, usage).

    Some endpoints (notably the GitHub Copilot proxy for newer Claude models)
    silently under-weight the system role, causing the model to treat the
    judging prompt as an agentic task and emit a code patch. To make behavior
    robust across providers, we fold the system instructions into the user
    message and send a minimal nudge as the system role.
    """
    import random

    import litellm  # imported lazily so unit tests don't need network

    if system_prompt:
        merged_user = (
            "Follow the JUDGE INSTRUCTIONS below exactly. Do not solve the task; "
            "rate it.\n\n"
            "===== JUDGE INSTRUCTIONS =====\n"
            f"{system_prompt}\n"
            "===== END JUDGE INSTRUCTIONS =====\n\n"
            f"{user_prompt}"
        )
        sys_msg = "You output ONLY the single JSON object requested below."
    else:
        merged_user = user_prompt
        sys_msg = ""

    messages: list[dict] = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": merged_user})

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=8192,
                timeout=timeout,
            )
            content = resp["choices"][0]["message"]["content"]
            usage = _normalize_usage(resp.get("usage"))
            return content, usage
        except Exception as e:  # noqa: BLE001 — litellm raises a wide variety
            last_err = e
            # Exponential backoff with jitter: 1, 2, 4, 8, 16 s (+ up to 1s jitter).
            time.sleep(min(2**attempt, 16) + random.random())
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def score_task(
    record: TaskRecord,
    *,
    model: str = DEFAULT_MODEL,
    ctx: RepoContext | None = None,
    use_cache: bool = True,
    skip_repo: bool = False,
    show_tests: bool = False,
    fewshot_block: str | None = None,
) -> dict:
    """Score a single task. Returns a dict ready to JSONL-dump.

    The returned dict is the union of judge output + bookkeeping fields
    (`instance_id`, `repo`, `difficulty`, `model`, `usage`, `cached`,
    `context_unavailable`).
    """
    if ctx is None and not skip_repo:
        ctx = build_context(record.repo, record.base_commit, record.touched_files)

    user_prompt = build_user_prompt(
        record, ctx, show_tests=show_tests, fewshot_block=fewshot_block
    )

    cache_path = _cache_key(record.instance_id, model, user_prompt)
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except (OSError, json.JSONDecodeError):
            pass

    content, usage = _call_llm(model, SYSTEM_PROMPT, user_prompt)
    raw = _strip_json_fence(content)
    try:
        parsed = JudgeOutput.model_validate_json(raw)
    except ValidationError as e:
        # One retry with a corrective system note.
        corrective = (
            SYSTEM_PROMPT
            + "\n\nThe previous response was invalid JSON or violated the schema. "
            "Output ONLY the JSON object specified above, with all required fields."
        )
        content, usage2 = _call_llm(model, corrective, user_prompt)
        raw = _strip_json_fence(content)
        parsed = JudgeOutput.model_validate_json(raw)  # let it raise if still bad
        usage = _merge_usage(usage, usage2)
        _ = e  # silence linter

    record_out = {
        "instance_id": record.instance_id,
        "repo": record.repo,
        "difficulty": record.difficulty,
        "model": model,
        "p_success": parsed.p_success,
        "complexity_score": parsed.complexity_score,
        "subfactors": parsed.subfactors.model_dump(),
        "rationale": parsed.rationale,
        "reasoning_notes": parsed.reasoning_notes.model_dump() if parsed.reasoning_notes else None,
        "usage": usage,
        "context_unavailable": ctx is None or bool(ctx and ctx.error),
        "cached": False,
    }

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(record_out, indent=2), encoding="utf-8")
    return record_out


def _merge_usage(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, (int, float)) and isinstance(out.get(k), (int, float)):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _normalize_usage(raw: object) -> dict:
    """Coerce LiteLLM/pydantic usage objects into a plain JSON-serializable dict."""
    if raw is None:
        return {}
    # Pydantic-like .model_dump()
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        try:
            return json.loads(json.dumps(dump(), default=_jsonable))
        except Exception:  # noqa: BLE001
            pass
    if isinstance(raw, dict):
        try:
            return json.loads(json.dumps(raw, default=_jsonable))
        except Exception:  # noqa: BLE001
            return {k: _jsonable(v) for k, v in raw.items()}
    return {"raw": str(raw)}


def _jsonable(v: object) -> object:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        return dump()
    if hasattr(v, "__dict__"):
        return {k: _jsonable(x) for k, x in vars(v).items() if not k.startswith("_")}
    return str(v)
