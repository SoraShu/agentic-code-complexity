"""Batch runner: score every (or a filtered subset of) tasks in parallel."""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from llm_judge.judge import DEFAULT_MODEL, score_task
from llm_judge.loader import load_all
from llm_judge.schema import TaskRecord


def _filter(records: Iterable[TaskRecord], pattern: str | None, limit: int | None) -> list[TaskRecord]:
    out: list[TaskRecord] = []
    rx = re.compile(pattern) if pattern else None
    for r in records:
        if rx and not rx.search(r.instance_id):
            continue
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    return out


def run_batch(
    dataset_root: Path,
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    workers: int = 8,
    limit: int | None = None,
    pattern: str | None = None,
    skip_repo: bool = False,
    use_cache: bool = True,
    show_tests: bool = False,
    few_shot: int = 0,
    few_shot_seed: int = 0,
    results_dir: Path | None = None,
) -> dict:
    """Score tasks and append JSONL records to `output_path`. Returns a small summary dict.

    When ``few_shot > 0``, that many tasks are selected as labeled calibration
    exemplars (stratified across observed success rate), held OUT of the
    evaluation set, and injected into every prompt.
    """
    all_records = list(load_all(dataset_root))

    fewshot_block: str | None = None
    exemplar_ids: set[str] = set()
    if few_shot > 0:
        from llm_judge.fewshot import build_fewshot_block, select_exemplars
        from llm_judge.ground_truth import build_or_load, truth_cache_path

        rdir = results_dir or Path("results/swe-bench-verified")
        # Historical exemplars used all reports in the denominator. This does not
        # change the Boolean-only policy used to evaluate saved predictions.
        truth = build_or_load(rdir, truth_cache_path(rdir, policy="all_reports"), policy="all_reports")
        sr_map = dict(zip(truth["instance_id"], truth["success_rate"]))
        chosen = select_exemplars(truth, few_shot, seed=few_shot_seed)
        exemplar_ids = set(chosen)
        by_id = {r.instance_id: r for r in all_records}
        # Preserve the chosen order (already easy→hard).
        exemplars = [by_id[i] for i in chosen if i in by_id]
        fewshot_block = build_fewshot_block(exemplars, sr_map)

    # Exemplars are held out of evaluation to avoid leakage.
    eval_records = [r for r in all_records if r.instance_id not in exemplar_ids]
    records = _filter(eval_records, pattern, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track which instance_ids are already in the output file (resume safety).
    done: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    pending = [r for r in records if r.instance_id not in done]
    lock = threading.Lock()
    summary = {
        "total": len(records),
        "skipped": len(records) - len(pending),
        "ok": 0,
        "failed": 0,
        "few_shot": few_shot,
        "exemplars_held_out": len(exemplar_ids),
    }

    def _worker(rec: TaskRecord) -> tuple[str, dict | None, str | None]:
        try:
            result = score_task(
                rec,
                model=model,
                use_cache=use_cache,
                skip_repo=skip_repo,
                show_tests=show_tests,
                fewshot_block=fewshot_block,
            )
            return rec.instance_id, result, None
        except Exception as e:  # noqa: BLE001
            return rec.instance_id, None, f"{type(e).__name__}: {e}"

    with output_path.open("a", encoding="utf-8") as out_f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, r) for r in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            instance_id, result, err = fut.result()
            if err is not None:
                summary["failed"] += 1
                tqdm.write(f"[fail] {instance_id}: {err}")
                continue
            with lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            summary["ok"] += 1

    return summary
