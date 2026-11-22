"""Few-shot exemplar selection and prompt-block construction.

Exemplars are real tasks paired with their OBSERVED success rate (from the local
agent-run CSVs). They teach the judge to calibrate. To avoid leakage, exemplars
are excluded from the evaluation set by the runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llm_judge.prompt import _truncate
from llm_judge.schema import TaskRecord

# Compact budgets for exemplars (smaller than the task-under-eval).
_EX_INSTRUCTION_CHARS = 500
_EX_PATCH_CHARS = 1_200


def select_exemplars(truth: pd.DataFrame, n: int, seed: int = 0) -> list[str]:
    """Pick n instance_ids stratified across the observed success-rate range.

    Sorting by success_rate then taking evenly-spaced indices guarantees the
    judge sees the full difficulty spectrum (easy → hard), which is what makes
    the calibration useful.
    """
    df = truth.dropna(subset=["success_rate"]).sort_values("success_rate").reset_index(drop=True)
    if n >= len(df):
        return df["instance_id"].tolist()
    # Evenly-spaced positions across the sorted range.
    idx = np.linspace(0, len(df) - 1, num=n).round().astype(int)
    idx = sorted(set(idx.tolist()))
    # If rounding collapsed duplicates, top up deterministically.
    i = 0
    while len(idx) < n and i < len(df):
        if i not in idx:
            idx.append(i)
        i += 1
    idx = sorted(idx[:n])
    return df.loc[idx, "instance_id"].tolist()


def build_fewshot_block(
    exemplars: list[TaskRecord], success_rates: dict[str, float]
) -> str:
    """Render the labeled exemplars into a calibration block for the prompt."""
    parts: list[str] = []
    parts.append("## Calibration examples (real tasks with OBSERVED agent success rates)")
    parts.append(
        "Below are example tasks. Each shows the ACTUAL fraction of independent "
        "agent attempts that solved it (observed_success_rate). Study how task "
        "characteristics map to real outcomes, then apply the same calibration "
        "when you score the target task. Do NOT just copy a number — reason by "
        "analogy from the closest examples."
    )
    parts.append("")
    for i, rec in enumerate(exemplars, 1):
        sr = success_rates.get(rec.instance_id)
        if sr is None:
            continue
        parts.append(f"### Example {i} — observed_success_rate = {sr:.2f}")
        parts.append(f"Repo: {rec.repo}")
        parts.append(
            f"Tests: {len(rec.fail_to_pass)} FAIL_TO_PASS, {len(rec.pass_to_pass)} PASS_TO_PASS"
        )
        parts.append("Instruction:")
        parts.append(_truncate(rec.instruction_md.strip(), _EX_INSTRUCTION_CHARS, "instruction"))
        files = ", ".join(fp.path for fp in rec.touched_files) or "(unparsed)"
        parts.append(f"Touched files: {files}")
        parts.append("Gold patch (truncated):")
        parts.append("```diff")
        parts.append(_truncate(rec.gold_patch, _EX_PATCH_CHARS, "patch"))
        parts.append("```")
        parts.append("")
    parts.append("--- End of calibration examples. Now score the TARGET task below. ---")
    parts.append("")
    return "\n".join(parts)
