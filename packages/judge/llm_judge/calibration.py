"""Join judge JSONL with per-instance ground truth and compute calibration metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DIFFICULTY_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]


def load_judge_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Drop duplicates from resumed runs (keep first).
    df = df.drop_duplicates(subset=["instance_id"], keep="first")
    return df


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = (~x.isna()) & (~y.isna())
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    if method == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    raise ValueError(method)


def calibrate(judge: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Join and compute metrics. Returns a structured dict."""
    df = judge.merge(truth, on="instance_id", how="inner")
    if df.empty:
        return {"error": "no overlap between judge JSONL and ground truth"}

    p = df["p_success"].astype(float)
    sr = df["success_rate"].astype(float)
    cs = df["complexity_score"].astype(float)
    steps = df["mean_steps"].astype(float)
    toks = df["mean_total_tokens"].astype(float)

    metrics = {
        "n_judged": int(len(judge)),
        "n_truth": int(len(truth)),
        "n_joined": int(len(df)),
        "spearman_psuccess_vs_successrate": _safe_corr(p, sr, "spearman"),
        "pearson_psuccess_vs_successrate": _safe_corr(p, sr, "pearson"),
        "spearman_complexity_vs_meansteps": _safe_corr(cs, steps, "spearman"),
        "spearman_complexity_vs_meantokens": _safe_corr(cs, toks, "spearman"),
        "mse": float(np.mean((p - sr) ** 2)),
        "mae": float(np.mean(np.abs(p - sr))),
        "log_loss_bernoulli": _bernoulli_log_loss(df),
        "mean_p_success_by_difficulty": _per_difficulty(df),
        "per_repo": _per_repo(df),
    }
    return metrics


def _bernoulli_log_loss(df: pd.DataFrame) -> float:
    """Treat each (instance, run) as a Bernoulli trial; uses n_resolved/n_runs."""
    eps = 1e-6
    p = df["p_success"].clip(eps, 1 - eps).to_numpy()
    n_runs = df["n_runs"].to_numpy()
    n_res = df["n_resolved"].to_numpy()
    n_fail = n_runs - n_res
    total = n_runs.sum()
    if total == 0:
        return float("nan")
    ll = (n_res * np.log(p) + n_fail * np.log(1 - p)).sum() / total
    return float(-ll)


def _per_difficulty(df: pd.DataFrame) -> list[dict]:
    out = []
    for d in DIFFICULTY_ORDER:
        sub = df[df["difficulty"] == d]
        if sub.empty:
            continue
        out.append(
            {
                "difficulty": d,
                "n": int(len(sub)),
                "mean_p_success": float(sub["p_success"].mean()),
                "mean_actual_success_rate": float(sub["success_rate"].mean()),
                "mean_complexity_score": float(sub["complexity_score"].mean()),
            }
        )
    return out


def _per_repo(df: pd.DataFrame) -> list[dict]:
    out = []
    for repo, sub in df.groupby("repo"):
        out.append(
            {
                "repo": repo,
                "n": int(len(sub)),
                "mean_p_success": float(sub["p_success"].mean()),
                "mean_actual_success_rate": float(sub["success_rate"].mean()),
                "spearman": _safe_corr(sub["p_success"], sub["success_rate"], "spearman"),
            }
        )
    return sorted(out, key=lambda r: -r["n"])


def render_markdown(metrics: dict, model: str | None = None) -> str:
    if "error" in metrics:
        return f"# Calibration report\n\n**Error:** {metrics['error']}\n"

    lines = ["# Calibration report"]
    if model:
        lines.append(f"\n_Model:_ `{model}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Judged tasks: **{metrics['n_judged']}**")
    lines.append(f"- Ground-truth tasks: **{metrics['n_truth']}**")
    lines.append(f"- Joined: **{metrics['n_joined']}**")
    lines.append("")
    lines.append("## Correlations")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Spearman ρ (p_success vs success_rate) | {metrics['spearman_psuccess_vs_successrate']:.3f} |")
    lines.append(f"| Pearson r (p_success vs success_rate)  | {metrics['pearson_psuccess_vs_successrate']:.3f} |")
    lines.append(f"| Spearman ρ (complexity_score vs mean_steps)  | {metrics['spearman_complexity_vs_meansteps']:.3f} |")
    lines.append(f"| Spearman ρ (complexity_score vs mean_tokens) | {metrics['spearman_complexity_vs_meantokens']:.3f} |")
    lines.append(f"| Task-level MSE (lower better) | {metrics['mse']:.4f} |")
    lines.append(f"| Task-level MAE (lower better) | {metrics['mae']:.4f} |")
    lines.append(f"| Bernoulli log-loss (lower better) | {metrics['log_loss_bernoulli']:.4f} |")
    lines.append("")
    lines.append("## Mean p_success by built-in difficulty")
    lines.append("| Difficulty | n | mean p_success | mean actual rate | mean complexity |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in metrics["mean_p_success_by_difficulty"]:
        lines.append(
            f"| {row['difficulty']} | {row['n']} | {row['mean_p_success']:.3f}"
            f" | {row['mean_actual_success_rate']:.3f} | {row['mean_complexity_score']:.2f} |"
        )
    lines.append("")
    lines.append("## Per-repo breakdown")
    lines.append("| Repo | n | mean p_success | mean actual rate | Spearman ρ |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in metrics["per_repo"]:
        rho = "n/a" if np.isnan(row["spearman"]) else f"{row['spearman']:.3f}"
        lines.append(
            f"| {row['repo']} | {row['n']} | {row['mean_p_success']:.3f}"
            f" | {row['mean_actual_success_rate']:.3f} | {rho} |"
        )
    return "\n".join(lines) + "\n"
