"""Aggregate local agent-run CSVs into per-instance ground-truth stats."""
from __future__ import annotations

from pathlib import Path

import hashlib
import json

import pandas as pd


def truth_cache_path(results_dir: Path, cache_root: Path = Path("data"), policy: str = "boolean_only") -> Path:
    """Per-benchmark cache path so different result sets never collide."""
    return cache_root / f"instance_stats__{results_dir.name}__{policy}.json"


EXPECTED_COLS = {
    "RunId",
    "BenchmarkName",
    "InstanceId",
    "Resolved",
    "TotalSteps",
    "TotalInputTokens",
    "TotalOutputTokens",
    "TotalCachedInputTokens",
    "TotalTokens",
}


def load_runs(results_dir: Path, policy: str = "boolean_only") -> pd.DataFrame:
    """Keep known binary outcomes; all_reports is for historical exemplar labels."""
    if policy not in {"boolean_only", "all_reports"}:
        raise ValueError(f"Unknown outcome policy: {policy}")
    files = sorted(results_dir.glob("report-*.csv"))
    if not files:
        raise FileNotFoundError(f"No report-*.csv under {results_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = EXPECTED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} missing columns: {missing}")
        status = df["Resolved"].astype(str).str.strip().str.lower()
        if not status.isin(["true", "false", "error", "missing"]).all():
            raise ValueError(f"Unknown outcome status in {f.name}")
        if df["InstanceId"].duplicated().any():
            raise ValueError(f"Duplicate task outcome in {f.name}")
        if policy == "boolean_only":
            df = df.loc[status.isin(["true", "false"])].copy()
        df["Resolved"] = status.loc[df.index].eq("true")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-InstanceId aggregation, weighting every run equally."""
    resolved_only = runs[runs["Resolved"]].groupby("InstanceId")
    full = runs.groupby("InstanceId")

    agg = pd.DataFrame(
        {
            "n_runs": full["Resolved"].count(),
            "n_resolved": full["Resolved"].sum(),
            "mean_steps": full["TotalSteps"].mean(),
            "median_steps": full["TotalSteps"].median(),
            "mean_total_tokens": full["TotalTokens"].mean(),
            "median_total_tokens": full["TotalTokens"].median(),
            "mean_steps_resolved": resolved_only["TotalSteps"].mean(),
            "mean_total_tokens_resolved": resolved_only["TotalTokens"].mean(),
        }
    )
    agg["success_rate"] = agg["n_resolved"] / agg["n_runs"]
    agg = agg.reset_index().rename(columns={"InstanceId": "instance_id"})
    return agg


def build_or_load(results_dir: Path, cache_path: Path | None = None,
                  policy: str = "boolean_only") -> pd.DataFrame:
    """Cache identity includes policy and input bytes, avoiding legacy truth caches."""
    files = sorted(results_dir.glob("report-*.csv"))
    fingerprint = hashlib.sha256(policy.encode() + b"".join(
        f.name.encode() + hashlib.sha256(f.read_bytes()).digest() for f in files)).hexdigest()
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint:
            return pd.DataFrame(cached["rows"])
    agg = aggregate(load_runs(results_dir, policy))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dict(policy=policy, fingerprint=fingerprint,
            rows=json.loads(agg.to_json(orient="records"))), indent=2) + "\n")
    return agg
