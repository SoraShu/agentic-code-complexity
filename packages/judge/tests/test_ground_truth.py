"""Unit tests for ground-truth aggregation."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.ground_truth import aggregate, load_runs

RESULTS = Path(__file__).resolve().parents[1] / "results" / "swe-bench-verified"


def test_load_and_aggregate() -> None:
    if not RESULTS.exists():
        pytest.skip("results dir not present")
    runs = load_runs(RESULTS)
    assert len(runs) > 0
    agg = aggregate(runs)
    assert "instance_id" in agg.columns
    assert "success_rate" in agg.columns
    assert (agg["success_rate"] >= 0).all() and (agg["success_rate"] <= 1).all()
    assert (agg["n_runs"] >= 1).all()
    # Sanity: at least some instances were attempted by multiple runs.
    assert agg["n_runs"].max() >= 2
