"""Typer-based CLI: `uv run judge {run,calibrate,summary}`."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from llm_judge.calibration import calibrate, load_judge_jsonl, render_markdown
from llm_judge.ground_truth import build_or_load, truth_cache_path
from llm_judge.judge import DEFAULT_MODEL
from llm_judge.runner import run_batch

app = typer.Typer(help="LLM judge for SWE-bench Verified task complexity.")

DEFAULT_DATASET = Path("dataset/swe-bench-verified")
DEFAULT_RESULTS_DIR = Path("results/swe-bench-verified")
DEFAULT_OUT_DIR = Path("results/judge")


@app.command("run")
def cmd_run(
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Root of harbor tasks."),
    model: str = typer.Option(DEFAULT_MODEL, help="LiteLLM model id."),
    out: Path | None = typer.Option(None, help="Output JSONL (default: results/judge/<model>__<ts>.jsonl)."),
    workers: int = typer.Option(8, help="Parallel workers."),
    limit: int | None = typer.Option(None, help="Score at most this many tasks."),
    filter: str | None = typer.Option(None, "--filter", help="Regex on instance_id."),
    results_dir: Path = typer.Option(
        DEFAULT_RESULTS_DIR,
        help="Dir of report-*.csv (only used for --few-shot exemplar labels).",
    ),
    skip_repo: bool = typer.Option(False, help="Skip cloning the repo (faster, less context)."),
    show_tests: bool = typer.Option(
        True,
        "--show-tests/--no-show-tests",
        help="Show the eval tests (FAIL/PASS names + test_patch) to the judge.",
    ),
    few_shot: int = typer.Option(
        0,
        "--few-shot",
        help="Use N labeled exemplars (held out of eval) to calibrate the judge.",
    ),
    few_shot_seed: int = typer.Option(0, help="Seed for exemplar selection."),
    no_cache: bool = typer.Option(False, help="Disable per-task disk cache."),
) -> None:
    """Score tasks and write a JSONL of judge outputs."""
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = model.replace("/", "_").replace(":", "_")
        out = DEFAULT_OUT_DIR / f"{safe}__{ts}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writing to {out}")
    summary = run_batch(
        dataset_root=dataset,
        output_path=out,
        model=model,
        workers=workers,
        limit=limit,
        pattern=filter,
        skip_repo=skip_repo,
        use_cache=not no_cache,
        show_tests=show_tests,
        few_shot=few_shot,
        few_shot_seed=few_shot_seed,
        results_dir=results_dir,
    )
    typer.echo(json.dumps(summary, indent=2))


@app.command("calibrate")
def cmd_calibrate(
    judge_jsonl: Path = typer.Argument(..., help="Path to judge JSONL output."),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, help="Dir of report-*.csv."),
    truth_cache: Path | None = typer.Option(
        None, help="Cache file for aggregated truth (default: per-benchmark under data/)."
    ),
    report: Path | None = typer.Option(None, help="Markdown report path (default beside JSONL)."),
    rebuild_truth: bool = typer.Option(False, help="Force rebuild of truth cache."),
) -> None:
    """Compute calibration metrics against local agent-run CSVs."""
    if truth_cache is None:
        truth_cache = truth_cache_path(results_dir)
    if rebuild_truth and truth_cache.exists():
        truth_cache.unlink()
    truth = build_or_load(results_dir, truth_cache)
    typer.echo(f"Loaded {len(truth)} ground-truth instances from {results_dir}")
    judge = load_judge_jsonl(judge_jsonl)
    typer.echo(f"Loaded {len(judge)} judge records from {judge_jsonl}")
    if judge.empty:
        typer.echo("Empty judge JSONL; nothing to calibrate.")
        raise typer.Exit(code=1)
    metrics = calibrate(judge, truth)
    model = judge["model"].iloc[0] if "model" in judge.columns and not judge.empty else None
    md = render_markdown(metrics, model=model)
    if report is None:
        report = judge_jsonl.with_suffix(".calibration.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(md, encoding="utf-8")
    typer.echo(md)
    typer.echo(f"\nWrote report to {report}")


@app.command("summary")
def cmd_summary(
    dataset: Path = typer.Option(DEFAULT_DATASET, help="Root of harbor tasks."),
) -> None:
    """Print basic dataset stats (count by difficulty and repo)."""
    from collections import Counter

    from llm_judge.loader import iter_task_dirs, load_task

    diffs: Counter[str] = Counter()
    repos: Counter[str] = Counter()
    total = 0
    for d in iter_task_dirs(dataset):
        try:
            rec = load_task(d)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[skip] {d.name}: {e}")
            continue
        diffs[str(rec.difficulty)] += 1
        repos[rec.repo] += 1
        total += 1
    typer.echo(f"Total tasks: {total}")
    typer.echo("\nBy difficulty:")
    for d, n in diffs.most_common():
        typer.echo(f"  {d}: {n}")
    typer.echo("\nBy repo:")
    for r, n in repos.most_common():
        typer.echo(f"  {r}: {n}")


if __name__ == "__main__":
    app()
