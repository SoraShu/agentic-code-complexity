"""Reproducible nested CV, using immutable local snapshots of thesis inputs.

Run from this directory with: uv run python evaluate.py --workers 8
No network access, agent calls, or writes to the historical experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Avoid multiplying BLAS/OpenMP threads by the number of search workers.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from joblib import parallel_config
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "lines_changed", "additions", "deletions", "files_changed",
    "top_level_dirs", "second_level_dirs", "cc_total", "cc_mean", "cc_max",
    "cog_total", "cog_mean", "cog_max", "mi_total", "mi_mean", "mi_min",
    "hv_total", "hv_mean", "loc_total", "loc_mean",
]
XGB_GRID = {
    "n_estimators": [50, 100, 200], "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [3, 5, 10],
}
# Fixed before evaluating the outer folds. This replaces the two inconsistent
# historical alpha grids. Selection loss remains MSE, as in original LassoCV.
ALPHAS = np.logspace(-4, 0, 100).tolist()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def feature_vector(record):
    values = {k: record.get(k, 0) for k in FEATURES[:4]}
    values.update({k: record.get("dispersity", {}).get(k, 0) for k in FEATURES[4:6]})
    for short, long, aggregations in [
        ("cc", "cyclomatic", ("total", "mean", "max")),
        ("cog", "cognitive", ("total", "mean", "max")),
        ("mi", "maintainability_index", ("total", "mean", "min")),
        ("hv", "halstead_volume", ("total", "mean")),
        ("loc", "loc", ("total", "mean")),
    ]:
        for aggregation in aggregations:
            values[f"{short}_{aggregation}"] = record.get("complexity", {}).get(long, {}).get(aggregation, 0)
    return [values[k] for k in FEATURES]


def load_dataset(name, arrow_policy="historical", feature_root=None):
    """Preserve historical ordering/features; make outcome filtering explicit."""
    folder = ROOT / "inputs" / name
    metric_folder = (Path(feature_root) / name) if feature_root else folder
    total, success, errors = Counter(), Counter(), Counter()
    oracle_pass, oracle_fail = set(), set()
    if name == "verified":
        metric_file = metric_folder / "task_metrics.jsonl"
        metric_rows = [json.loads(line) for line in metric_file.read_text().splitlines()]
        metrics = {r["task_name"]: r for r in metric_rows}
        for file in sorted((folder / "results").glob("*.csv")):
            seen = set()
            for row in csv.DictReader(io.StringIO(file.read_text())):
                task = row["InstanceId"]
                assert task not in seen, (file, task)
                seen.add(task)
                total[task] += 1
                success[task] += row["Resolved"].strip().lower() == "true"
        task_ids = sorted(total)
    else:
        metric_file = metric_folder / "pr_metrics.jsonl"
        metric_rows = [json.loads(line) for line in metric_file.read_text().splitlines()]
        metrics = {f"pr-{r['pr_number']}": r for r in metric_rows}
        for file in sorted(folder.glob("run_*.json")):
            seen = set()
            for evaluation in json.loads(file.read_text())["stats"]["evals"].values():
                for reward, tasks in evaluation["reward_stats"]["reward"].items():
                    assert reward in ("0.0", "1.0")
                    for trial in tasks:
                        task = trial.split("__")[0]
                        assert task not in seen, (file, task)
                        seen.add(task)
                        total[task] += 1
                        success[task] += reward == "1.0"
                for trials in evaluation.get("exception_stats", {}).values():
                    for trial in trials:
                        task = trial.split("__")[0]
                        assert task not in seen, (file, task)
                        seen.add(task)
                        errors[task] += 1
        for evaluation in json.loads((folder / "oracle.json").read_text())["stats"]["evals"].values():
            for reward, trials in evaluation["reward_stats"]["reward"].items():
                target = oracle_pass if reward == "1.0" else oracle_fail
                target.update(trial.split("__")[0] for trial in trials)
        assert not oracle_pass & oracle_fail
        task_ids = sorted(total, key=lambda task: int(task.split("-")[1]))
    assert len(metrics) == len(metric_rows), "Duplicate metric rows"
    excluded, rows = [], []
    for task in task_ids:
        assert task in metrics, f"Missing metrics: {task}"
        if name == "arrow":
            assert task in oracle_pass | oracle_fail
            exclude = (task not in oracle_pass) if arrow_policy == "oracle_pass" else (task in oracle_fail and success[task] == 0)
            if exclude:
                excluded.append(task)
                continue
        denominator = total[task] + (errors[task] if arrow_policy == "errors_as_failures" else 0)
        rows.append({
            "task_id": task, "n_resolved": success[task], "n_valid": total[task],
            "n_errors": errors[task], "denominator": denominator,
            "oracle_pass": task in oracle_pass if name == "arrow" else None,
            "complexity": 1 - success[task] / denominator,
            "features": feature_vector(metrics[task]),
        })
    X = np.asarray([r["features"] for r in rows], dtype=float)
    y = np.asarray([r["complexity"] for r in rows])
    assert not np.isinf(X).any() and np.isfinite(y).all()
    return rows, X, y, excluded


def make_search(model, workers, seed, tiny=False, inner_splits=None):
    if model == "lasso":
        estimator = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Lasso(max_iter=100000, tol=1e-6, selection="cyclic")),
        ])
        grid = {"model__alpha": [0.01, 0.1] if tiny else ALPHAS}
        scoring = "neg_mean_squared_error"
    else:
        estimator = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", XGBRegressor(random_state=seed, n_jobs=1, verbosity=0, importance_type="gain")),
        ])
        raw_grid = {"n_estimators": [5], "max_depth": [2]} if tiny else XGB_GRID
        grid = {f"model__{key}": value for key, value in raw_grid.items()}
        scoring = "neg_mean_absolute_error"
    return GridSearchCV(
        estimator, grid, scoring=scoring,
        cv=inner_splits if inner_splits is not None else KFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=workers, pre_dispatch=workers, refit=True, error_score="raise",
        return_train_score=False,
    )


def fit_outer_fold(model, X, y, train, test, workers, seed, tiny=False, inner_splits=None):
    assert not set(train) & set(test)
    assert not np.isnan(X[train]).all(axis=0).any(), "A training feature is entirely missing"
    search = make_search(model, workers, seed, tiny, inner_splits)
    # Both the search and its final refit receive ONLY the outer training set.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        search.fit(X[train], y[train])
    prediction = search.predict(X[test])
    best = search.best_estimator_
    detail = {"best_params": search.best_params_, "inner_best_score": float(search.best_score_),
              "imputer_statistics": best["impute"].statistics_.tolist(),
              "training_missing_counts": np.isnan(X[train]).sum(axis=0).tolist()}
    if model == "lasso":
        detail.update(coefficients=best["model"].coef_.tolist(),
                      scaler_mean=best["scale"].mean_.tolist(),
                      scaler_n_samples=int(best["scale"].n_samples_seen_),
                      intercept=float(best["model"].intercept_))
    else:
        detail["gain_importance"] = best["model"].feature_importances_.tolist()
    return prediction, detail


def scores(y, prediction):
    return {"r2": float(r2_score(y, prediction)), "mae": float(mean_absolute_error(y, prediction))}


def summarize(y, predictions, folds):
    result = {}
    for model, predicted in predictions.items():
        per_fold = [scores(y[test], predicted[test]) for _, test in folds]
        result[model] = {
            "folds": per_fold,
            "r2_mean": float(np.mean([f["r2"] for f in per_fold])),
            "r2_std": float(np.std([f["r2"] for f in per_fold])),
            "mae_mean": float(np.mean([f["mae"] for f in per_fold])),
            "mae_std": float(np.std([f["mae"] for f in per_fold])),
            "pooled_oof": scores(y, predicted),
        }
    baseline = result["baseline"]["mae_mean"]
    for result_row in result.values():
        result_row["mae_improvement_pct"] = 100 * (baseline - result_row["mae_mean"]) / baseline
    return result


def nested_splits(ids, seed, groups=None):
    """Return task-ID splits; a group is held together at both CV levels."""
    def split(indices):
        if groups is None:
            return KFold(5, shuffle=True, random_state=seed).split(indices)
        labels = [groups[ids[j]] for j in indices]
        return GroupKFold(5, shuffle=True, random_state=seed).split(indices, groups=labels)

    return [
        {"fold": i + 1, "train_ids": [ids[j] for j in train], "test_ids": [ids[j] for j in test],
         "inner_splits": [
             {"train_ids": [ids[train[j]] for j in itrain],
              "validation_ids": [ids[train[j]] for j in ival]}
             for itrain, ival in split(train)]}
        for i, (train, test) in enumerate(split(np.arange(len(ids))))]


def validate_splits(ids, saved_splits, groups=None):
    """Check coverage and isolation before fitting, including inner validation."""
    assert len(ids) == len(set(ids))
    if groups is not None:
        assert set(groups) == set(ids)
    tests = []
    for s in saved_splits:
        train, test = set(s['train_ids']), set(s['test_ids'])
        assert not train & test and train | test == set(ids)
        tests.extend(s['test_ids'])
        validations = []
        for inner in s['inner_splits']:
            a, b = set(inner['train_ids']), set(inner['validation_ids'])
            assert not a & b and a | b == train
            validations.extend(inner['validation_ids'])
            if groups is not None:
                assert not {groups[t] for t in a} & {groups[t] for t in b}
        assert sorted(validations) == sorted(train)
        if groups is not None:
            assert not {groups[t] for t in train} & {groups[t] for t in test}
    assert sorted(tests) == sorted(ids)


def run(name, policy, output, workers, seed, feature_root=None, splits_from=None,
        dataset_snapshot=None, groups=None):
    if dataset_snapshot is None:
        rows, X, y, excluded = load_dataset(name, policy, feature_root)
    else:
        assert dataset_snapshot['feature_names'] == FEATURES
        rows, excluded = dataset_snapshot['rows'], dataset_snapshot['excluded']
        X = np.asarray([r['features'] for r in rows], dtype=float)
        y = np.asarray([r['complexity'] for r in rows])
    tag = name if name == "verified" or policy == "historical" else f"arrow_{policy}"
    if groups is not None:
        assert name == 'arrow' and policy == 'oracle_pass'
        tag += '_grouped'
    destination = output / tag
    destination.mkdir(parents=True, exist_ok=False)
    ids = [r["task_id"] for r in rows]
    if splits_from:
        previous = json.loads((Path(splits_from) / tag / "dataset.json").read_text())
        assert previous["feature_names"] == FEATURES
        assert previous["excluded"] == excluded
        for old, new in zip(previous["rows"], rows, strict=True):
            assert {k: v for k, v in old.items() if k != "features"} == {k: v for k, v in new.items() if k != "features"}
        saved_splits = json.loads((Path(splits_from) / tag / "splits.json").read_text())
    else:
        saved_splits = nested_splits(ids, seed, groups)
    validate_splits(ids, saved_splits, groups)
    lookup = {task: i for i, task in enumerate(ids)}
    folds = [(np.asarray([lookup[t] for t in s["train_ids"]]),
              np.asarray([lookup[t] for t in s["test_ids"]])) for s in saved_splits]
    dump(destination / "dataset.json", {"feature_names": FEATURES, "rows": rows, "excluded": excluded})
    dump(destination / "splits.json", saved_splits)
    predictions = {model: np.full(len(y), np.nan) for model in ("baseline", "lasso", "xgboost")}
    details = defaultdict(list)
    coverage = np.zeros(len(y), dtype=int)
    for i, (train, test) in enumerate(folds, 1):
        assert not set(train) & set(test) and set(train) | set(test) == set(range(len(ids)))
        inner_lookup = {ids[j]: k for k, j in enumerate(train)}
        inner_splits = [(np.asarray([inner_lookup[t] for t in s["train_ids"]]),
                         np.asarray([inner_lookup[t] for t in s["validation_ids"]]))
                        for s in saved_splits[i - 1]["inner_splits"]]
        for itrain, ival in inner_splits:
            assert not set(itrain) & set(ival) and set(itrain) | set(ival) == set(range(len(train)))
            assert not np.isnan(X[train][itrain]).all(axis=0).any()
        coverage[test] += 1
        predictions["baseline"][test] = y[train].mean()
        for model in ("lasso", "xgboost"):
            start = time.monotonic()
            print(f"{tag} fold {i}/5 {model}: start ({len(train)} train, {len(test)} test)", flush=True)
            predicted, detail = fit_outer_fold(model, X, y, train, test, workers, seed, inner_splits=inner_splits)
            predictions[model][test] = predicted
            detail.update(fold=i, elapsed_seconds=time.monotonic() - start, **scores(y[test], predicted))
            details[model].append(detail)
            dump(destination / f"{model}_fold_{i}.json", detail)
            print(f"{tag} fold {i}/5 {model}: R2={detail['r2']:.4f} MAE={detail['mae']:.4f} ({detail['elapsed_seconds']:.1f}s)", flush=True)
    assert (coverage == 1).all()
    assert all(np.isfinite(p).all() for p in predictions.values())
    result = {"dataset": name, "arrow_policy": policy if name == "arrow" else None,
              "n": len(y), "target_mean": float(y.mean()), "target_std": float(y.std()),
              "metrics": summarize(y, predictions, folds), "model_folds": details}
    coefficients = np.asarray([d["coefficients"] for d in details["lasso"]])
    gains = np.asarray([d["gain_importance"] for d in details["xgboost"]])
    result["features"] = [
        {"name": feature, "lasso_mean": float(coefficients[:, j].mean()),
         "lasso_std": float(coefficients[:, j].std()),
         "lasso_selected_folds": int((np.abs(coefficients[:, j]) > 1e-8).sum()),
         "gain_mean": float(gains[:, j].mean()), "gain_std": float(gains[:, j].std())}
        for j, feature in enumerate(FEATURES)
    ]
    with (destination / "oof_predictions.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["task_id", "outer_fold", "complexity", *predictions])
        fold_numbers = {j: i + 1 for i, (_, test) in enumerate(folds) for j in test}
        for j, task in enumerate(ids):
            writer.writerow([task, fold_numbers[j], y[j], *(p[j] for p in predictions.values())])
    dump(destination / "results.json", result)
    print(tag, json.dumps(result["metrics"]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--splits-from", type=Path)
    parser.add_argument("--scenarios", nargs="+", default=["arrow_oracle_pass"],
                        choices=["verified", "arrow", "arrow_oracle_pass", "arrow_errors_as_failures"])
    args = parser.parse_args()
    assert args.workers > 0
    if args.feature_root:
        assert args.splits_from is not None, "Reference-feature evaluation must reuse saved splits"
        feature_manifest = json.loads((args.feature_root / "manifest.json").read_text())
        for path, expected in feature_manifest["files"].items():
            assert digest(args.feature_root / path) == expected
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "inputs/manifest.json").read_text())
    for entry in manifest:
        assert digest(ROOT / entry["snapshot"]) == entry["sha256"], entry["snapshot"]
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "command": sys.argv,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "xgboost", "joblib", "threadpoolctl"]},
        "script_sha256": digest(__file__), "input_manifest_sha256": digest(ROOT / "inputs/manifest.json"),
        "seed": args.seed, "outer_folds": 5, "inner_folds": 5,
        "lasso_alphas": ALPHAS, "lasso_selection_loss": "MSE",
        "xgb_grid": XGB_GRID, "xgb_candidates": len(ParameterGrid(XGB_GRID)), "xgb_selection_loss": "MAE",
        "summary": "Unweighted mean and population SD across outer folds; pooled OOF also retained; no prediction clipping.",
        "preprocessing": "Training-fold median imputation for both models; Lasso additionally standardizes within each fit.",
        "feature_root": str(args.feature_root.resolve()) if args.feature_root else None,
        "feature_manifest_sha256": digest(args.feature_root / "manifest.json") if args.feature_root else None,
        "splits_from": str(args.splits_from.resolve()) if args.splits_from else None,
    }
    dump(output / "metadata.json", metadata)
    for scenario in args.scenarios:
        name = "verified" if scenario == "verified" else "arrow"
        policy = scenario.removeprefix("arrow_") if scenario.startswith("arrow_") else "historical"
        run(name, policy, output, args.workers, args.seed, args.feature_root, args.splits_from)
    metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
