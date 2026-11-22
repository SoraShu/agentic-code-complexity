# Data and public sources

This directory keeps outputs produced for the thesis, not copies of publicly
obtainable original datasets. See each subdirectory's README for field definitions.

| Retained material | Role |
|---|---|
| `features/` | Final 19-feature extraction for 500 Verified and 73 Arrow candidates |
| `outcomes/arrow/` | This study's ten solver-run summaries and oracle evaluation |
| `targets/pro.json` | This study's 731 task-level aggregates from 13 public outcome reports |
| `targets/pro-complete-traceable.json` | Nine-report aggregates for the same 381 tasks covered by all 13 reports |
| `results/` | Final Arrow OOF predictions/folds, Pro Judge predictions and published aggregates |
| `selection/` | Arrow screening counts, 100 construction outcomes, candidate IDs and saved PR labels |
| `reference-inputs.json` | Task/repository/base identities, source-file and reference-patch hashes; no code text |
| `sources.json` | Official source URLs, benchmark configurations, pinned Pro result commit and report hashes |

Verified outcome reports and their task-level derived targets/predictions remain in
the separate private archive. The public Verified feature values do not include
those outcomes.

## Obtain benchmark tasks

Official sources:

- [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified), test split, 500 tasks.
- [SWE-bench Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro), **v1** configuration, test split, 731 tasks. The default now refers to V2; the thesis uses V1. See the [official version instructions](https://github.com/scaleapi/SWE-bench_Pro-os#swe-bench-pro-v2).
- [Arrow](https://github.com/arrow-py/arrow), pinned task base/fix commits and public PR URLs recorded in `reference-inputs.json`.

For example, export the original task records into ignored local storage:

```bash
uv run --with datasets python - <<'PY'
from pathlib import Path
from datasets import load_dataset
out = Path('work/benchmark-records')
out.mkdir(parents=True, exist_ok=True)
load_dataset('princeton-nlp/SWE-bench_Verified', split='test').to_json(out/'verified.jsonl')
load_dataset('ScaleAI/SWE-bench_Pro', 'v1', split='test').to_json(out/'pro-v1.jsonl')
PY
```

These are original benchmark records. They are sufficient to retrieve reference
patches and problem text; they are not a promise to reconstruct the exact historical
agent-facing instruction wrappers or the early 20-shot exemplar set.

## Obtain public source files for feature extraction

```bash
uv run scripts/fetch_reference_sources.py \
  --verified-jsonl work/benchmark-records/verified.jsonl \
  --output work/reference-sources.jsonl.gz
uv run --project packages/regression --locked python packages/regression/extract_reference_features.py \
  --source-bundle work/reference-sources.jsonl.gz --output work/reextracted-features
```

The acquisition script retrieves source files at pinned base commits and Arrow PR
diffs, checking every file and patch against the recorded SHA-256. If an upstream
diff differs, it stops; it does not substitute PR head or a newer patch. An existing
matching public-source cache can be supplied with `--base-cache` and `--arrow-patches`.
The latter contains `pr-N.diff` files. For a download smoke test, select one task
with `--task-id`. The full bundle is generated locally and excluded from Git.

## Obtain Pro outcome reports

```bash
uv run scripts/fetch_pro_reports.py --output work/pro-reports
```

This downloads the **nine independently matched** Scale AI reports at commit
`66f92766bba642462d4bbe5479e83f91f9211862` and converts them to the expected CSV
schema. Both upstream bytes and converted CSVs are checked. Step/token zeros are
conversion placeholders, not measured resource use.

These nine reports suffice to reproduce the report-population sensitivity:

```bash
uv run scripts/aggregate_pro_targets.py --reports work/pro-reports --traceable-only \
  --output work/rebuilt-pro-complete-traceable.json
```

The script uses the 381 complete-coverage task IDs from `targets/pro.json` and
checks equality with `targets/pro-complete-traceable.json`. It uses the same
saved Judge predictions as the 13-report comparison.

Four other historical reports were identified by the author as originating from
Scale AI, but their exact upstream run mapping was not recovered. Their identifiers
and hashes are listed in `sources.json`. The downloader does not invent URLs for
these reports or call nine reports an exact reproduction of all thirteen.

The published derived `targets/pro.json` allows saved-prediction verification without
duplicating raw reports. Given authorized copies of **all 13** originals, rebuild it:

```bash
uv run scripts/aggregate_pro_targets.py --reports /path/to/all-13-reports \
  --output work/rebuilt-pro-targets.json
```

The script checks the original report hashes, joins the exact 731-task cohort and
verifies equality with the published aggregate. It excludes three `missing` outcomes
from denominators and retains every task. No model calls are made.
