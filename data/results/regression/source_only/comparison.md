# Complete-PR and source-only comparison

All 19 features exclude repository-root `tests/` in the adopted analysis. Each comparison retains the original task order, outcome targets and inner/outer CV splits. Models use the same full search grids. The original fitted results are preserved; this comparison is outside the thesis.

| Scenario | Model | Complete PR R² | Source-only R² | Complete PR MAE | Source-only MAE |
|---|---|---:|---:|---:|---:|
| arrow_oracle_pass | baseline | -0.215 | -0.215 | 0.326 | 0.326 |
| arrow_oracle_pass | median | -0.466 | -0.466 | 0.227 | 0.227 |
| arrow_oracle_pass | lasso | -0.200 | -0.215 | 0.320 | 0.326 |
| arrow_oracle_pass | xgboost | -0.363 | -0.243 | 0.304 | 0.291 |
| arrow | baseline | -0.021 | -0.021 | 0.320 | 0.320 |
| arrow | median | -0.421 | -0.421 | 0.235 | 0.235 |
| arrow | lasso | -0.223 | -0.021 | 0.330 | 0.320 |
| arrow | xgboost | -0.181 | -0.022 | 0.300 | 0.265 |
| arrow_errors_as_failures | baseline | -0.017 | -0.017 | 0.318 | 0.318 |
| arrow_errors_as_failures | median | -0.432 | -0.432 | 0.238 | 0.238 |
| arrow_errors_as_failures | lasso | -0.212 | -0.017 | 0.324 | 0.318 |
| arrow_errors_as_failures | xgboost | -0.117 | -0.077 | 0.294 | 0.274 |
| arrow_oracle_pass_grouped | baseline | -0.202 | -0.202 | 0.320 | 0.320 |
| arrow_oracle_pass_grouped | median | -0.411 | -0.411 | 0.226 | 0.226 |
| arrow_oracle_pass_grouped | lasso | -0.170 | -0.202 | 0.317 | 0.320 |
| arrow_oracle_pass_grouped | xgboost | -0.257 | -0.308 | 0.315 | 0.291 |

## Main Arrow correlations

| Feature | Complete PR N | Complete PR rho | Source-only N | Source-only rho |
|---|---:|---:|---:|---:|
| lines_changed | 55 | 0.138 | 55 | 0.101 |
| additions | 55 | 0.180 | 55 | 0.147 |
| deletions | 55 | -0.148 | 55 | -0.121 |
| files_changed | 55 | 0.160 | 55 | 0.109 |
| top_level_dirs | 55 | 0.143 | 55 | 0.066 |
| second_level_dirs | 55 | -0.132 | 55 | -0.132 |
| cc_total | 55 | 0.317 | 55 | 0.206 |
| cc_mean | 55 | 0.335 | 55 | 0.058 |
| cc_max | 55 | 0.302 | 55 | -0.029 |
| cog_total | 55 | 0.122 | 55 | 0.130 |
| cog_mean | 55 | -0.044 | 55 | 0.078 |
| cog_max | 55 | -0.028 | 55 | -0.026 |
| mi_total | 50 | -0.279 | 48 | 0.086 |
| mi_mean | 50 | -0.261 | 48 | 0.157 |
| mi_min | 50 | -0.311 | 48 | -0.224 |
| hv_total | 55 | 0.321 | 55 | 0.019 |
| hv_mean | 51 | 0.253 | 49 | -0.198 |
| loc_total | 55 | 0.250 | 55 | 0.219 |
| loc_mean | 51 | 0.153 | 49 | 0.144 |

The primary XGBoost MAE changes from 0.304 to 0.291. Its R² varies across
folds from -1.634 to 0.466; the median baseline remains lowest in mean MAE.
Lasso selects zero coefficients in every source-only fold across all four
Arrow scenarios and reproduces the training-fold mean. Original PR tests
contribute to the complete-PR structural summaries; changing file scope is
not an intervention on task outcomes. MI and file-level means have different
available-task counts after filtering.

PR 700's unavailable reference file is under `tests/`, so source-only structural
features are available in the extended set. PR 798 still has unavailable source
files. All tasks remain in their original cohorts. Verified has no changed
feature vectors; its saved regression results and recomputed correlations are
unchanged. No solving agents or Judges were rerun.
