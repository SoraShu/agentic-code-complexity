# Pro Judge predictions

`pro-zero-shot.jsonl` contains the 731 saved Claude Opus 4.7 predictions from the shared early three-step rubric. The Judge receives task/source context, the reference patch and evaluation-test information. Pro targets are in `../../targets/pro.json`. The 15 predictions lacking source context remain included. Verified zero-/20-shot predictions are held in the private archive.

The shared rubric is reconstructed from historical source and run records;
complete per-call prompts are unavailable. See `../../../packages/judge/README.md`
for the early and released message layouts and response reuse. The saved scores
are not outputs from a new execution of the released wrapper.

`scripts/verify_results.py` also evaluates a constant probability of 0.50 on the
same 731 targets and uses `../../targets/pro-complete-traceable.json` for the
nine-report sensitivity on the fixed 381-task complete-coverage subset. Constant
predictions have undefined Pearson/Spearman correlation, represented as JSON null.
