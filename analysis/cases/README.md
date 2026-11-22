# Appendix B case evidence

The retained cases are Django 16502, SymPy 20916 and Django 15098.
`build_evidence.py` extracts selected trajectory records and observed test output,
exports the problem/reference/test/candidate patch materials, and produces the
19 feature values and final out-of-fold predictions in `case-metrics.csv`.
`compact_evidence.py` selects the excerpts used by the shortened appendix.

```bash
python analysis/cases/build_evidence.py --private-root /path/to/private-archive \
  --tasks /path/to/retrieved/verified-tasks --output work/case-materials
python analysis/cases/compact_evidence.py --materials work/case-materials \
  --output work/case-excerpts
```

Task directories contain `tests/config.json` and `instruction.md`. The source task
records come from the official Verified dataset; use the matching historical
instruction wrapper when checking exact evidence text. Internal trajectories,
candidate patches and raw evaluation logs are supplied only through the authorized
private archive and are never copied into the public source tree. The extractor
reads the adopted Boolean-only regression results, not earlier fitted predictions.
The final formatted appendix remains in the thesis repository.
