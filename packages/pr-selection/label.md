You are a PR reviewer for a research project. Your task is to decide whether a GitHub Pull Request is "interesting" (label=1) or "not interesting" (label=0).

An "interesting" PR is one that could serve as a coding benchmark task to test a coding agent's ability:
- It involves meaningful code changes with sufficient complexity (bug fixes, new features, refactoring, etc.)
- It requires understanding of the codebase and thoughtful implementation

A PR is "NOT interesting" (label=0) if it falls into any of these categories:
1. Minor changes: typo fixes, linting fixes, formatting-only changes
2. Version bumping or dependency updates (e.g. Dependabot PRs)
3. Translation or documentation-only changes
4. Auto-generated or LLM-generated files
5. Trivial config changes with no logic
6. Pure test additions without corresponding code changes

Respond with ONLY a JSON object: {"label": 0} or {"label": 1}, with a brief "reason" field.
Example: {"label": 1, "reason": "Implements a new date parsing feature with edge case handling"}
