# Dependencies and external data

Python dependency versions are recorded in the component `uv.lock` files. They
include NumPy, SciPy, scikit-learn, XGBoost, Radon, cognitive-complexity, LiteLLM,
pandas and Pydantic. Dependencies are installed, not vendored.

SWE-bench Verified, SWE-bench Pro, Scale AI outcome records and Arrow source code
are retrieved from their respective public sources. Their links and acquisition
instructions are in `data/README.md`; original datasets, patches and source bundles
are not included in this repository.

Benchpress is an internal company repository and is not part of this publication.
The solver configuration used Harbor's native Copilot integration. The unused
external Copilot adapter is not included.
