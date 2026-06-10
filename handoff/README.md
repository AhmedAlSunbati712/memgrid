# memgrid Handoff Package

Standalone package for reproducing grid DAM and vision retrieval experiments. All code, datasets, and reload artifacts live under `handoff/`; repo-root `vision/`, `experiments.py`, etc. are legacy duplicates not used by handoff.

## What to open

| Track | Notebook | Source notebooks (unchanged at repo root) |
|-------|----------|-------------------------------------------|
| Grid DAM (Parts 4 + 4b) | [`notebooks/GridDAM.ipynb`](notebooks/GridDAM.ipynb) | `DenseAM.ipynb` |
| Vision retrieval | [`notebooks/VisionExperiments.ipynb`](notebooks/VisionExperiments.ipynb) | `VisionFrontendDemo.ipynb`, `VisionLayerwiseBaseline.ipynb`, … |

## Setup (monorepo or standalone handoff checkout)

```bash
cd handoff
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Bundled under `handoff/data/` (~120 images + Peterson pickle) and pre-seeded outputs under `handoff/results/` (see [`ARTIFACTS.md`](ARTIFACTS.md)).

## Quick start

```bash
cd handoff
source .venv/bin/activate
pytest -q
python scripts/verify_artifacts.py
```

From the memgrid monorepo root:

```bash
pytest -q
python handoff/scripts/verify_artifacts.py
```

See [`COMMANDS.md`](COMMANDS.md) for CLI details and [`ARTIFACTS.md`](ARTIFACTS.md) for artifact layout.

## Directory layout

```
handoff/
  core/          # vendored grid stack (Encoder, DAM, experiments, …)
  vision/        # vendored vision modules
  data/          # datasets_peterson.pkl + animals/fruits/vegetables/
  lib/           # runners, paths bootstrap, manifest helpers
  scripts/       # CLIs (bootstrap via scripts/_bootstrap.py)
  notebooks/     # GridDAM.ipynb, VisionExperiments.ipynb
  tests/         # pytest suite (conftest.py bootstraps paths)
  results/       # bundled reload outputs + default path for new runs (MANIFEST.json)
```

Import bootstrap: `handoff/lib/paths.py` → `ensure_handoff_paths()` puts `HANDOFF_ROOT` and `core/` on `sys.path`.
