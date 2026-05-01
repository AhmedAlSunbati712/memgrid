# Repository Guidelines

## Project Structure & Module Organization
Core research code lives at the repository root: `DAM.py`, `Encoder.py`, `Hopfield.py`, `experiments.py`, `plotting.py`, `helpers.py`, and `utils.py`. Vision-specific code is under `vision/` (`image_generator.py`, `model_wrapper.py`, `feature_preprocess.py`, `run_synthetic_features.py`). Tests live in `tests/` and focus on smoke coverage for model APIs, notebook definitions, and multiscale parity. Generated artifacts and cached outputs belong in `results/` and should stay out of core logic changes. Exploratory work is kept in notebooks such as `PerformanceAnalysis.ipynb` and `VisionFrontendDemo.ipynb`.

## Build, Test, and Development Commands
Use the project virtual environment before running code: `source .venv/bin/activate`.

- `pytest -q` runs the full test suite.
- `pytest -q tests/test_smoke.py` checks core DAM, encoder, plotting, and experiment entrypoints.
- `pytest -q tests/test_vision_smoke.py` runs vision-module smoke tests; `timm`-dependent coverage is skipped if unavailable.
- `python vision/run_synthetic_features.py --output-dir results/vision` generates synthetic image features and preview assets.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and explicit imports. Keep modules small and numerical code vectorized where practical; this repo already uses `numpy` and `numba` for performance-sensitive paths. Prefer concise docstrings on public methods and avoid introducing notebook-only logic into reusable modules.

## Testing Guidelines
Add or update `pytest` smoke tests for every change to retrieval logic, experiment orchestration, or vision preprocessing. Name tests `test_<behavior>.py` and keep randomness seeded with `np.random.seed(...)` for reproducibility. When changing notebook-backed workflows, extend the corresponding smoke coverage in `tests/test_performance_analysis_smoke.py` instead of relying on manual notebook execution.

## Commit & Pull Request Guidelines
Recent history uses short, descriptive commit subjects such as `refactoring`, `finalized optimizations`, and `adding smoke tests + refactoring out plotting functionalities`. Keep commit titles brief, lower-case when practical, and focused on the main change. PRs should include a short summary, affected modules or notebooks, exact test commands run, and screenshots or regenerated paths when outputs in `results/` or notebook visuals change.

## Research Artifact Hygiene
Do not commit large generated files unless they are intentional deliverables. Keep reproducible scripts in code, keep intermediate outputs under `results/`, and document any new external dependency or model requirement near the code that uses it.

## Things to do after implementing something new
- Update @AGENT_MEMORY_2.MD: Explain what you implemented, how was it tested and also next steps on what should be implemented next.
