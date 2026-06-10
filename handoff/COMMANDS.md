# Handoff Commands

Activate the environment first:

```bash
cd handoff
source .venv/bin/activate
```

## Tests

```bash
# standalone (inside handoff/)
pytest -q
python scripts/verify_artifacts.py

# from memgrid monorepo root
pytest -q
python handoff/scripts/verify_artifacts.py
pytest -q handoff/tests/test_handoff_grid_scripts_smoke.py
```

## Grid DAM (Parts 4 + 4b)

```bash
python scripts/run_grid_multiscale.py --quick
python scripts/run_grid_multiscale.py --output-dir results/grid/multiscale
python scripts/run_grid_breakit.py --experiment all --quick
python scripts/run_grid_breakit.py --experiment orientations --quick
```

## Vision experiments

```bash
python scripts/run_vision_synthetic.py --quick
python scripts/run_vision_model_comparison.py --models vit_base_patch16_224
python scripts/run_vision_naturalistic_baseline.py --category fruits --include-clip
python scripts/run_vision_naturalistic_dam.py --category animals --max-stage-a-configs 2
# full sweep (hours): omit --quick and pass explicit caps or none
# notebook live regen (capped): RUN_EXPERIMENTS=True, QUICK_MODE=False in VisionExperiments.ipynb
python scripts/run_vision_animals_graphs.py --max-settings 1 --max-seeds 1 --max-layers 2
```

## Notebook execution

Open `notebooks/GridDAM.ipynb` or `notebooks/VisionExperiments.ipynb` with kernel cwd at `handoff/` or `handoff/notebooks/`.

Set `QUICK_MODE = True` for a fast local run. `VisionExperiments.ipynb` defaults to `RUN_EXPERIMENTS = False` (reload bundled results from `handoff/results/`).

```bash
jupyter nbconvert --execute notebooks/GridDAM.ipynb --ExecutePreprocessor.timeout=300
jupyter nbconvert --execute notebooks/VisionExperiments.ipynb --ExecutePreprocessor.timeout=600
```

## Standalone layout validation

```bash
python scripts/bootstrap_standalone.py
```
