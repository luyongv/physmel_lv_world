# PhysMel LV World

PhysMel LV World is a research codebase for melanoma outcome modeling with
patient-level tumor-immune dynamics. It combines frozen pathology, transcriptomic,
and immune-fraction features with a constrained Lotka-Volterra world model and a
survival-risk head.

The repository is organized as an executable Python package rather than a notebook
dump. Configuration files define the experiment, gateway modules run the workflows,
and the model internals live under `src/physmel_lv_world`.

## What Is In This Repository

| Area | Path | Purpose |
| --- | --- | --- |
| Command gateways | `src/physmel_lv_world/gateways` | Prepare, train, score, and infer entry points |
| Feature fusion | `src/physmel_lv_world/signal` | Multimodal fusion for frozen encoder outputs |
| Dynamic model | `src/physmel_lv_world/engines` | ODE heads, latent dynamics, and basis modules |
| Survival logic | `src/physmel_lv_world/riskcraft` | Cox objective, trajectory summaries, metrics, reports |
| Data loading | `src/physmel_lv_world/atlas` | Manifest parsing, folds, and batch records |
| Training utilities | `src/physmel_lv_world/trial` | Optimizer, loop, checkpoints, split plans |
| Run settings | `settings` | Tiny checks and long-horizon experiment configs |
| Smoke tests | `bench` | Minimal tests for data, losses, metrics, and training |

## Model Pipeline

1. Load case manifests and precomputed tensors.
2. Fuse pathology, gene, and immune feature vectors.
3. Predict patient-specific dynamic parameters.
4. Integrate tumor-immune trajectories over the configured horizon.
5. Convert trajectories into survival features.
6. Optimize Cox likelihood with physics and residual penalties.
7. Score folds with concordance, time-AUC, Brier, and report helpers.

Foundation encoders are intentionally outside this package. The training code
expects their outputs as tensors referenced by a manifest.

## Environment

Python 3.10 is required.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda users can build the bundled environment:

```bash
conda env create -f environment.yml
conda activate physmel_lv_world
python -m pip install -e .
```

## Quick Check

Use the CPU-sized config before launching full experiments:

```bash
python -m physmel_lv_world.gateways.train settings/checks/tiny.yaml
```

The installed console script is equivalent:

```bash
physmel-train settings/checks/tiny.yaml
```

## Data Contract

The manifest consumed by the training and scoring code should include:

```text
case_id, path_embed, gene_embed, immune_fraction, time, event, fold, endpoint, site, region
```

Feature paths should point to tensor files under the local data area used for the
run. Public source references are recorded in `notes/DATASETS.txt`; access rules
for controlled clinical or genomic material remain the responsibility of the user.

## Main Commands

Prepare manifests:

```bash
bash bin/prepare_manifest.sh
```

Train the primary long-horizon configuration:

```bash
python -m physmel_lv_world.gateways.train settings/longhaul/main.yaml
```

Run ablations:

```bash
python -m physmel_lv_world.gateways.train settings/longhaul/ablate_free_ode.yaml
python -m physmel_lv_world.gateways.train settings/longhaul/ablate_path_only.yaml
python -m physmel_lv_world.gateways.train settings/longhaul/ablate_gene_only.yaml
python -m physmel_lv_world.gateways.train settings/longhaul/ablate_no_physics.yaml
```

Score a checkpoint:

```bash
python -m physmel_lv_world.gateways.score settings/longhaul/main.yaml --checkpoint checkpoints/main.pt
```

## Configuration Notes

`settings/checks/tiny.yaml` is for local validation and fast CI-style checks.
`settings/longhaul/main.yaml` and its ablation variants are intended for the
reported experimental workflow and assume substantially larger compute.

The long-horizon defaults target multi-GPU review runs. For a workstation run,
start from the tiny config and scale dimensions, epochs, batch size, and trajectory
points deliberately.

## Validation

Run the smoke test suite with:

```bash
python -m pytest
```

The tests exercise manifest handling, model loss calculation, metrics, and a small
training path. They do not replace full fold-and-seed survival evaluation.

## Reference Notes

Additional repository context is kept in `notes/`:

- `notes/MAP.txt` maps manuscript concepts to implementation files.
- `notes/DATASETS.txt` lists public dataset locations.
- `notes/DEVIATIONS.txt` records implementation deviations.
- `notes/CONTEXT.txt` summarizes project assumptions.

## Checkpoints

Large trained checkpoints are not stored in this repository. Keep generated
artifacts under the configured checkpoint or run directories, and publish them
separately when required by a release workflow.
