# AcousticPose local runner

This folder contains a local command-line version of the Colab notebook. It uses real audio-video or audio-motion data only. Quick runs are allowed for debugging, but paper-ready claims are gated by clip count, dataset count, significance, ablations, and improvement over baselines.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
brew install ffmpeg git-lfs
git lfs install
```

## Quick smoke run on local videos

This checks that extraction and training work, but it is not publishable evidence:

```bash
python acousticpose_local.py \
  --quick \
  --av-root /path/to/local/videos \
  --project-root acousticpose_runs/smoke
```

## Real evidence run

For a serious AAAI-style run, use multiple real datasets and keep strict mode enabled:

```bash
python acousticpose_local.py \
  --download-ravdess full \
  --download-cremad \
  --download-meld \
  --beat-root /path/to/BEAT \
  --talkshow-root /path/to/TalkSHOW \
  --project-root acousticpose_runs/full_real \
  --epochs 80 \
  --strict
```

Outputs are written under `<project-root>/outputs`, including:

- `tables/main_real_results.csv`
- `tables/ablation_real_results.csv`
- `tables/significance_report.csv`
- `tables/reviewer_proof_gate.csv`
- `figures/main_real_mae.png`
- `models/*.pt`

The script will not print a paper-ready verdict unless the strict reviewer gate passes. If it fails, add stronger datasets, improve the model, or revise the claim.
