# MnemoDyn: Learning Resting State Dynamics from 40K fMRI Sequences

[[Paper]]() [[Poster]]() [[Slide]]()

### Sourav Pal, Viet Luong, Hoseok Lee, Tingting Dan, Guorong Wu, Richard Davidson, Won Hwa Kim, Vikas Singh

![MnemoDyn architecture](asset/braine-1.png)

MnemoDyn is an operator-learning foundation model for resting-state fMRI, combining multi-resolution wavelet dynamics with CDE-style temporal modeling.

## Update

The Hugging Face release will be available soon.

## At A Glance

- Pretraining backbones: `coe/light/model/main.py`, `coe/light/model/main_masked_autoencode.py`, `coe/light/model/main_masked_autoencode_jepa.py`, `coe/light/model/main_denoise.py`, `coe/light/model/orion.py`
- Core model modules: `coe/light/model/conv1d_optimize.py`, `coe/light/model/dense_layer.py`, `coe/light/model/ema.py`, `coe/light/model/normalizer.py`
- Downstream tasks: HBN, ADHD200, ADNI, ABIDE, NKIR, UK Biobank, HCP Aging under `coe/light/*.py`
- Launch scripts: `coe/light/script/*.sh`

## Repository Layout

```text
.
├── highdim_req.txt
├── pyproject.toml
├── coe/
│   ├── parcellation/
│   └── light/
│       ├── model/
│       ├── script/
│       ├── *_dataset.py
│       └── *classification*.py, *regress*.py
└── README.md
```

## Environment Setup

Python 3.10+ is recommended.

### Option A (recommended): uv

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r highdim_req.txt
```

Ensure your PyTorch build matches your CUDA stack.

## Data Expectations

Most scripts expect:

- Parcellated rs-fMRI files (`*.dtseries.nii`)
- Dataset metadata CSV/TSV files (labels, demographics, IDs)
- Local paths that are often hardcoded in scripts (`/mnt/...`)

Example paths used in code:

- `/mnt/sourav/HBN_dtseries/`
- `/mnt/sourav/ADHD200_dtseries/`
- `/mnt/vhluong/...csv`

If you are running on a new machine, update dataset paths in the training scripts before launching.

## Quick Start

### 1) Inspect pretraining CLIs

```bash
cd coe/light/model
python main.py --help
python main_masked_autoencode.py --help
python main_masked_autoencode_jepa.py --help
python main_denoise.py --help
```

### 2) Run downstream examples

```bash
cd coe/light
bash script/hbn_classification.sh
bash script/adhd_200_diagnose.sh
```

## Common Script Entry Points

From `coe/light`:

- `bash script/abide_classifcation_normal.sh`
- `bash script/adhd_200_diagnose.sh`
- `bash script/adhd_200_sex_classification.sh`
- `bash script/adni_classification_amyloid.sh`
- `bash script/adni_classification_sex.sh`
- `bash script/hbn_classification.sh`
- `bash script/hbn_regression.sh`
- `bash script/hcp_aging_450.sh`
- `bash script/hcp_aging_classification.sh`
- `bash script/hcp_aging_regress_flanker.sh`
- `bash script/hcp_aging_regress_neuroticism.sh`
- `bash script/nkir_classification.sh`
- `bash script/ukbiobank_age_regression.sh`
- `bash script/ukbiobank_sex_classification.sh`

## Typical Workflow

1. Pretrain a foundation checkpoint (`coe/light/model/main*.py`).
2. Save Lightning checkpoints under a versioned results directory.
3. Fine-tune a downstream head using a task script in `coe/light/`.
4. Track outputs and metrics under `Result/<ExperimentName>/...`.

## Notes and Caveats

- This is a research codebase and is still being consolidated.
- Some scripts may require branch-specific import/path adjustments.
- Normalization and dataset utilities are partially duplicated across modules.
- Reproducibility depends on matching preprocessing, atlas/parcellation, and dataset splits.

## Citation

If this work helps your research, please cite:

```bibtex
@inproceedings{
pal2026mnemodyn,
title={MnemoDyn: Learning Resting State Dynamics from  $40$K {FMRI} sequences},
author={Sourav Pal and Viet Luong and Hoseok Lee and Tingting Dan and Guorong Wu and Richard Davidson and Won Hwa Kim and Vikas Singh},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=zexMILcQOV}
}
```

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
