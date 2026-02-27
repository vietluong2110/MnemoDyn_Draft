# MnemoDyn: Learning Resting State Dynamics from 40K fMRI Sequences

[[Paper]]() [[Poster]]() [[Slide]]()

### Sourav Pal, Viet Luong, Hoseok Lee, Tingting Dan, Guorong Wu, Richard Davidson, Won Hwa Kim, Vikas Singh

![MnemoDyn architecture](asset/braine-1.png)

MnemoDyn is an operator-learning foundation model for resting-state fMRI, combining multi-resolution wavelet dynamics with CDE-style temporal modeling.

## Update

MnemoDyn is now published on Hugging Face: https://huggingface.co/vhluong/MnemoDyn

You can also publish your own trained checkpoint directly from this repo.

## Tutorial

A usage walkthrough is available as a Google Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1IeWYPmwZAj5zA_khQHmgKOXjF8DJXVNo/view?usp=sharing)

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

<!-- ## Data Processing Flow

MnemoDyn expects parcellated rs-fMRI time series data (`*.dtseries.nii`) as input.

If you are starting from volumetric NIfTI files (e.g., from fMRIPrep), you must run them through our **Preprocessing Pipeline** (described above) before training. This ensures proper alignment and time-step continuity.

To use custom datasets:
1. Preprocess your NIfTI files through `coe.preprocess.pipeline`.
2. Ensure you have dataset metadata CSV/TSV files (labels, demographics, IDs).
3. Update the hardcoded dataset paths (e.g., `/mnt/sourav/HBN_dtseries/`) in the downstream training launch scripts (`coe/light/script/*.sh`) to point to your new output directories. -->

## Preprocessing Pipeline (NIfTI to Parcellated CIFTI)

We provide a unified, Python-based CLI pipeline to automate mapping volumetric NIfTI images to fs_LR surfaces and parcellating the resulting dense time series. The pipeline dynamically extracts the Repetition Time (TR) from your NIfTI files to ensure downstream models learn accurate temporal dynamics.

### Requirements
- Connectome Workbench (`wb_command`) installed and on your system PATH.
- `nibabel` and `tqdm` Python packages.

### Usage
Run the pipeline from the repository root:

```bash
python -m coe.preprocess.pipeline \
  --input-dir /path/to/niftis \
  --output-dir /path/to/output_dir \
  --atlas /path/to/atlas.dlabel.nii \
  --pattern "*_task-rest_space-MNI305_preproc.nii.gz"
```

The script will automatically orchestrate `wb_command` for left/right mapping and resampling, output an intermediate `.dtseries.nii`, and finally parcellate it using the provided atlas, injecting the correct native TR throughout.


## Quick Start

### 1) Inspect pretraining CLIs

```bash
cd coe/light/model
python main.py --help
python main_masked_autoencode.py --help
python main_masked_autoencode_jepa.py --help
python main_denoise.py --help
```

### 2) Pretraining

```bash
bash orion.sh
```

### 3) Run downstream examples

```bash
cd coe/light
bash script/hbn_classification.sh
bash script/adhd_200_diagnose.sh
```

<!-- ## Common Script Entry Points

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
- `bash script/ukbiobank_sex_classification.sh` -->

## Typical Workflow

1. Pretrain a foundation checkpoint (`coe/light/model/main*.py`).
2. Save Lightning checkpoints under a versioned results directory.
3. Fine-tune a downstream head using a task script in `coe/light/`.
4. Track outputs and metrics under `Result/<ExperimentName>/...`.

<!-- ## Publish to Hugging Face

Install Hub client:

```bash
pip install huggingface_hub
```

Log in once:

```bash
huggingface-cli login
```

Publish a training run folder (auto-picks best checkpoint by lowest `val_mae` in filename):

```bash
python -m coe.light.model.publish_to_hf \
  --repo-id <your-hf-username>/<model-name> \
  --version-dir /path/to/version_17
```

Or publish an explicit checkpoint:

```bash
python -m coe.light.model.publish_to_hf \
  --repo-id <your-hf-username>/<model-name> \
  --checkpoint /path/to/model.ckpt \
  --hparams /path/to/hparams.yaml \
  --metrics /path/to/metrics.csv
```

Load it back:

```python
from huggingface_hub import hf_hub_download
from coe.light.model.main import LitORionModelOptimized

ckpt = hf_hub_download(repo_id="<your-hf-username>/<model-name>", filename="model.ckpt")
model = LitORionModelOptimized.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()
``` -->

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
