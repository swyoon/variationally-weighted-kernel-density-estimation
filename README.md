# variationally-weighted-kernel-density-estimation

This repository contains a PyTorch implementation for the paper [Variational Weighting for Kernel Density Ratios](https://arxiv.org/pdf/2311.03001.pdf), presented at NeurIPS 2023.

## Dependencies

* PyTorch

* Scikit-learn

* SciPy

* NumPy

* CuPy

## Running the experiments
```bash
python train.py --model [model name] --device [device]
```

Here `model name` is one of the following:

- `KDE` This is the naive Kernel Density Estimation model.
- `based` Corresponds to VWKDE-MB in Figure 4.
- `free` Corresponds to VWKDE-MF in Figure 4.

Example:
```bash
python train.py --model free --device cuda
```