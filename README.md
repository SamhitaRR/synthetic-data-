# Synthetic Data–Based Cell Segmentation Pipeline

This repo covers four related processes for generating synthetic training data,
training/running a multi-channel nnU-Net segmentation model, converting its output
into instance segmentation masks, and manually correcting/annotating cells with a
napari plugin.

## Contents

| Folder | Process |
|---|---|
| [`01_synthetic_data_generation`](./01_synthetic_data_generation/README.md) | Generate synthetic 3D fluorescence training data, apply PSF blur, prepare labels, and train an nnU-Net model on it |
| [`02_nnunet_setup_inference`](./02_nnunet_setup_inference/README.md) | Install nnU-Net, install the pretrained model, and run inference on new images |
| [`03_instance_segmentation`](./03_instance_segmentation/README.md) | Convert nnU-Net's mask + centre outputs into per-cell instance segmentation masks |
| [`04_napari_ovoid_fitter`](./04_napari_ovoid_fitter/README.md) | A napari widget for manually segmenting cells by fitting 3D ovoids to clicked boundary points |

## Shared files

The [`shared/`](./shared) folder contains files used by more than one process:

- `shared/nnunetv2/` — the custom nnU-Net package folder (multi-channel preprocessor,
  label manager, and trainer). Used by both Step 1 (training) and Step 2 (inference)
  — see each folder's README for where to copy it in.

## Suggested order

1. Start with **01_synthetic_data_generation** if you need to generate training data
   and train your own model from scratch. Skip straight to **02_nnunet_setup_inference**
   if you just want to use the pretrained model provided there to run inference on
   your own images — no training required.
2. Run **03_instance_segmentation** on the output of Step 2.
3. Use **04_napari_ovoid_fitter** independently, at any point, for manual
   segmentation or correcting automated results.
