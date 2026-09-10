# Synthetic Vacuole Data → nnU-Net Training Pipeline

This repo covers two things:

1. Generating synthetic 3D fluorescence microscopy training data (cells with
   overlaps and dimmer vacuoles), applying PSF blur, and preparing labels.
2. Feeding that data into nnU-Net (v2) and training a multi-channel
   segmentation model on it.

> Replace every `DatasetXXX_YourDatasetName` / `XXX` placeholder below with
> your actual nnU-Net dataset ID and name (e.g. `Dataset063_noPSFbrightvacuoles`).

## Contents of this repo

- `01_synthetic_cell_data_overlaps_centres_dimmervacuoles.py` — synthetic data generator
- `convolute_psf.py` — PSF convolution script
- `PSF_NA1-4.tif` — the PSF used for convolution
- `make_spacing_jsons.py` — spacing JSON sidecar generator
- `generate_dataset_json_multichannel.py` — `dataset.json` generator
- `nnunetv2/` — custom nnU-Net package folder (multi-channel preprocessor, label manager, trainer)

---

## Requirements

For the synthetic data generator:

```bash
pip install numpy scipy tifffile
```

Everything else it uses (`os`, `time`, `concurrent.futures`) is part of the
Python standard library. `convolute_psf.py` additionally needs `scipy.signal`
(included with scipy).

For the nnU-Net side, see the environment setup below.

---

## 1. Generate synthetic training data

Run the generator script:

```bash
python 01_synthetic_cell_data_overlaps_centres_dimmervacuoles.py
```

By default (`main()`), this sweeps a grid of cell counts and intensity means
and generates one synthetic volume per combination using a process pool. Key
parameters live in `generate_synthetic_fluorescence()` if you want to
customize a single run (volume size, cell radius distribution, vacuole
fraction, blur, etc.) — see the docstring at the top of the file for details
on how it differs from the plain-overlap variant.

Each generated case writes 4 files into `out_dir`:

| File | Location | Contents |
|---|---|---|
| `img_XXX_0000.tif` | `imagesTr/` | Synthetic fluorescence volume (single channel, normalized to 16-bit) |
| `img_XXX.tif` | `labelsTr/` | Plain foreground/background mask |
| `img_XXX_seg1.tif` | `labelsTr/` | Not important — ignore |
| `img_XXX_seg2.tif` | `labelsTr/` | Union of shrunken per-cell "centre" masks |

These three label channels per case are what the custom `MultiChannelLabelManager`
(Step 10) expects — they're treated together as a multi-channel segmentation
target, not as three separate single-channel problems:

- `img_XXX.tif` (plain mask) and `_seg1` are both required simply because
  nnU-Net is set up to expect 3 label channels per case — `_seg1` needs to
  be present for the run to work, but isn't something you need to look at
  or use.
- `_seg2` (the centres) is the one worth paying attention to downstream:
  once the model is trained, its predicted mask and predicted centres can be
  combined to get instance segmentation.

Before this data is ready for nnU-Net, the images need PSF convolution
(Step 2) and the labels need binarizing (Step 3).

---

## 2. Convolve images with the PSF

Optical blur is applied by convolving the raw synthetic images from Step 1
with a measured point-spread function, using `convolute_psf.py`. The PSF
file it uses (`PSF_NA1-4.tif`) is included in this repo.

Edit the paths at the top of the script before running:

```python
input_folder = "/path/to/synthetic_output/imagesTr"
output_folder = "/path/to/synthetic_output/imagesTrPSF"
...
psf = imread("/path/to/PSF_NA1-4.tif").astype(np.float32)
```

Then run:

```bash
python convolute_psf.py
```

This convolves every `.tif`/`.tiff` file in `input_folder` with the
normalized PSF (FFT convolution, same-size output) and writes the result to
`output_folder` under the same filename.

These PSF-convolved images — not the raw ones from Step 1 — are what should
end up in nnU-Net's `imagesTr` (Step 6). Either set `output_folder` directly
to `$nnUNet_raw/DatasetXXX_YourDatasetName/imagesTr` to skip a separate copy
step, or copy the contents of `imagesTrPSF` over afterward.

## 3. Binarize the label files

`save_stack()` in the generator writes label masks scaled to 16-bit
(`0` / `65535`), not the plain `0`/`1` integer values nnU-Net expects for
class labels. Convert every file in `labelsTr` before using it:

```python
import os
import numpy as np
import tifffile

labels_dir = "/path/to/synthetic_output/labelsTr"

for fname in os.listdir(labels_dir):
    if fname.lower().endswith((".tif", ".tiff")):
        path = os.path.join(labels_dir, fname)
        arr = tifffile.imread(path)
        arr = (arr > 0).astype(np.uint8)
        tifffile.imwrite(path, arr)
```

Run this once over `labelsTr` from Step 1's output — it applies to all three
label files per case (`img_XXX.tif`, `_seg1`, `_seg2`) — before copying it
into the nnU-Net dataset folder in Step 6.

---

## 4. Set up the nnU-Net environment

For full step-by-step detail beyond what's summarized here, see the
[official nnU-Net installation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/getting-started/installation-and-setup.md).

**Make a conda environment (Python 3.10+):**

```bash
conda create --name nnunet python=3.11
conda activate nnunet
```

**Install PyTorch for your hardware:**
Follow the selector at https://pytorch.org/get-started/locally/ and run the
command it gives you.

**Install nnU-Net v2:**

```bash
pip install nnunetv2
```

**Create the three storage locations:**

```bash
mkdir -p nnunet_data/nnUNet_raw
mkdir -p nnunet_data/nnUNet_preprocessed
mkdir -p nnunet_data/nnUNet_results
```

**Set the environment variables** (add these to `~/.bashrc` so they persist
across sessions, then `source ~/.bashrc`):

```bash
export nnUNet_raw="$(pwd)/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/nnunet_data/nnUNet_results"
```

**Verify they're set correctly:**

```bash
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

**Swap in the custom `nnunetv2` package folder:**

Locate where nnU-Net was installed:

```bash
python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))"
```

Navigate to that location, rename the existing folder so you don't lose it,
then copy in the custom one provided in this repo:

```bash
mv nnunetv2 nnunetv2_original
```

Then copy the custom `nnunetv2` folder from this repo into the same location
(this is what adds `MultiChannelPreprocessor`, `MultiChannelLabelManager`,
and `nnUNetTrainerMultiChannelSeg`).

---

## 5. Create the dataset folder in `nnUNet_raw`

```bash
mkdir -p $nnUNet_raw/DatasetXXX_YourDatasetName/imagesTr
mkdir -p $nnUNet_raw/DatasetXXX_YourDatasetName/labelsTr
```

## 6. Copy the prepared data in

Copy the **PSF-convolved images** (Step 2 output) and the **binarized
labels** (Step 3 output) — not the raw Step 1 output:

```bash
cp /path/to/synthetic_output/imagesTrPSF/* $nnUNet_raw/DatasetXXX_YourDatasetName/imagesTr/
cp /path/to/synthetic_output/labelsTr/* $nnUNet_raw/DatasetXXX_YourDatasetName/labelsTr/
```

## 7. Generate the spacing JSONs

Each image and label file needs a spacing JSON sidecar with the same case
ID, e.g. `img_000.json` for both `img_000_0000.tif` (imagesTr) and
`img_000.tif` (labelsTr):

```json
{
  "spacing": [0.108, 0.108, 0.25]
}
```

Use `make_spacing_jsons.py` (included in this repo) to generate these for
the whole dataset in one go — point `--base-dir` at the **dataset root**
(the folder containing `imagesTr/` and `labelsTr/`), not at `imagesTr`
itself; the script handles both subfolders internally:

```bash
python make_spacing_jsons.py \
  --base-dir $nnUNet_raw/DatasetXXX_YourDatasetName \
  --spacing 0.108 0.108 0.25
```

Adjust the spacing values to match your imaging system's actual voxel size
(x y z). The script uses `labelsTr` filenames as the source of truth for
case IDs, so it correctly handles image filenames that happen to end in a
4-digit sequence that isn't actually the nnU-Net channel suffix (e.g.
`v14img_0000.tif`) — it'll print a warning if a case looks ambiguous, so
skim the output for `⚠️` lines and double check any it flags. Files that
already have a spacing JSON are skipped, so it's safe to re-run.

## 8. Generate `dataset.json`

Use `generate_dataset_json_multichannel.py` (included in this repo). It's
config-driven rather than argparse-driven — open it and edit the block near
the top before running:

```python
DATASET_ROOT  = "/path/to/nnUNet_raw/DatasetXXX_YourDatasetName"
DATASET_NAME  = "DatasetXXX_YourDatasetName"
DESCRIPTION   = "Describe this dataset/run here"
CONVERTED_BY  = "Your name"
```

Then run it:

```bash
python generate_dataset_json_multichannel.py
```

For each case in `imagesTr`, it checks that all three label files exist
(`{case}.tif`, `{case}_seg1.tif`, `{case}_seg2.tif`) and skips (with a
warning) any case that's missing one. The resulting `dataset.json` includes
`"num_label_channels": 3` — that's the field the custom
`MultiChannelLabelManager` reads to know it should also pick up the
`_seg1`/`_seg2` sibling files for each case, even though only the base label
path is listed under `"training"`.

## 9. Run initial planning + preprocessing

```bash
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity
```

This creates `nnUNetPlans.json` under
`$nnUNet_preprocessed/DatasetXXX_YourDatasetName/`.

## 10. Patch `nnUNetPlans.json` for the multi-channel setup

```bash
# Change preprocessor_name
sed -i 's/"preprocessor_name": "DefaultPreprocessor"/"preprocessor_name": "MultiChannelPreprocessor"/' \
  $nnUNet_preprocessed/DatasetXXX_YourDatasetName/nnUNetPlans.json

# Change label_manager
sed -i 's/"label_manager": "LabelManager"/"label_manager": "MultiChannelLabelManager"/' \
  $nnUNet_preprocessed/DatasetXXX_YourDatasetName/nnUNetPlans.json
```

## 11. Re-run just preprocessing (not planning)

Delete the old preprocessed data for the config you're using, then
re-preprocess with the patched plans:

```bash
rm -rf $nnUNet_preprocessed/DatasetXXX_YourDatasetName/nnUNetPlans_3d_fullres
nnUNetv2_preprocess -d XXX -c 3d_fullres -np 80
```

`-np` is the number of parallel processes — adjust to your CPU core count.

## 12. Train

```bash
nnUNetv2_train XXX 3d_fullres 0 -tr nnUNetTrainerMultiChannelSeg
```

`0` is the fold — repeat for other folds (`1`–`4`) if you're doing full
cross-validation. Training checkpoints and logs land under
`$nnUNet_results/DatasetXXX_YourDatasetName/`.

---

## Notes

- Steps 9–11 must happen in that order: plan-and-preprocess first (so
  `nnUNetPlans.json` and the dataset fingerprint exist), *then* patch the
  plans file, *then* re-preprocess only the `3d_fullres` configuration — do
  not delete the fingerprint or re-run `nnUNetv2_plan_and_preprocess`, or
  you'll overwrite your edits.
- If you generated data with `long_radius_mean=40` or otherwise changed
  volume dimensions, double check `target_z` and array sizes are consistent
  across the whole dataset before running Step 9's integrity check.
