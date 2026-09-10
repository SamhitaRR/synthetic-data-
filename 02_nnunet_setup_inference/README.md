# nnU-Net Setup and Inference Guide

This guide walks through installing nnU-Net (v2) in a conda environment, installing a pretrained model, preparing images, and running inference.

For full details on Steps 1–7 (installing nnU-Net itself), see the [official nnU-Net GitHub repository](https://github.com/MIC-DKFZ/nnUNet). Steps 8 onward are specific to this workflow and are not covered there.

> **Already have nnU-Net installed?** Skip Steps 1–7 and begin at Step 8.

## 1. Create a Conda Environment

nnU-Net requires Python 3.10 or newer.

```bash
conda create --name nnunet python=3.11
conda activate nnunet
```

## 2. Install PyTorch

Install the version of PyTorch appropriate for your hardware (CPU, CUDA, ROCm, etc.) using the official selector:

https://pytorch.org/get-started/locally/

## 3. Install nnU-Net

```bash
pip install nnunetv2
```

## 4. Create Storage Directories

nnU-Net needs three directories for raw data, preprocessed data, and results:

```bash
mkdir -p nnunet_data/nnUNet_raw
mkdir -p nnunet_data/nnUNet_preprocessed
mkdir -p nnunet_data/nnUNet_results
```

## 5. Set Environment Variables

Point nnU-Net to the directories you just created (use the **absolute path**):

```bash
export nnUNet_raw="/absolute/path/to/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/absolute/path/to/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/absolute/path/to/nnunet_data/nnUNet_results"
```

> **Tip:** Add these lines to your `~/.bashrc` (or equivalent shell config) so they persist across sessions.

## 6. Verify the Environment Variables

Check that each variable was set correctly:

```bash
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

Each command should print the path you set in Step 5.

## 8. Swap in the Custom nnunetv2 Package Folder

Locate where nnU-Net was installed:

```bash
python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))"
```

Navigate to that location, rename the existing `nnunetv2` folder, and copy in the custom `nnunetv2` folder provided to you:

```bash
mv nnunetv2 nnunetv2_original
```

Then copy the provided `nnunetv2` folder into the same location.

## 9. Install the Pretrained Model

The pretrained model zip file is provided to you: [Download model_67_200.zip](https://github.com/SamhitaRR/synthetic-data-/releases/download/model_67_nnunet/model_67_200.zip)

```bash
nnUNetv2_install_pretrained_model_from_zip /path/of/model_67_200.zip
```

After installation, you should see a `Dataset067_PSFbgVacuoles` folder inside your `nnUNet_results` directory.

> **Already have a `Dataset067` folder in your results directory?** Rename it to something else before installing the new one to avoid it being overwritten.

## 10. Prepare Images for Inference

- Images must be in **TIFF** format with a single input channel.
- Each image filename must end in `_0000.tif`.
- Each image needs a corresponding spacing JSON file with the **same name as the image, minus the `_0000` suffix**.

### Option A: Create the spacing JSON manually (single image)

```json
{
  "spacing": [
    0.108,
    0.108,
    0.25
  ]
}
```

Copy this into a text editor and save it as a `.json` file with the appropriate name (matching your image name, minus the `_0000` suffix). Replace the spacing values with your own image's x, y, and z spacing.

### Option B: Generate spacing JSONs in bulk

Use the provided `make_spacing_jsons.py` script:

```bash
python make_spacing_jsons.py --base-dir /home/Samhita/images/ --spacing 0.108 0.108 0.25
```

- `--base-dir`: path to the folder containing your images
- `--spacing`: spacing values in **x y z** order

## 11. Run Inference

```bash
python -m nnunetv2.inference.multi_channel_predictor \
  -i /path/of/input/images/ \
  -o /path/of/output/ \
  -d 067 \
  -c 3d_fullres \
  -f 0 \
  -tr nnUNetTrainerMultiChannelSeg \
  -chk checkpoint_epoch_200.pth
```

**Flags:**
| Flag | Description |
|------|-------------|
| `-i` | Path to input images |
| `-o` | Path to output directory |
| `-d` | Dataset ID (067) |
| `-c` | Configuration (3d_fullres) |
| `-f` | Fold number (0) |
| `-tr` | Trainer class (nnUNetTrainerMultiChannelSeg) |
| `-chk` | Checkpoint file to use |

## 12. Understanding the Output

For each input image, inference produces **three output images**:

| Output | Contents | Notes |
|--------|----------|-------|
| `_seg0` | Predicted cell masks | Keep |
| `_seg1` | — | Can be deleted/ignored |
| `_seg2` | Predicted cell centres | Keep |

`_seg0` (masks) and `_seg2` (centres) can be combined to achieve **instance segmentation**.

## 13. Instance Segmentation

Use the provided `watershed_instances.py` script to combine the `_seg0` and `_seg2` outputs into a single instance segmentation mask, where each cell is assigned a unique integer label.

```bash
python watershed_instances.py \
  --input-dir  /path/of/nnunet/output/ \
  --output-dir /path/of/instance/output/
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--input-dir` | Folder where the nnU-Net outputs are saved |
| `--output-dir` | Folder to save the instance segmentation masks to |
