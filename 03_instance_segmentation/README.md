# Instance Segmentation

This step converts nnU-Net's raw output (cell masks + cell centres) into a single
instance segmentation mask, where each cell is assigned its own unique integer label.

## Requirements

```bash
pip install numpy scipy tifffile scikit-image
```

## Background

nnU-Net inference (see [`02_nnunet_setup_inference`](../02_nnunet_setup_inference/README.md))
produces three output images per input image:

| Output | Contents | Notes |
|--------|----------|-------|
| `_seg0` | Predicted cell masks | Keep |
| `_seg1` | — | Can be deleted/ignored |
| `_seg2` | Predicted cell centres | Keep |

`_seg0` (masks) and `_seg2` (centres) are combined here to achieve **instance
segmentation**, using a distance-transform + seeded watershed approach: each
labelled centre acts as a seed, and the watershed floods outward through the
cell mask until it hits another cell's territory or the mask boundary.

## Usage

`watershed_instances.py` supports three input modes:

### Mode 1: Single file pair

```bash
python watershed_instances.py \
    --cells    path/to/cells.tiff \
    --centres  path/to/centres.tiff \
    --output   path/to/output_instances.tiff
```

### Mode 2: Two separate folders

If your cell masks and centre masks live in two different folders, with matching
filenames in each:

```bash
python watershed_instances.py \
    --cells-dir   path/to/cells_folder \
    --centres-dir path/to/centres_folder \
    --output-dir  path/to/output_folder
```

### Mode 3: One combined folder (e.g. straight from nnU-Net inference output)

```bash
python watershed_instances.py \
    --input-dir  /path/of/nnunet/output/ \
    --output-dir /path/of/instance/output/
```

This uses the default suffixes `_seg0` (cells) and `_seg2` (centres). Override them
if your files use different suffixes:

```bash
python watershed_instances.py \
    --input-dir      path/to/folder \
    --output-dir     path/to/output_folder \
    --cells-suffix   _seg0 \
    --centres-suffix _seg2
```

**Notes:**
- Only use one mode's flags at a time — mixing them (e.g. `--cells` with `--input-dir`) will throw an error.
- Any files without a matching pair are skipped, with a warning printed to the console.
- Outputs are named `<basename>_instances.tiff`, saved as 16-bit TIFFs.
- Accepts both `.tif` and `.tiff` extensions.
