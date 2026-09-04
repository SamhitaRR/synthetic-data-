"""
watershed_instances.py
======================
Instance segmentation using distance transform + seeded watershed.

Supports three input modes:

1. Single file pair:
   --cells path/to/cells.tiff --centres path/to/centres.tiff --output path/to/output.tiff

2. Two folders (matching filenames):
   --cells-dir path/to/cells_folder --centres-dir path/to/centres_folder --output-dir path/to/output_folder

3. Single combined folder (e.g. nnU-Net inference output):
   --input-dir path/to/folder --output-dir path/to/output_folder
   Cell masks are matched by files ending in "_seg0" and centre masks by
   files ending in "_seg2" (override with --cells-suffix / --centres-suffix).

Output:  16-bit instance mask(s) where each cell has a unique integer label.

Usage
-----
# Single pair
python watershed_instances.py \
    --cells    path/to/cells.tiff \
    --centres  path/to/centres.tiff \
    --output   path/to/output.tiff

# Two folders
python watershed_instances.py \
    --cells-dir   path/to/cells_folder \
    --centres-dir path/to/centres_folder \
    --output-dir  path/to/output_folder

# Single combined folder (nnU-Net style outputs)
python watershed_instances.py \
    --input-dir  path/to/folder \
    --output-dir path/to/output_folder
"""
import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import tifffile
from scipy.ndimage import label, distance_transform_edt, binary_fill_holes
from skimage.segmentation import watershed

TIFF_EXTENSIONS = (".tif", ".tiff")


def read_tiff(path: str) -> np.ndarray:
    return np.squeeze(tifffile.imread(path))


def save_tiff_16(arr: np.ndarray, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tifffile.imwrite(path, arr.astype(np.uint16), photometric="minisblack")


def is_tiff(filename: str) -> bool:
    return filename.lower().endswith(TIFF_EXTENSIONS)


def strip_extension(filename: str) -> Tuple[str, str]:
    for ext in TIFF_EXTENSIONS:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)], filename[-len(ext):]
    root, ext = os.path.splitext(filename)
    return root, ext


def run_watershed(cells_raw: np.ndarray, centres_raw: np.ndarray) -> np.ndarray:
    """Core watershed instance segmentation logic. Returns a uint16 instance mask."""
    print(f"  Cells shape:   {cells_raw.shape}  dtype: {cells_raw.dtype}  unique: {np.unique(cells_raw)}")
    print(f"  Centres shape: {centres_raw.shape}  dtype: {centres_raw.dtype}  unique: {np.unique(centres_raw)}")

    if cells_raw.shape != centres_raw.shape:
        raise ValueError(f"Shape mismatch: cells {cells_raw.shape} vs centres {centres_raw.shape}")

    cell_mask = binary_fill_holes(cells_raw > 0)  # fill small interior holes
    centre_mask = centres_raw > 0
    print(f"  Cell voxels:   {cell_mask.sum():,}")
    print(f"  Centre voxels: {centre_mask.sum():,}")

    # ---- Label individual centres (seeds) ----
    print("  Labelling centres...")
    labelled_centres, n_labels = label(centre_mask)
    labelled_centres = labelled_centres.astype(np.int32)
    print(f"  {n_labels} centres found")

    if n_labels == 0:
        print("  No centres found — check the centre mask. Skipping.")
        return None

    # ---- Distance transform ----
    print("  Computing distance transform...")
    edt = distance_transform_edt(cell_mask).astype(np.float32)
    edt_inv = edt.max() - edt  # invert so watershed floods from peaks inward

    # ---- Watershed ----
    print("  Running watershed...")
    instances = watershed(edt_inv, markers=labelled_centres, mask=cell_mask)
    instances = instances.astype(np.uint16)
    print(f"  {instances.max()} instances segmented")

    return instances


def process_pair(cells_path: str, centres_path: str, output_path: str) -> None:
    print(f"\nReading cells:   {cells_path}")
    cells_raw = read_tiff(cells_path)
    print(f"Reading centres: {centres_path}")
    centres_raw = read_tiff(centres_path)

    instances = run_watershed(cells_raw, centres_raw)
    if instances is None:
        return

    save_tiff_16(instances, output_path)
    print(f"Saved → {output_path}")


def find_pairs_two_dirs(cells_dir: str, centres_dir: str) -> List[Tuple[str, str, str]]:
    """Match files with identical filenames across two folders."""
    cells_files = {f for f in os.listdir(cells_dir) if is_tiff(f)}
    centres_files = {f for f in os.listdir(centres_dir) if is_tiff(f)}
    common = sorted(cells_files & centres_files)

    missing_centres = sorted(cells_files - centres_files)
    missing_cells = sorted(centres_files - cells_files)
    if missing_centres:
        print(f"Warning: no matching centre file for: {missing_centres}")
    if missing_cells:
        print(f"Warning: no matching cell file for: {missing_cells}")

    pairs = []
    for filename in common:
        base, _ = strip_extension(filename)
        out_name = f"{base}_instances.tiff"
        pairs.append((
            os.path.join(cells_dir, filename),
            os.path.join(centres_dir, filename),
            out_name,
        ))
    return pairs


def find_pairs_single_dir(input_dir: str, cells_suffix: str, centres_suffix: str) -> List[Tuple[str, str, str]]:
    """Match cell/centre files within one folder based on filename suffixes."""
    files = [f for f in os.listdir(input_dir) if is_tiff(f)]

    cells_by_base = {}
    centres_by_base = {}
    for filename in files:
        base, ext = strip_extension(filename)
        if base.endswith(cells_suffix):
            cells_by_base[base[: -len(cells_suffix)]] = filename
        elif base.endswith(centres_suffix):
            centres_by_base[base[: -len(centres_suffix)]] = filename

    common_bases = sorted(set(cells_by_base) & set(centres_by_base))
    missing_centres = sorted(set(cells_by_base) - set(centres_by_base))
    missing_cells = sorted(set(centres_by_base) - set(cells_by_base))
    if missing_centres:
        print(f"Warning: no matching '{centres_suffix}' file for base(s): {missing_centres}")
    if missing_cells:
        print(f"Warning: no matching '{cells_suffix}' file for base(s): {missing_cells}")

    pairs = []
    for base in common_bases:
        out_name = f"{base}_instances.tiff"
        pairs.append((
            os.path.join(input_dir, cells_by_base[base]),
            os.path.join(input_dir, centres_by_base[base]),
            out_name,
        ))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Watershed instance segmentation from cell and centre masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode 1: single file pair
    parser.add_argument("--cells", help="Path to a single binary cell mask TIFF")
    parser.add_argument("--centres", help="Path to a single binary centre mask TIFF")
    parser.add_argument("--output", help="Path to output instance TIFF (used with --cells/--centres)")

    # Mode 2: two folders
    parser.add_argument("--cells-dir", help="Folder of binary cell mask TIFFs")
    parser.add_argument("--centres-dir", help="Folder of binary centre mask TIFFs (matched by filename)")

    # Mode 3: single combined folder
    parser.add_argument("--input-dir", help="Single folder containing both cell and centre masks")
    parser.add_argument("--cells-suffix", default="_seg0",
                         help="Filename suffix identifying cell masks in --input-dir (default: _seg0)")
    parser.add_argument("--centres-suffix", default="_seg2",
                         help="Filename suffix identifying centre masks in --input-dir (default: _seg2)")

    # Shared output folder for modes 2 and 3
    parser.add_argument("--output-dir", help="Folder to write output instance TIFFs to (used with folder modes)")

    args = parser.parse_args()

    single_pair_mode = args.cells or args.centres or args.output
    two_dir_mode = args.cells_dir or args.centres_dir
    single_dir_mode = args.input_dir

    modes_selected = sum(bool(m) for m in (single_pair_mode, two_dir_mode, single_dir_mode))
    if modes_selected == 0:
        parser.error(
            "You must specify one input mode: "
            "(--cells & --centres & --output), "
            "(--cells-dir & --centres-dir & --output-dir), or "
            "(--input-dir & --output-dir)."
        )
    if modes_selected > 1:
        parser.error("Please use only one input mode at a time — do not mix file, two-folder, and single-folder options.")

    if single_pair_mode:
        if not (args.cells and args.centres and args.output):
            parser.error("Single file mode requires all of: --cells, --centres, --output")
        process_pair(args.cells, args.centres, args.output)

    elif two_dir_mode:
        if not (args.cells_dir and args.centres_dir and args.output_dir):
            parser.error("Two-folder mode requires all of: --cells-dir, --centres-dir, --output-dir")
        pairs = find_pairs_two_dirs(args.cells_dir, args.centres_dir)
        if not pairs:
            print("No matching cell/centre pairs found.")
            return
        print(f"Found {len(pairs)} matching pair(s).")
        for cells_path, centres_path, out_name in pairs:
            process_pair(cells_path, centres_path, os.path.join(args.output_dir, out_name))

    elif single_dir_mode:
        if not args.output_dir:
            parser.error("Single-folder mode requires --output-dir")
        pairs = find_pairs_single_dir(args.input_dir, args.cells_suffix, args.centres_suffix)
        if not pairs:
            print(f"No matching '{args.cells_suffix}'/'{args.centres_suffix}' pairs found in {args.input_dir}.")
            return
        print(f"Found {len(pairs)} matching pair(s).")
        for cells_path, centres_path, out_name in pairs:
            process_pair(cells_path, centres_path, os.path.join(args.output_dir, out_name))


if __name__ == "__main__":
    main()
