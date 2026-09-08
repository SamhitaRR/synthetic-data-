#!/usr/bin/env python3
"""
make_spacing_jsons_flat.py
---------------------------
Generates a spacing JSON sidecar file for every TIFF image in a single
input folder (not an nnU-Net imagesTr/labelsTr dataset layout).

If a filename ends in a 4-digit channel suffix (e.g. "_0000"), it is
stripped so the JSON is named to match nnU-Net's expected pairing
(e.g. "sample01_0000.tif" -> "sample01.json"). Filenames without a
4-digit suffix are used as-is.

Usage:
    python make_spacing_jsons_flat.py \\
        --input-dir /path/to/images/ \\
        --spacing 0.108 0.108 0.25
"""

import os
import re
import json
import argparse

TIFF_EXTENSIONS = (".tif", ".tiff")
SUFFIX_PATTERN = re.compile(r"_(\d{4})$")


def resolve_case_id(base_name: str) -> str:
    """Strip a trailing 4-digit channel suffix (e.g. _0000) if present."""
    m = SUFFIX_PATTERN.search(base_name)
    if m:
        return base_name[: -5]
    return base_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate spacing JSON files for every image in a flat input folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                         help="Path to the folder containing your TIFF images")
    parser.add_argument("--spacing", nargs=3, type=float, required=True,
                         metavar=("X", "Y", "Z"),
                         help="Spacing to write into each JSON, e.g. --spacing 0.108 0.108 0.25")
    args = parser.parse_args()

    input_dir = args.input_dir
    spacing = list(args.spacing)

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(TIFF_EXTENSIONS))
    if not files:
        print(f"No TIFF images found in {input_dir}")
        return

    for file in files:
        base_name = os.path.splitext(file)[0]
        case_id = resolve_case_id(base_name)

        json_path = os.path.join(input_dir, f"{case_id}.json")
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"spacing": spacing}, f, indent=4)
            print(f"✅ Created {json_path}")
        else:
            print(f"⏩ Skipped (already exists): {json_path}")


if __name__ == "__main__":
    main()
