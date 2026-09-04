#!/usr/bin/env python3
"""
make_spacing_jsons.py
----------------------
Generates per-case spacing JSON sidecar files for an nnU-Net dataset
(imagesTr and labelsTr), correctly handling the ambiguity where an
image's case ID may itself end in a 4-digit number (e.g. "v14img_0000")
that looks like — but is NOT — the nnU-Net channel suffix.

How it resolves the ambiguity:
  labelsTr filenames never carry a channel suffix, so they are treated
  as the ground truth for what a "case ID" actually is.

  For each imagesTr file:
    1. If its full base name (no extension) matches a label exactly,
       that's the case ID as-is (no stripping).
    2. Else, if it ends in "_dddd" and stripping that suffix matches
       a label, the suffix is a real nnU-Net channel index -> strip it.
    3. Else, it's ambiguous -> the script defaults to stripping but
       prints a warning so you can double check that case by hand.

Usage:
    python make_spacing_jsons.py \\
        --base-dir /data7/samhitar/nnUNet/nnUNet_raw/Dataset049_multichannelblur \\
        --spacing 0.108 0.108 0.25

If --base-dir / --spacing are omitted, the defaults below are used.
"""

import os
import re
import json
import argparse

DEFAULT_BASE_DIR = "/data7/samhitar/nnUNet/nnUNet_raw/Dataset067_PSFbgVacuoles/"
DEFAULT_SPACING   = [0.108, 0.108, 0.25]

SUFFIX_PATTERN = re.compile(r"_(\d{4})$")


def collect_label_stems(labels_dir: str) -> set:
    stems = set()
    if os.path.exists(labels_dir):
        for f in os.listdir(labels_dir):
            if f.lower().endswith((".tif", ".tiff")):
                stems.add(os.path.splitext(f)[0])
    return stems


def resolve_case_id(base_name: str, label_stems: set) -> str:
    # Case A: base name already matches a label exactly -> keep as-is
    if base_name in label_stems:
        return base_name

    # Case B: ends in _dddd -- only strip if the stripped version matches a label
    m = SUFFIX_PATTERN.search(base_name)
    if m:
        stripped = base_name[: -5]
        if stripped in label_stems:
            return stripped
        print(
            f"⚠️  '{base_name}' ends in a 4-digit suffix but neither "
            f"'{base_name}' nor '{stripped}' matches a label file. "
            f"Defaulting to stripped form '{stripped}' — please verify this case."
        )
        return stripped

    # Case C: no suffix at all
    return base_name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Path to the Dataset0XX_name folder containing imagesTr/labelsTr")
    parser.add_argument("--spacing", nargs=3, type=float, default=DEFAULT_SPACING,
                        metavar=("X", "Y", "Z"),
                        help="Spacing to write into each JSON, e.g. --spacing 0.108 0.108 0.25")
    args = parser.parse_args()

    base_dir = args.base_dir
    spacing = list(args.spacing)

    images_dir = os.path.join(base_dir, "imagesTr")
    labels_dir = os.path.join(base_dir, "labelsTr")

    label_stems = collect_label_stems(labels_dir)
    if not label_stems:
        print(f"⚠️  No label files found in {labels_dir} — case-ID resolution for "
              f"imagesTr will fall back to naive suffix stripping / warnings for every ambiguous file.")

    for folder in ["imagesTr", "labelsTr"]:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            print(f"Folder not found, skipping: {path}")
            continue

        for file in sorted(os.listdir(path)):
            if not file.lower().endswith((".tif", ".tiff")):
                continue

            base_name = os.path.splitext(file)[0]

            if folder == "imagesTr":
                case_id = resolve_case_id(base_name, label_stems)
            else:
                # labelsTr file names are already the case ID, no suffix to strip
                case_id = base_name

            json_path = os.path.join(path, f"{case_id}.json")
            if not os.path.exists(json_path):
                with open(json_path, "w") as f:
                    json.dump({"spacing": spacing}, f, indent=4)
                print(f"✅ Created {json_path}")
            else:
                print(f"⏩ Skipped (already exists): {json_path}")


if __name__ == "__main__":
    main()
