#!/usr/bin/env python3
import os
import json
import glob

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
DATASET_ROOT  = "/data7/samhitar/nnUNet/nnUNet_raw/Dataset067_PSFbgVacuoles"
DATASET_NAME  = "Dataset067_PSFbgVacuoles"
DESCRIPTION   = "Multi-channel segmentation with 3 independent binary label channels. Synthetic data using orignal steps, vacuole with background intensity) and NA1.4 PSFs."
CONVERTED_BY  = "Samhita"
# ─────────────────────────────────────────────────────────────────────────────

images_dir = os.path.join(DATASET_ROOT, "imagesTr")
labels_dir = os.path.join(DATASET_ROOT, "labelsTr")

image_files = sorted(glob.glob(os.path.join(images_dir, "*_0000.tif")))
if not image_files:
    raise FileNotFoundError(f"No *_0000.tif files found in {images_dir}")

training = []
missing_labels = []

for img_path in image_files:
    basename = os.path.basename(img_path)
    stem = basename.replace("_0000.tif", "")

    lbl_paths = [
        os.path.join(labels_dir, f"{stem}.tif"),
        os.path.join(labels_dir, f"{stem}_seg1.tif"),
        os.path.join(labels_dir, f"{stem}_seg2.tif"),
    ]

    all_exist = True
    for lbl_path in lbl_paths:
        if not os.path.isfile(lbl_path):
            missing_labels.append(lbl_path)
            all_exist = False

    if all_exist:
        training.append({
            "image": f"imagesTr/{stem}_0000.tif",
            "label": f"labelsTr/{stem}.tif"
        })
    else:
        print(f"  ⚠️  Skipping {stem} — missing label files")

if missing_labels:
    print(f"\nMissing label files ({len(missing_labels)}):")
    for p in missing_labels:
        print(f"  {p}")

dataset_json = {
    "channel_names": {"0": "channel0"},
    "labels": {"background": 0, "foreground": 1},
    "num_label_channels": 3,
    "numTraining": len(training),
    "file_ending": ".tif",
    "licence": "",
    "converted_by": CONVERTED_BY,
    "name": DATASET_NAME,
    "reference": "",
    "description": DESCRIPTION,
    "overwrite_image_reader_writer": "Tiff3DIO",
    "training": training
}

out_path = os.path.join(DATASET_ROOT, "dataset.json")
with open(out_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"\n✅  Written {out_path}")
print(f"    {len(training)} training cases")
