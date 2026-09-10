"""
Synthetic 3D fluorescence microscopy data generator - "dimmer vacuoles"
variant.

Same as synthetic_cell_data_overlaps_centres.py, except vacuole intensity is
handled the way synthetic_fl_bigcentres_overlaps_samechannel_22jan.R does it:
instead of proportionally dimming each vacuole voxel relative to its own
cell's intensity, every vacuole location (across all cells) is accumulated
into a single global mask, and only after every cell's fluorescence has been
drawn are those voxels overwritten with fresh background-level noise
(~U(95, 115)) - i.e. a vacuole looks like background rather than "a dimmer
version of its cell".

Cells (ellipsoids) are placed randomly and are allowed to overlap each other
directly (no plane-splitting, no gap band). For each generated volume, three
label stacks are written:

    imagesTr/img_XXX.tif        - synthetic fluorescence image
    labelsTr/img_XXX.tif        - plain foreground/background mask (no gaps)
    labelsTr/img_XXX_seg1.tif   - union of all cell-cell overlap regions
    labelsTr/img_XXX_seg2.tif   - union of shrunken per-cell "centre" masks

Requires: numpy, scipy, tifffile
"""

import os
import time
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
import tifffile
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rotation_matrix(u):
    """Build an orthonormal 3x3 basis with `u` (normalized) as the first
    column, matching the R version's get_rotation_matrix()."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    ref = np.array([0.0, 1.0, 0.0]) if abs(u[0]) > 0.1 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, ref)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    return np.column_stack([u, v, w])  # columns: u, v, w


def _bbox(center, max_r, dims):
    """Integer bounding box (inclusive-exclusive, 0-indexed) clipped to dims."""
    width, height, depth = dims
    xmin = max(0, int(np.floor(center[0] - max_r)))
    xmax = min(width, int(np.ceil(center[0] + max_r)) + 1)
    ymin = max(0, int(np.floor(center[1] - max_r)))
    ymax = min(height, int(np.ceil(center[1] + max_r)) + 1)
    zmin = max(0, int(np.floor(center[2] - max_r)))
    zmax = min(depth, int(np.ceil(center[2] + max_r)) + 1)
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        return None
    return xmin, xmax, ymin, ymax, zmin, zmax


def create_ellipsoid_mask(center, long_radius, short_radius, orient, dims):
    """Return (xmin, xmax, ymin, ymax, zmin, zmax, submask) for the ellipsoid,
    or None if it falls entirely outside the volume. `submask` is a boolean
    array of shape (xmax-xmin, ymax-ymin, zmax-zmin)."""
    box = _bbox(center, long_radius, dims)
    if box is None:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = box

    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    zs = np.arange(zmin, zmax)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=0).astype(float)

    p_minus_c = coords - np.asarray(center).reshape(3, 1)
    R = get_rotation_matrix(orient)
    local = R.T @ p_minus_c

    inside = ((local[0, :] / long_radius) ** 2 +
              (local[1, :] / short_radius) ** 2 +
              (local[2, :] / short_radius) ** 2) <= 1.0

    submask = inside.reshape(gx.shape)
    if not submask.any():
        return None
    return xmin, xmax, ymin, ymax, zmin, zmax, submask


def create_sphere_mask(center, radius, dims):
    """Same idea as create_ellipsoid_mask but for a sphere."""
    box = _bbox(center, radius, dims)
    if box is None:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = box

    xs = np.arange(xmin, xmax)
    ys = np.arange(ymin, ymax)
    zs = np.arange(zmin, zmax)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    dx = gx - center[0]
    dy = gy - center[1]
    dz = gz - center[2]
    dist2 = dx * dx + dy * dy + dz * dz
    submask = dist2 <= radius ** 2
    if not submask.any():
        return None
    return xmin, xmax, ymin, ymax, zmin, zmax, submask


def save_stack(arr01, path, scale_16bit=True):
    """Write a (width, height, depth) array as a multi-page 16-bit TIFF,
    one page per z-slice, mirroring R's writeTIFF(..., bits.per.sample=16)
    behaviour on [0,1]-scaled data."""
    data = np.clip(arr01.astype(np.float64), 0.0, 1.0)
    if scale_16bit:
        data = np.round(data * 65535.0)
    data = data.astype(np.uint16)
    # (X, Y, Z) -> (Z, Y, X) for a conventional page/row/col TIFF layout
    data = np.transpose(data, (2, 1, 0))
    tifffile.imwrite(path, data)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_fluorescence(
    width=512, height=512, depth=123,
    n_cells_target=100, max_attempts=1000,
    long_radius_mean=15, long_radius_sd=3,
    short_ratio_min=0.6, short_ratio_max=0.9,
    overlap_threshold=0.3,
    target_z=53,
    erode_steps=4, erode_size=3,
    # Fluorescence params (16-bit units)
    MEANacross_16bit=200,
    SDacross_16bit=30,
    SDwithin_16bit=30,
    minthreshold=110,
    vac_min_frac=0.05, vac_max_frac=0.5,
    blur_sigma=1.0,
    # z falloff control
    z_falloff_fraction_dim=0.3,
    # "centre" sub-ellipsoid scale (fraction of the cell's long/short radius)
    center_scale=0.75,
    rescale_to_target=False,
    filenumber=0,
    out_dir=".",
):
    rng = np.random.default_rng()
    dims = (width, height, depth)

    M_seg = np.zeros(dims, dtype=np.int32)
    M_fluoLab = np.zeros(dims, dtype=np.int32)
    M_overlap = np.zeros(dims, dtype=bool)
    M_center = np.zeros(dims, dtype=bool)

    L_centers, L_long, L_short, L_orient = [], [], [], []

    ncell = 0
    attempts = 0

    while ncell < n_cells_target and attempts < max_attempts:
        attempts += 1

        center = np.array([
            rng.uniform(0, width),
            rng.uniform(0, height),
            rng.uniform(0, depth),
        ])
        long_radius = abs(rng.normal(long_radius_mean, long_radius_sd))
        short_ratio = rng.uniform(short_ratio_min, short_ratio_max)
        short_radius = long_radius * short_ratio
        orient = rng.normal(size=3)
        orient = orient / np.linalg.norm(orient)

        result = create_ellipsoid_mask(center, long_radius, short_radius, orient, dims)
        if result is None:
            continue
        xmin, xmax, ymin, ymax, zmin, zmax, submask = result
        new_pixels = submask.sum()
        if new_pixels == 0:
            continue

        seg_region = M_seg[xmin:xmax, ymin:ymax, zmin:zmax]
        overlap_local = submask & (seg_region > 0)
        overlap_pixels = int(overlap_local.sum())
        if overlap_pixels / new_pixels > overlap_threshold:
            continue

        overlapping_labels = np.unique(seg_region[overlap_local])
        overlapping_labels = overlapping_labels[overlapping_labels > 0]

        discard = False
        for lab in overlapping_labels:
            overlap_this = int(np.sum((seg_region == lab) & overlap_local))
            size_this = int(np.sum(M_seg == lab))  # full-volume count, as in the R version
            if size_this > 0 and overlap_this / size_this > overlap_threshold:
                discard = True
                break
        if discard:
            continue

        temp_label = ncell + 1

        # record overlap BEFORE overwriting - persistent union across the run
        if overlap_local.any():
            M_overlap[xmin:xmax, ymin:ymax, zmin:zmax][overlap_local] = True

        # new cell occupies its full extent; shared voxels are overwritten
        # (no plane split, no gap)
        seg_region[submask] = temp_label
        M_fluoLab[xmin:xmax, ymin:ymax, zmin:zmax][submask] = temp_label

        # "centre" sub-ellipsoid for this cell
        centre_result = create_ellipsoid_mask(
            center, long_radius * center_scale, short_radius * center_scale,
            orient, dims,
        )
        if centre_result is not None:
            cxmin, cxmax, cymin, cymax, czmin, czmax, csubmask = centre_result
            M_center[cxmin:cxmax, cymin:cymax, czmin:czmax][csubmask] = True

        ncell += 1
        L_centers.append(center)
        L_long.append(long_radius)
        L_short.append(short_radius)
        L_orient.append(orient)

    if ncell < n_cells_target:
        print(f"WARNING: only {ncell} cells added after {max_attempts} attempts.")

    # ---- Morphology for vacuole placement ----
    struct = np.ones((erode_size, erode_size, erode_size), dtype=bool)
    M4 = binary_erosion(M_seg > 0, structure=struct, iterations=erode_steps, border_value=0)

    # ---- Fluorescence synthesis (16-bit absolute units) ----
    Mfluo = rng.uniform(95, 115, size=dims)

    # Global vacuole mask accumulator - vacuole voxels from every cell are
    # collected here and overwritten with background noise at the end,
    # rather than being dimmed in-place per cell.
    M_vacuoles = np.zeros(dims, dtype=bool)

    for icell in range(1, ncell + 1):
        print(f"{icell}th cell")
        cell_mask = (M_fluoLab == icell)
        npix = int(cell_mask.sum())
        if npix == 0:
            continue

        MEANwithin = rng.normal(MEANacross_16bit, SDacross_16bit)
        MEANwithin = max(MEANwithin, minthreshold)

        vals = rng.normal(MEANwithin, SDwithin_16bit, size=npix)
        vals = np.clip(vals, 0, 65535)
        Mfluo[cell_mask] = vals

        # --- vacuole dimming ---
        vac_in_cell = np.zeros(dims, dtype=bool)
        inner_cell = M4 & cell_mask
        if inner_cell.any():
            inner_idx = np.argwhere(inner_cell)
            vac_center = inner_idx[rng.integers(0, len(inner_idx))]

            vac_frac = rng.uniform(vac_min_frac, vac_max_frac)
            vac_vol = vac_frac * npix
            vac_radius = (vac_vol * 3 / (4 * np.pi)) ** (1 / 3)

            sph = create_sphere_mask(vac_center, vac_radius, dims)
            if sph is not None:
                sxmin, sxmax, symin, symax, szmin, szmax, ssub = sph
                local_cell = cell_mask[sxmin:sxmax, symin:symax, szmin:szmax]
                local_vac = ssub & local_cell
                vac_in_cell[sxmin:sxmax, symin:symax, szmin:szmax] = local_vac

                # accumulate into the global mask instead of dimming now
                M_vacuoles[sxmin:sxmax, symin:symax, szmin:szmax] |= local_vac

        # --- add spots, avoiding any vacuole placed so far (global, not just
        # this cell's own) ---
        spots_mask = cell_mask & ~M_vacuoles
        for _ in range(35):
            brightness = rng.uniform(0.75, 1.25)
            spot_idx_pool = np.argwhere(spots_mask)
            if len(spot_idx_pool) == 0:
                continue
            spot_center = spot_idx_pool[rng.integers(0, len(spot_idx_pool))]

            spot_frac = rng.uniform(0.03, 0.05)
            spot_vol = spot_frac * npix
            spot_radius = (spot_vol * 3 / (4 * np.pi)) ** (1 / 3)

            sph = create_sphere_mask(spot_center, spot_radius, dims)
            if sph is None:
                continue
            sxmin, sxmax, symin, symax, szmin, szmax, ssub = sph
            local_spots = spots_mask[sxmin:sxmax, symin:symax, szmin:szmax]
            local_mask = ssub & local_spots
            if local_mask.any():
                region = Mfluo[sxmin:sxmax, symin:symax, szmin:szmax]
                region[local_mask] *= brightness

        # --- per-cell z falloff (focus attenuation) ---
        cell_vox_idx = np.argwhere(cell_mask)
        z_coords = cell_vox_idx[:, 2]
        z0 = z_coords.mean()
        dz = np.abs(z_coords - z0)
        dz_max = dz.max()

        if dz_max > 0:
            target_min_scale = 1 - z_falloff_fraction_dim
            sigma = dz_max / np.sqrt(-2 * np.log(target_min_scale))
            scale_vec = np.exp(-(dz ** 2) / (2 * sigma ** 2))
        else:
            scale_vec = np.ones(len(z_coords))

        Mfluo[cell_vox_idx[:, 0], cell_vox_idx[:, 1], cell_vox_idx[:, 2]] *= scale_vec

    # ---- Vacuole intensity overwrite (background-like) ----
    n_vac = int(M_vacuoles.sum())
    if n_vac > 0:
        Mfluo[M_vacuoles] = rng.uniform(95, 115, size=n_vac)

    # ---- Blur (PSF-ish) ----
    Mfluo = gaussian_filter(Mfluo, sigma=blur_sigma)

    # ---- Optional global rescale so average cell intensity hits MEANacross_16bit ----
    if rescale_to_target:
        cell_pixels = M_fluoLab > 0
        if cell_pixels.any():
            cur_mean = Mfluo[cell_pixels].mean()
            if np.isfinite(cur_mean) and cur_mean > 0:
                Mfluo = Mfluo * (MEANacross_16bit / cur_mean)
                Mfluo = np.clip(Mfluo, 0, 65535)

    # adding noise: -10% to +10% of its value
    Mfluo = Mfluo + rng.uniform(-0.1, 0.1, size=Mfluo.shape) * Mfluo

    # --- Downscale Z dimension ---
    original_z = Mfluo.shape[2]
    z_indices = np.round(np.linspace(0, original_z - 1, target_z)).astype(int)

    M_seg_down = M_seg[:, :, z_indices]
    M_overlap_down = M_overlap[:, :, z_indices]
    M_center_down = M_center[:, :, z_indices]
    Mfluo_down = Mfluo[:, :, z_indices]

    # ---- Save TIFFs ----
    images_dir = os.path.join(out_dir, "imagesTr")
    labels_dir = os.path.join(out_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    base_name = f"img_{filenumber:03d}"

    # Label 1: plain foreground/background, no gaps
    gt = (M_seg_down > 0).astype(np.float64)
    save_stack(gt, os.path.join(labels_dir, f"{base_name}.tif"))

    # Label 2 (_seg1): union of all overlap regions
    ov = M_overlap_down.astype(np.float64)
    save_stack(ov, os.path.join(labels_dir, f"{base_name}_seg1.tif"))

    # Label 3 (_seg2): union of shrunken per-cell "centre" masks
    ce = M_center_down.astype(np.float64)
    save_stack(ce, os.path.join(labels_dir, f"{base_name}_seg2.tif"))

    # Synthetic fluorescence (normalized 0-1)
    fluo_norm = np.clip(Mfluo_down / 65535.0, 0, 1)
    save_stack(fluo_norm, os.path.join(images_dir, f"{base_name}_0000.tif"))

    print("images saved")

    return {
        "seg_plain": gt,
        "seg_overlap": ov,
        "seg_centre": ce,
        "fluorescence_16bit": Mfluo,
        "labels_segmentation": M_seg,
        "labels_fluo": M_fluoLab,
        "centers": L_centers,
        "long_radii": L_long,
        "short_radii": L_short,
        "orientations": L_orient,
    }


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------

def run_iteration(params):
    n_cells, mean_across, filenumber = params
    # different seed per process, matching the R script's per-process seeding
    seed = int(time.time() * 1000) + os.getpid()
    np.random.seed(seed % (2 ** 32 - 1))

    return generate_synthetic_fluorescence(
        width=512, height=512,
        n_cells_target=n_cells,
        depth=123,
        target_z=53,
        long_radius_mean=30,
        overlap_threshold=0.1,
        MEANacross_16bit=mean_across,
        minthreshold=130,
        blur_sigma=3,
        SDacross_16bit=250,
        SDwithin_16bit=180,
        z_falloff_fraction_dim=0.3,
        center_scale=0.7,
        rescale_to_target=False,
        filenumber=filenumber,
    )


def main():
    n_cells_values = [16, 18, 20, 22, 24, 26, 28, 30 ,32, 34, 36, 38]
    mean_across_values = list(range(300, 650, 25))
    long_radius_mean_values = [20, 30, 40]
    
    #n_cells_values = [16]
    #mean_across_values = [400]

    param_grid = [
        (n_cells, mean_across)
        for n_cells in n_cells_values
        for mean_across in mean_across_values
        for long_radius_mean in long_radius_mean_values
    ]
    start_filenumber = 0000
    params = [
        (n_cells, mean_across, start_filenumber + i)
        for i, (n_cells, mean_across) in enumerate(param_grid)
    ]

    n_workers = 100
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(run_iteration, params))


if __name__ == "__main__":
    main()
