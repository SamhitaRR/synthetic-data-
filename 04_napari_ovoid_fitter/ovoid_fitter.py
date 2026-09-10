"""
Napari plugin widget: fit 3D ovoids (ellipsoids that can be asymmetric
along one axis, i.e. egg-shaped) to user-clicked boundary points, then
interactively adjust their center / radii / rotation. Setting the two
"long axis" radii equal gives a plain symmetric ellipsoid. Each ovoid is
rasterized into a Labels layer as its own label id.

Quick start (no packaging needed to try it locally):

    import napari
    from ellipsoid_fitter import EllipsoidFitterWidget

    viewer = napari.Viewer(ndisplay=3)
    viewer.open("my_volume.tif")          # or viewer.add_image(my_array)
    widget = EllipsoidFitterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Ovoid Fitter")
    napari.run()

Workflow:
  1. Pick the image layer you're annotating from the dropdown.
  2. Click "Start New Ovoid" -> a Points layer ("boundary points") is
     created/activated in add mode. Switch the viewer to 3D display
     (Ctrl+Y or the ndisplay button) and click on the object's surface
     several times (>= 9 points recommended) to lay down boundary points.
     Use "Switch to X-Z View" to add more points from an orthogonal
     scrolling direction without losing what you've already placed.
  3. Click "Fit Ovoid From Points". The points are fit to an ovoid, added
     to the list on the left, and rasterized into the "Ellipsoid Labels"
     layer.
  4. Select an entry in the list to load its parameters into the sliders
     below. Dragging a slider live-updates that ovoid's voxels in the
     Labels layer. The "long axis radius (+ end)" and "(- end)" controls
     are what make it egg-shaped -- set them equal for a symmetric
     ellipsoid.
  5. "Delete Selected" removes an ovoid (clears its voxels too).

Notes / things you'll likely want to adapt:
  - The exact napari mouse behavior for placing points on a volume surface
    in 3D view has changed across napari versions -- test against the
    version you're pinned to.
  - The ovoid fit uses a PCA-seeded nonlinear least squares solve; this
    matters a lot for convergence (a naive initial guess reliably lands
    in a bad local minimum for asymmetric shapes).
  - Rasterization here is a straightforward per-shape bounding-box loop;
    fine for typical cell/nucleus-sized objects, but for very large shapes
    in huge volumes you may want to downsample or use a coarser mask +
    upsample.
  - Overlap between shapes: rebuilt fresh from all ovoids on every change,
    in ascending id order, so a higher id always wins an overlap -- and
    deleting an ovoid lets anything it was covering reappear automatically.
    Adjust `_rebuild_labels` if you want a different conflict policy.
"""

from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from napari.utils.notifications import show_error, show_info
from napari.layers import Image, Points

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QFileDialog,
)
from qtpy.QtCore import Qt


# --------------------------------------------------------------------------
# Ovoid fitting math (an ellipsoid whose long axis can have a different
# radius on each end -- reduces exactly to an ellipsoid when r1_pos == r1_neg)
# --------------------------------------------------------------------------

def fit_ovoid(points: np.ndarray):
    """Best-fit ovoid to 3D points via bounded nonlinear least squares.

    The shape is: pick a local frame (rotation `rot`); along local axis 0,
    use radius `r1_pos` on the positive side and `r1_neg` on the negative
    side (this is what makes it egg-shaped instead of symmetric); axes 1
    and 2 use single radii `r2`, `r3` on both sides. Setting r1_pos ==
    r1_neg recovers a plain ellipsoid.

    Uses PCA on the point cloud to seed a sensible initial rotation and
    per-axis radius/asymmetry estimate -- without this, the optimizer
    reliably lands in a bad local minimum (axes swapped/misaligned).

    points: (N, 3) array of boundary point coordinates.

    Returns
    -------
    center : (3,) array
    radii  : (4,) array -- (r1_pos, r1_neg, r2, r3)
    rot    : (3, 3) rotation matrix, columns are the ovoid's principal axes
             expressed in the original coordinate frame
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 9:
        raise ValueError("Need at least 9 points to fit an ovoid "
                          f"(got {points.shape[0]}).")

    centroid = points.mean(axis=0)
    p = points - centroid
    extent = np.max(np.linalg.norm(p, axis=1))
    if extent < 1e-6:
        raise ValueError("Points are all on top of each other -- click "
                          "boundary points spread around the object.")

    # PCA-seeded initial rotation and radii -- critical for convergence
    cov = (p.T @ p) / max(len(p) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]  # largest variance first -> axis 0
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1  # keep it a proper rotation (det = +1)
    rotvec0 = Rotation.from_rotvec(Rotation.from_matrix(evecs).as_rotvec()).as_rotvec()

    local0 = p @ evecs
    u0 = local0[:, 0]
    r1p0 = max(u0[u0 >= 0].max() if np.any(u0 >= 0) else 1.0, 1e-2)
    r1n0 = max(-u0[u0 < 0].min() if np.any(u0 < 0) else 1.0, 1e-2)
    r20 = max(np.abs(local0[:, 1]).max(), 1e-2)
    r30 = max(np.abs(local0[:, 2]).max(), 1e-2)

    x0 = np.concatenate([np.zeros(3), np.log([r1p0, r1n0, r20, r30]), rotvec0])

    lo = np.concatenate([
        -2 * extent * np.ones(3),
        np.log(1e-3 * extent) * np.ones(4),
        -4 * np.ones(3),
    ])
    hi = np.concatenate([
        2 * extent * np.ones(3),
        np.log(3 * extent) * np.ones(4),
        4 * np.ones(3),
    ])
    x0 = np.clip(x0, lo, hi)

    def residuals(params):
        c = params[0:3]
        r1p, r1n, r2, r3 = np.exp(params[3:7])
        rot = Rotation.from_rotvec(params[7:10]).as_matrix()
        local = (p - c) @ rot
        u, v, w = local[:, 0], local[:, 1], local[:, 2]
        a = np.where(u >= 0, r1p, r1n)
        return (u / a) ** 2 + (v / r2) ** 2 + (w / r3) ** 2 - 1.0

    result = least_squares(residuals, x0, method="trf", bounds=(lo, hi), max_nfev=20000)

    center = result.x[0:3] + centroid
    radii = np.exp(result.x[3:7])  # (r1_pos, r1_neg, r2, r3)
    rot = Rotation.from_rotvec(result.x[7:10]).as_matrix()
    return center, radii, rot


def ovoid_base_radius(direction, radii):
    """Radius of the plain ovoid surface along unit vector(s) `direction`
    (expressed in the ovoid's local/rotated frame). Vectorized over the
    leading dimensions of `direction`.
    """
    r1p, r1n, r2, r3 = radii
    d0, d1, d2 = direction[..., 0], direction[..., 1], direction[..., 2]
    a = np.where(d0 >= 0, r1p, r1n)
    denom = np.sqrt((d0 / a) ** 2 + (d1 / r2) ** 2 + (d2 / r3) ** 2)
    denom = np.maximum(denom, 1e-12)
    return 1.0 / denom


def ovoid_total_radius(direction, radii, bumps):
    """Base ovoid radius plus any local Gaussian "pull" bumps.

    Each bump is a dict with:
      dir   : (3,) unit vector in the ovoid's local frame -- where it was pulled
      delta : signed extra radius at the bump center (+ pulls out, - pushes in)
      sigma : angular width in radians (how localized the pull is)
    """
    r = ovoid_base_radius(direction, radii)
    for b in bumps:
        cos_sim = direction @ b["dir"]
        chordal_sq = np.clip(2 - 2 * cos_sim, 0, None)  # squared chord dist on unit sphere
        r = r + b["delta"] * np.exp(-chordal_sq / (2 * b["sigma"] ** 2))
    return r


def ovoid_mask(shape, center, radii, rot_matrix, bumps=None, margin=1.0):
    """Boolean mask of `shape` that is True inside the ovoid, including any
    local bumps.

    radii: (r1_pos, r1_neg, r2, r3). Only touches a bounding box around
    the (possibly bumped) shape for speed.
    """
    center = np.asarray(center, dtype=float)
    bumps = bumps or []
    r1p, r1n, r2, r3 = radii
    max_bump = max((b["delta"] for b in bumps), default=0.0)
    max_r = (max(r1p, r1n, r2, r3) + max(max_bump, 0.0)) * margin + 1.0  # +1 px padding

    lo = np.maximum(np.floor(center - max_r).astype(int), 0)
    hi = np.minimum(np.ceil(center + max_r).astype(int) + 1, shape)
    if np.any(hi <= lo):
        return None, None  # entirely outside the volume

    zz, yy, xx = np.meshgrid(
        np.arange(lo[0], hi[0]),
        np.arange(lo[1], hi[1]),
        np.arange(lo[2], hi[2]),
        indexing="ij",
    )
    pts = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3) - center
    local = pts @ rot_matrix  # rotate into ovoid-local frame
    dist = np.linalg.norm(local, axis=-1)
    dist_safe = np.where(dist < 1e-9, 1e-9, dist)
    direction = local / dist_safe[:, None]

    r_thresh = ovoid_total_radius(direction, radii, bumps)
    inside = dist <= r_thresh
    inside = inside.reshape(hi - lo)
    return inside, (lo, hi)


def format_seconds(seconds):
    """Human-readable duration, e.g. 12.0 -> '12s', 75.0 -> '1m 15s'."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def compute_outline_array(labels_array):
    """For each label, keep only the voxels touching background WITHIN
    their own 2D slice (axis 0 = the current slice axis) -- i.e. a
    per-slice 2D outline, not a true 3D surface. This is deliberately
    orientation-dependent: recomputing it fresh after axis 0/1 get
    swapped (see `toggle_axes_view`) gives correct outlines for whichever
    orientation you're currently viewing, since a middle slice through a
    solid 3D shape has a very different 2D outline than its 3D surface.
    """
    outline = np.zeros_like(labels_array)
    # 3x3x3 structuring element with in-plane (axis 1/2) 4-connectivity
    # only -- no connection across axis 0 (the slice axis), so erosion
    # never "sees" neighboring slices.
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = ndimage.generate_binary_structure(2, 1)
    for label_id in np.unique(labels_array):
        if label_id == 0:
            continue
        mask = labels_array == label_id
        eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
        outline[mask & ~eroded] = label_id
    return outline


# --------------------------------------------------------------------------
# Widget
# --------------------------------------------------------------------------

class EllipsoidFitterWidget(QWidget):
    LABELS_LAYER_NAME = "Ellipsoid Labels"
    POINTS_LAYER_NAME = "boundary points"
    BUMP_POINTS_LAYER_NAME = "bump points"
    OUTLINES_LAYER_NAME = "Ovoid Outlines"

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.ellipsoids: dict[int, dict] = {}  # label_id -> params
        self._next_label_id = 1
        self._next_bump_id = 1
        self.time_spent = {}  # label_id -> accumulated seconds
        self._active_target = None  # None | "pending" | <label_id>
        self._active_checkpoint = time.monotonic()
        self._pending_accum = 0.0  # accumulated time for a not-yet-fit ovoid
        self._paused = False
        self._paused_target = None  # what to resume tracking once un-paused
        self._updating_ui = False
        self._axes_swapped = False

        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- image layer selection ---
        layout.addWidget(QLabel("Image layer:"))
        self.image_combo = QComboBox()
        self._refresh_image_choices()
        layout.addWidget(self.image_combo)
        self.viewer.layers.events.inserted.connect(lambda e: self._refresh_image_choices())
        self.viewer.layers.events.removed.connect(lambda e: self._refresh_image_choices())

        # --- axis swap (x-y <-> x-z scrolling view) ---
        self.swap_btn = QPushButton("Switch to X-Z View (scroll Y)")
        self.swap_btn.clicked.connect(self.toggle_axes_view)
        layout.addWidget(self.swap_btn)

        # --- point picking controls ---
        btn_row = QHBoxLayout()
        self.new_btn = QPushButton("Start New Ovoid")
        self.new_btn.clicked.connect(self.start_new_ellipsoid)
        self.fit_btn = QPushButton("Fit Ovoid From Points")
        self.fit_btn.clicked.connect(self.fit_from_points)
        btn_row.addWidget(self.new_btn)
        btn_row.addWidget(self.fit_btn)
        layout.addLayout(btn_row)

        # --- ellipsoid list ---
        layout.addWidget(QLabel("Ovoids:"))
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected)
        layout.addWidget(self.delete_btn)

        self.outlines_btn = QPushButton("See Outlines")
        self.outlines_btn.clicked.connect(self.show_outlines)
        layout.addWidget(self.outlines_btn)

        # --- pause the per-ovoid time tracker (e.g. for breaks) ---
        self.pause_btn = QPushButton("Pause Timer")
        self.pause_btn.clicked.connect(self.toggle_pause)
        layout.addWidget(self.pause_btn)

        # --- save/load all ovoid parameters as JSON ---
        save_load_row = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Parameters (JSON)")
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.load_params_btn = QPushButton("Load Parameters (JSON)")
        self.load_params_btn.clicked.connect(self.load_parameters)
        save_load_row.addWidget(self.save_params_btn)
        save_load_row.addWidget(self.load_params_btn)
        layout.addLayout(save_load_row)

        # --- bumps (local "pull" corrections on top of the base ovoid) ---
        layout.addWidget(QLabel("Bumps (select an ovoid above first):"))
        self.spin_bump_width = self._make_spin(1, 89, None, None, standalone=True)
        self.spin_bump_width.setValue(15.0)
        bump_width_row = QHBoxLayout()
        bump_width_row.addWidget(QLabel("Bump width (deg):"))
        bump_width_row.addWidget(self.spin_bump_width)
        layout.addLayout(bump_width_row)

        bump_btn_row = QHBoxLayout()
        self.bump_start_btn = QPushButton("Add Bump Points")
        self.bump_start_btn.clicked.connect(self.start_bump_points)
        self.bump_apply_btn = QPushButton("Pull Bumps From Points")
        self.bump_apply_btn.clicked.connect(self.apply_bumps_from_points)
        bump_btn_row.addWidget(self.bump_start_btn)
        bump_btn_row.addWidget(self.bump_apply_btn)
        layout.addLayout(bump_btn_row)

        self.bump_list_widget = QListWidget()
        layout.addWidget(self.bump_list_widget)

        bump_delete_row = QHBoxLayout()
        self.bump_delete_btn = QPushButton("Delete Selected Bump")
        self.bump_delete_btn.clicked.connect(self.delete_selected_bump)
        self.bump_clear_btn = QPushButton("Clear All Bumps (Selected Ovoid)")
        self.bump_clear_btn.clicked.connect(self.clear_bumps_selected)
        bump_delete_row.addWidget(self.bump_delete_btn)
        bump_delete_row.addWidget(self.bump_clear_btn)
        layout.addLayout(bump_delete_row)

        # --- parameter editing ---
        params_box = QGroupBox("Edit Selected Ovoid")
        form = QFormLayout()
        params_box.setLayout(form)
        layout.addWidget(params_box)

        self.spin_cz = self._make_spin(-1e6, 1e6, form, "Center Z")
        self.spin_cy = self._make_spin(-1e6, 1e6, form, "Center Y")
        self.spin_cx = self._make_spin(-1e6, 1e6, form, "Center X")
        self.spin_r1p = self._make_spin(0.1, 1e6, form, "Long axis radius (+ end)")
        self.spin_r1n = self._make_spin(0.1, 1e6, form, "Long axis radius (- end)")
        self.spin_r2 = self._make_spin(0.1, 1e6, form, "Radius 2")
        self.spin_r3 = self._make_spin(0.1, 1e6, form, "Radius 3")
        self.spin_rot_z = self._make_spin(-180, 180, form, "Rotation Z (deg)")
        self.spin_rot_y = self._make_spin(-180, 180, form, "Rotation Y (deg)")
        self.spin_rot_x = self._make_spin(-180, 180, form, "Rotation X (deg)")

        for spin in (
            self.spin_cz, self.spin_cy, self.spin_cx,
            self.spin_r1p, self.spin_r1n, self.spin_r2, self.spin_r3,
            self.spin_rot_z, self.spin_rot_y, self.spin_rot_x,
        ):
            spin.valueChanged.connect(self._on_params_edited)

        layout.addStretch()

    def _make_spin(self, lo, hi, form, label, standalone=False):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        if not standalone:
            form.addRow(label, spin)
        return spin

    # ---------------------------------------------------------------- utils

    def _refresh_image_choices(self):
        current = self.image_combo.currentText()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        names = [l.name for l in self.viewer.layers if l.__class__.__name__ in ("Image", "Labels")
                 and l.name != self.LABELS_LAYER_NAME]
        self.image_combo.addItems(names)
        if current in names:
            self.image_combo.setCurrentText(current)
        self.image_combo.blockSignals(False)

    def _reference_shape(self):
        name = self.image_combo.currentText()
        if not name or name not in self.viewer.layers:
            return None
        return tuple(self.viewer.layers[name].data.shape[-3:])

    def _touch(self, target):
        """Record that work has just shifted to `target` (an ovoid's
        label_id, the string "pending" for a not-yet-created ovoid while
        its boundary points are being clicked, or None to just stop the
        clock). Adds elapsed time since the last touch to whichever
        target was PREVIOUSLY active -- which may be the same target
        (a routine "checkpoint" flush, e.g. from a slider edit) or a
        different one (e.g. selecting a different ovoid in the list, which
        correctly finishes crediting time to the one you were just on).

        While paused, this is a complete no-op -- selecting a different
        ovoid or nudging a slider during a break won't start tracking
        anything until you explicitly resume.
        """
        if self._paused:
            return
        now = time.monotonic()
        if self._active_target is not None:
            elapsed = now - self._active_checkpoint
            if self._active_target == "pending":
                self._pending_accum += elapsed
            else:
                self.time_spent[self._active_target] = (
                    self.time_spent.get(self._active_target, 0.0) + elapsed
                )
                self._update_list_item_text(self._active_target)
        self._active_target = target
        self._active_checkpoint = now

    def toggle_pause(self):
        """Pause: flush and stop the clock, remembering what was active so
        it can resume seamlessly later. Resume: pick tracking back up on
        that same target, with no time counted for the paused interval.
        """
        if not self._paused:
            self._paused_target = self._active_target
            self._touch(None)  # flush elapsed time up to now (still unpaused at this point)
            self._paused = True
            self.pause_btn.setText("Resume Timer")
            show_info("Timer paused -- nothing will be tracked until you resume.")
        else:
            self._paused = False
            self._active_target = self._paused_target
            self._active_checkpoint = time.monotonic()
            self._paused_target = None
            self.pause_btn.setText("Pause Timer")
            show_info("Timer resumed.")

    def _update_list_item_text(self, label_id):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == label_id:
                elapsed = self.time_spent.get(label_id, 0.0)
                item.setText(f"Ovoid {label_id}  ({format_seconds(elapsed)})")
                return

    def _get_or_create_points_layer(self):
        if self.POINTS_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[self.POINTS_LAYER_NAME]
        layer = self.viewer.add_points(
            np.empty((0, 3)),
            name=self.POINTS_LAYER_NAME,
            ndim=3,
            size=3,
            face_color="yellow",
        )
        return layer

    def _get_or_create_bump_points_layer(self):
        if self.BUMP_POINTS_LAYER_NAME in self.viewer.layers:
            return self.viewer.layers[self.BUMP_POINTS_LAYER_NAME]
        layer = self.viewer.add_points(
            np.empty((0, 3)),
            name=self.BUMP_POINTS_LAYER_NAME,
            ndim=3,
            size=3,
            face_color="magenta",
        )
        return layer

    def _get_or_create_labels_layer(self):
        shape = self._reference_shape()
        if shape is None:
            raise RuntimeError(
                "No image layer selected in the dropdown -- pick your loaded "
                "image there first."
            )
        if self.LABELS_LAYER_NAME in self.viewer.layers:
            layer = self.viewer.layers[self.LABELS_LAYER_NAME]
            if layer.data.shape != shape:
                raise RuntimeError(
                    "Existing Labels layer shape doesn't match the "
                    "selected image; delete it or pick a matching image."
                )
            return layer
        data = np.zeros(shape, dtype=np.uint32)
        return self.viewer.add_labels(data, name=self.LABELS_LAYER_NAME)

    # ------------------------------------------------------------ actions

    def start_new_ellipsoid(self):
        self._touch(None)  # flush whatever was active; a fresh timing window starts now
        self._pending_accum = 0.0
        self._active_target = "pending"
        self._active_checkpoint = time.monotonic()

        points_layer = self._get_or_create_points_layer()
        points_layer.data = np.empty((0, 3))
        points_layer.mode = "add"
        self.viewer.layers.selection.active = points_layer

    def fit_from_points(self):
        if self.POINTS_LAYER_NAME not in self.viewer.layers:
            show_error("No 'boundary points' layer yet -- click 'Start New Ellipsoid' first.")
            return
        points_layer = self.viewer.layers[self.POINTS_LAYER_NAME]
        pts = np.asarray(points_layer.data)
        if pts.shape[-1] > 3:
            pts = pts[:, -3:]  # drop any leading non-spatial dims

        if pts.shape[0] < 9:
            show_error(
                f"Only {pts.shape[0]} boundary point(s) clicked -- need at least 9 "
                "(15-20+ spread over the surface recommended)."
            )
            return

        try:
            center, radii, rot = fit_ovoid(pts)
        except ValueError as exc:
            show_error(f"Ovoid fit failed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected in the UI
            show_error(f"Unexpected error while fitting: {exc}")
            raise

        try:
            label_id = self._next_label_id
            self._next_label_id += 1
            new_params = {
                "center": center,
                "radii": radii,  # (r1_pos, r1_neg, r2, r3)
                "rot": rot,
                "bumps": {},  # bump_id -> bump dict
            }
            # `center`/`radii`/`rot` were just fit from points clicked in
            # whatever view is CURRENTLY displayed -- convert to canonical
            # (first-view) coordinates before storing, so ovoid parameters
            # never depend on which orientation you happened to fit it in
            if self._axes_swapped:
                self._relabel_ovoid_axes(new_params)
            self.ellipsoids[label_id] = new_params

            # transfer accumulated "pending" time (spent clicking boundary
            # points before this ovoid existed) into its own time budget,
            # then keep the clock running against this ovoid's real id
            self._touch("pending")
            self.time_spent[label_id] = self.time_spent.get(label_id, 0.0) + self._pending_accum
            self._pending_accum = 0.0
            self._active_target = label_id
            self._active_checkpoint = time.monotonic()

            self._rebuild_labels()
        except Exception as exc:  # noqa: BLE001
            show_error(f"Fit succeeded but rasterizing into Labels failed: {exc}")
            raise

        item = QListWidgetItem(f"Ovoid {label_id}  ({format_seconds(self.time_spent[label_id])})")
        item.setData(Qt.UserRole, label_id)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)

        # clear points so the next click sequence starts fresh
        points_layer.data = np.empty((0, 3))
        show_info(f"Fit Ovoid {label_id} from {pts.shape[0]} points.")

    def delete_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        label_id = item.data(Qt.UserRole)
        if self._active_target == label_id:
            self._touch(None)  # stop tracking -- this ovoid is about to be gone
        del self.ellipsoids[label_id]
        self.time_spent.pop(label_id, None)
        self.list_widget.takeItem(self.list_widget.row(item))
        # this ovoid's bumps are gone too (they were nested inside it) --
        # clear the bump list so it doesn't keep showing stale entries;
        # takeItem() doesn't reliably fire currentItemChanged on its own
        self.bump_list_widget.clear()
        self._rebuild_labels()  # lets any ovoid that was covered by this one reappear

    def _ellipsoids_to_serializable(self):
        """Build a plain-JSON-compatible dict from self.ellipsoids (numpy
        arrays -> nested lists, dict keys -> strings, since JSON object
        keys must be strings). Ovoid parameters are always stored
        canonically -- in terms of whichever orientation you saw first --
        so the saved file is meaningful regardless of which orientation
        happened to be displayed at save time.
        """
        data = {
            "next_label_id": self._next_label_id,
            "next_bump_id": self._next_bump_id,
            "ovoids": {},
        }
        for label_id, params in self.ellipsoids.items():
            bumps_out = {}
            for bump_id, bump in params.get("bumps", {}).items():
                bumps_out[str(bump_id)] = {
                    "dir": [float(v) for v in bump["dir"]],
                    "delta": float(bump["delta"]),
                    "sigma": float(bump["sigma"]),
                }
            data["ovoids"][str(label_id)] = {
                "center": [float(v) for v in params["center"]],
                "radii": [float(v) for v in params["radii"]],
                "rot": [[float(v) for v in row] for row in params["rot"]],
                "bumps": bumps_out,
                "time_spent_seconds": self.time_spent.get(label_id, 0.0),
            }
        return data

    def _ellipsoids_from_serializable(self, data):
        """Inverse of `_ellipsoids_to_serializable` -- rebuilds numpy
        arrays and integer-keyed dicts from a loaded JSON structure. No
        orientation correction needed on load: canonical storage means
        the file already means the same thing regardless of which
        orientation the volume happens to be displayed in right now.
        """
        ellipsoids = {}
        time_spent = {}
        for label_id_str, p in data.get("ovoids", {}).items():
            label_id = int(label_id_str)
            bumps = {}
            for bump_id_str, b in p.get("bumps", {}).items():
                bumps[int(bump_id_str)] = {
                    "dir": np.array(b["dir"], dtype=float),
                    "delta": float(b["delta"]),
                    "sigma": float(b["sigma"]),
                }
            ellipsoids[label_id] = {
                "center": np.array(p["center"], dtype=float),
                "radii": np.array(p["radii"], dtype=float),
                "rot": np.array(p["rot"], dtype=float),
                "bumps": bumps,
            }
            time_spent[label_id] = float(p.get("time_spent_seconds", 0.0))
        next_label_id = data.get("next_label_id", max(ellipsoids.keys(), default=0) + 1)
        all_bump_ids = [bid for p in ellipsoids.values() for bid in p["bumps"].keys()]
        next_bump_id = data.get("next_bump_id", max(all_bump_ids, default=0) + 1)
        return ellipsoids, next_label_id, next_bump_id, time_spent

    def save_parameters(self):
        if not self.ellipsoids:
            show_error("No ovoids to save yet.")
            return
        self._touch(self._active_target)  # flush the currently-running timer before saving
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ovoid Parameters", "ovoid_parameters.json", "JSON files (*.json)"
        )
        if not path:
            return  # user cancelled
        data = self._ellipsoids_to_serializable()
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            show_error(f"Failed to save parameters: {exc}")
            return
        show_info(f"Saved {len(self.ellipsoids)} ovoid(s) to {path}")

    def load_parameters(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ovoid Parameters", "", "JSON files (*.json)"
        )
        if not path:
            return  # user cancelled
        try:
            with open(path, "r") as f:
                data = json.load(f)
            ellipsoids, next_label_id, next_bump_id, time_spent = self._ellipsoids_from_serializable(data)
        except Exception as exc:  # noqa: BLE001
            show_error(f"Failed to load parameters: {exc}")
            return

        self.ellipsoids = ellipsoids
        self._next_label_id = next_label_id
        self._next_bump_id = next_bump_id
        self.time_spent = time_spent
        self._active_target = None  # nothing actively being worked on right after a load
        self._pending_accum = 0.0

        self.list_widget.clear()
        self.bump_list_widget.clear()
        for label_id in sorted(self.ellipsoids.keys()):
            elapsed = self.time_spent.get(label_id, 0.0)
            item = QListWidgetItem(f"Ovoid {label_id}  ({format_seconds(elapsed)})")
            item.setData(Qt.UserRole, label_id)
            self.list_widget.addItem(item)

        self._rebuild_labels()
        show_info(f"Loaded {len(self.ellipsoids)} ovoid(s) from {path}. Make sure the "
                   "correct image is selected above if the Labels layer needed re-creating.")

    def show_outlines(self):
        """Create (or refresh) a layer showing only the per-slice 2D
        outline of every ovoid currently placed -- interior voxels
        hidden, only the voxels touching background within their own
        slice shown. Once created, this layer keeps itself in sync
        automatically whenever an ovoid is fit/edited/deleted or the view
        is toggled (see `_refresh_outlines`), so you don't need to keep
        re-clicking this button.
        """
        if self.LABELS_LAYER_NAME not in self.viewer.layers:
            show_error("No ovoids fitted yet -- nothing to outline.")
            return
        labels_layer = self.viewer.layers[self.LABELS_LAYER_NAME]
        outline_data = compute_outline_array(labels_layer.data)

        if self.OUTLINES_LAYER_NAME in self.viewer.layers:
            outline_layer = self.viewer.layers[self.OUTLINES_LAYER_NAME]
            outline_layer.data = outline_data
        else:
            outline_layer = self.viewer.add_labels(outline_data, name=self.OUTLINES_LAYER_NAME)
        outline_layer.refresh()
        show_info("Showing ovoid outlines -- this updates automatically as you "
                   "edit ovoids or switch views.")

    def _refresh_outlines(self):
        """Recompute the outlines layer from the current Labels data, but
        only if that layer already exists (i.e. 'See Outlines' has been
        clicked at least once) -- otherwise this is a no-op, so it's safe
        to call unconditionally after every edit/toggle.
        """
        if self.OUTLINES_LAYER_NAME not in self.viewer.layers:
            return
        if self.LABELS_LAYER_NAME not in self.viewer.layers:
            return
        labels_layer = self.viewer.layers[self.LABELS_LAYER_NAME]
        outline_layer = self.viewer.layers[self.OUTLINES_LAYER_NAME]
        outline_layer.data = compute_outline_array(labels_layer.data)
        outline_layer.refresh()

    def start_bump_points(self):
        item = self.list_widget.currentItem()
        if item is None:
            show_error("Select an ovoid from the list first -- bumps are added to whichever "
                       "ovoid is currently selected.")
            return
        self._touch(item.data(Qt.UserRole))
        bump_layer = self._get_or_create_bump_points_layer()
        bump_layer.data = np.empty((0, 3))
        bump_layer.mode = "add"
        self.viewer.layers.selection.active = bump_layer
        show_info("Click on the object where it bulges out beyond the current ovoid, "
                   "then click 'Pull Bumps From Points'.")

    def apply_bumps_from_points(self):
        item = self.list_widget.currentItem()
        if item is None:
            show_error("Select an ovoid from the list first.")
            return
        if self.BUMP_POINTS_LAYER_NAME not in self.viewer.layers:
            show_error("No bump points placed yet -- click 'Add Bump Points' first.")
            return
        label_id = item.data(Qt.UserRole)
        self._touch(label_id)
        canonical_params = self.ellipsoids[label_id]
        # bump points were clicked in the CURRENTLY DISPLAYED orientation,
        # so compute direction/delta against the current-view representation
        view_params = self._current_view_params(canonical_params)

        bump_layer = self.viewer.layers[self.BUMP_POINTS_LAYER_NAME]
        pts = np.asarray(bump_layer.data)
        if pts.shape[-1] > 3:
            pts = pts[:, -3:]
        if pts.shape[0] == 0:
            show_error("No bump points placed yet.")
            return

        sigma = np.radians(self.spin_bump_width.value())
        canonical_bumps = canonical_params.setdefault("bumps", {})
        # running list of bumps-so-far in VIEW-frame terms, so each new
        # click is compared against the CURRENT surface (base + every bump
        # applied so far, including earlier ones in this same batch) --
        # not just the original base shape. Otherwise clicking a point
        # that's outside an already-pulled-in region but inside the
        # original shape reads as "still inside" and pulls further in
        # instead of moving the surface out toward where you clicked.
        view_bumps = list(view_params.get("bumps", {}).values())
        n_added = 0
        for pt in pts:
            rel = np.asarray(pt, dtype=float) - view_params["center"]
            local = rel @ view_params["rot"]
            dist = np.linalg.norm(local)
            if dist < 1e-6:
                continue  # clicked exactly on the center -- ambiguous direction, skip
            direction = local / dist
            current_r = ovoid_total_radius(direction, view_params["radii"], view_bumps)
            delta = dist - current_r
            new_bump_view = {"dir": direction, "delta": float(delta), "sigma": float(sigma)}
            view_bumps.append(new_bump_view)  # so later points in this batch see it too

            # this bump was computed in CURRENT-view local terms -- convert
            # to canonical before storing, same way a fresh fit would be
            # (bumps live in the ovoid's local frame, so this is just the
            # same "flip axis-0's direction" correction as the full ovoid
            # transform, applied to this one bump)
            if self._axes_swapped:
                d = new_bump_view["dir"].copy()
                d[0] = -d[0]
                new_bump = {"dir": d, "delta": new_bump_view["delta"], "sigma": new_bump_view["sigma"]}
            else:
                new_bump = new_bump_view

            bump_id = self._next_bump_id
            self._next_bump_id += 1
            canonical_bumps[bump_id] = new_bump
            n_added += 1

        self._rebuild_labels()
        self._refresh_bump_list(label_id)
        bump_layer.data = np.empty((0, 3))
        show_info(f"Added {n_added} bump(s) to Ovoid {label_id}.")

    def delete_selected_bump(self):
        ovoid_item = self.list_widget.currentItem()
        if ovoid_item is None:
            show_error("Select an ovoid from the list first.")
            return
        bump_item = self.bump_list_widget.currentItem()
        if bump_item is None:
            show_error("Select a bump to delete from the list.")
            return
        label_id = ovoid_item.data(Qt.UserRole)
        self._touch(label_id)
        bump_id = bump_item.data(Qt.UserRole)
        bumps = self.ellipsoids[label_id].get("bumps", {})
        if bump_id in bumps:
            del bumps[bump_id]
        self._rebuild_labels()
        self._refresh_bump_list(label_id)
        show_info(f"Deleted bump {bump_id} from Ovoid {label_id}.")

    def clear_bumps_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            show_error("Select an ovoid from the list first.")
            return
        label_id = item.data(Qt.UserRole)
        self._touch(label_id)
        self.ellipsoids[label_id]["bumps"] = {}
        self._rebuild_labels()
        self._refresh_bump_list(label_id)
        show_info(f"Cleared bumps on Ovoid {label_id}.")

    def _refresh_bump_list(self, label_id):
        self.bump_list_widget.clear()
        bumps = self.ellipsoids.get(label_id, {}).get("bumps", {})
        for bump_id, bump in bumps.items():
            width_deg = np.degrees(bump["sigma"])
            item = QListWidgetItem(
                f"Bump {bump_id}  (delta={bump['delta']:.2f}, width={width_deg:.0f} deg)"
            )
            item.setData(Qt.UserRole, bump_id)
            self.bump_list_widget.addItem(item)

    def _relabel_ovoid_axes(self, params):
        """Convert one ovoid's parameters (in place) to describe the exact
        same physical shape after global axes 0/1 get relabeled -- used
        both by `toggle_axes_view` (for every currently-live ovoid) and by
        `load_parameters` (for ovoids loaded from a JSON file saved in a
        different X-Y/X-Z orientation than the volume is currently in).

        Center: swap coord 0<->1. Rotation: swapping two GLOBAL axes always
        flips a rotation matrix's determinant from +1 to -1 (it becomes an
        improper "rotation + mirror" matrix) -- scipy's Rotation correctly
        rejects that. Fix: also flip the local axis-0 direction (negate
        that column), which restores det=+1 while describing the exact
        same physical shape -- and since axis 0 is the asymmetric long
        axis, flipping its direction means r1_pos/r1_neg swap meaning too.
        Bumps live in that same local frame, so their direction's first
        component flips too, or a bump would silently point the wrong way.
        """
        c = params["center"]
        params["center"] = c[[1, 0, 2]]
        rot = params["rot"].copy()
        rot[[0, 1], :] = rot[[1, 0], :]
        rot[:, 0] = -rot[:, 0]
        params["rot"] = rot
        r1p, r1n, r2, r3 = params["radii"]
        params["radii"] = np.array([r1n, r1p, r2, r3])
        for bump in params.get("bumps", {}).values():
            d = bump["dir"].copy()
            d[0] = -d[0]
            bump["dir"] = d

    def _current_view_params(self, params):
        """Ovoid parameters are always stored CANONICALLY -- in terms of
        whichever orientation you saw first, regardless of how many times
        you've toggled the view since. This returns a transformed COPY
        suitable for the volume's CURRENT physical orientation (used for
        rasterizing/displaying), leaving the stored canonical params
        untouched. Since `_relabel_ovoid_axes` is its own inverse, this
        same function also converts the other direction (current-view ->
        canonical), which is used when a fit/bump was just computed from
        freshly-clicked points in the current view.
        """
        if not self._axes_swapped:
            return params
        view_params = {
            "center": params["center"].copy(),
            "radii": params["radii"].copy(),
            "rot": params["rot"].copy(),
            "bumps": {
                bid: {"dir": b["dir"].copy(), "delta": b["delta"], "sigma": b["sigma"]}
                for bid, b in params.get("bumps", {}).items()
            },
        }
        self._relabel_ovoid_axes(view_params)
        return view_params

    def toggle_axes_view(self):
        """Swap axes 0 and 1 of the image(s) and boundary/bump-points
        layers so you can add boundary points from a second orthogonal
        scrolling direction. Ovoid parameters are stored CANONICALLY
        (always in terms of the orientation you first saw) and are never
        touched by this -- only the Labels layer gets recreated, freshly
        rasterized from that canonical data into the new orientation.
        """
        # recreating the Labels layer below auto-selects it in napari,
        # which would otherwise silently steal focus from whatever layer
        # you had selected -- capture it now (by name, since the Labels
        # layer gets rebuilt as a brand new object) and restore it at the end
        previously_active_name = None
        if self.viewer.layers.selection.active is not None:
            previously_active_name = self.viewer.layers.selection.active.name

        swapped_any = False
        for layer in list(self.viewer.layers):
            if layer.name in (self.OUTLINES_LAYER_NAME, self.LABELS_LAYER_NAME):
                continue  # both get freshly rebuilt from canonical data below
            if isinstance(layer, Points):
                data = np.asarray(layer.data)
                if data.size and data.shape[1] >= 2:
                    order = [1, 0] + list(range(2, data.shape[1]))
                    layer.data = data[:, order]
                swapped_any = True
            elif isinstance(layer, Image):
                if layer.data.ndim >= 2:
                    layer.data = np.swapaxes(layer.data, 0, 1)
                    swapped_any = True
            # other layer types (Shapes, Vectors, ...) are left untouched;
            # this widget doesn't create any, but warn if you add your own
            elif layer.data is not None and getattr(layer.data, "ndim", 0) >= 2:
                show_info(f"Skipped swapping layer '{layer.name}' (unsupported type).")

        if not swapped_any:
            show_info("No image or points layers to swap yet.")
            return

        # the old Labels layer's shape won't match the now-swapped image --
        # drop it so `_rebuild_labels` recreates it fresh at the right shape
        if self.LABELS_LAYER_NAME in self.viewer.layers:
            del self.viewer.layers[self.LABELS_LAYER_NAME]

        self._axes_swapped = not self._axes_swapped
        self.swap_btn.setText(
            "Switch to Original View (scroll Z)"
            if self._axes_swapped
            else "Switch to X-Z View (scroll Y)"
        )
        show_info("Axes swapped -- ovoid parameters are unaffected, only the display.")

        # re-activate point adding in the new view, if a points layer exists
        if self.POINTS_LAYER_NAME in self.viewer.layers:
            points_layer = self.viewer.layers[self.POINTS_LAYER_NAME]
            points_layer.mode = "add"
            self.viewer.layers.selection.active = points_layer

        self._jump_to_busiest_slice()
        self._rebuild_labels()

        # restore whichever layer was selected before toggling -- this
        # runs last specifically to override the auto-select that happens
        # when `_rebuild_labels` recreates the Labels layer above
        if previously_active_name is not None and previously_active_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[previously_active_name]

    def _jump_to_busiest_slice(self):
        """After swapping views, jump the slider to whichever slice (along
        the axis you now scroll through) has the most boundary/bump points
        on it -- otherwise it's easy to lose track of where you were
        clicking after switching orientation.
        """
        counts = Counter()
        for name in (self.POINTS_LAYER_NAME, self.BUMP_POINTS_LAYER_NAME):
            if name not in self.viewer.layers:
                continue
            data = np.asarray(self.viewer.layers[name].data)
            if data.size == 0:
                continue
            counts.update(np.round(data[:, 0]).astype(int).tolist())

        if not counts:
            return  # no points placed yet -- nothing to jump to

        busiest = counts.most_common(1)[0][0]

        shape = self._reference_shape()
        if shape is not None:
            busiest = max(0, min(shape[0] - 1, busiest))

        current_step = list(self.viewer.dims.current_step)
        if not current_step:
            return
        current_step[0] = busiest
        self.viewer.dims.current_step = tuple(current_step)

    # -------------------------------------------------------- UI <-> data

    def _on_selection_changed(self, current, _previous):
        if current is None:
            self._touch(None)
            return
        label_id = current.data(Qt.UserRole)
        self._touch(label_id)
        params = self.ellipsoids[label_id]
        euler = Rotation.from_matrix(params["rot"]).as_euler("zyx", degrees=True)
        r1p, r1n, r2, r3 = params["radii"]

        self._updating_ui = True
        self.spin_cz.setValue(params["center"][0])
        self.spin_cy.setValue(params["center"][1])
        self.spin_cx.setValue(params["center"][2])
        self.spin_r1p.setValue(r1p)
        self.spin_r1n.setValue(r1n)
        self.spin_r2.setValue(r2)
        self.spin_r3.setValue(r3)
        self.spin_rot_z.setValue(euler[0])
        self.spin_rot_y.setValue(euler[1])
        self.spin_rot_x.setValue(euler[2])
        self._updating_ui = False

        self._refresh_bump_list(label_id)

    def _on_params_edited(self, *_args):
        if self._updating_ui:
            return
        item = self.list_widget.currentItem()
        if item is None:
            return
        label_id = item.data(Qt.UserRole)
        self._touch(label_id)

        center = np.array([self.spin_cz.value(), self.spin_cy.value(), self.spin_cx.value()])
        radii = np.array([
            self.spin_r1p.value(), self.spin_r1n.value(),
            self.spin_r2.value(), self.spin_r3.value(),
        ])
        rot = Rotation.from_euler(
            "zyx",
            [self.spin_rot_z.value(), self.spin_rot_y.value(), self.spin_rot_x.value()],
            degrees=True,
        ).as_matrix()

        # keep any existing bumps -- they're defined in local coordinates so
        # they stay correctly attached as center/radii/rotation are edited
        existing_bumps = self.ellipsoids[label_id].get("bumps", {})
        self.ellipsoids[label_id] = {
            "center": center, "radii": radii, "rot": rot, "bumps": existing_bumps,
        }
        self._rebuild_labels()

    # --------------------------------------------------------- voxel I/O

    def _rebuild_labels(self):
        """Redraw every ovoid from scratch into the Labels layer, in
        ascending id order (so a higher id always wins an overlap, same
        as before). Rebuilding fresh rather than incrementally
        clearing/redrawing a single ovoid means deleting one automatically
        lets whatever was underneath it reappear -- no special-case
        "restore the covered one" logic needed.
        """
        if not self.ellipsoids and self.LABELS_LAYER_NAME not in self.viewer.layers:
            return  # nothing to draw and no layer to clear
        layer = self._get_or_create_labels_layer()
        layer.data[:] = 0
        out_of_bounds = []
        for label_id in sorted(self.ellipsoids.keys()):
            params = self._current_view_params(self.ellipsoids[label_id])
            mask, bbox = ovoid_mask(
                layer.data.shape, params["center"], params["radii"], params["rot"],
                bumps=list(params.get("bumps", {}).values()),
            )
            if mask is None:
                out_of_bounds.append(label_id)
                continue
            (z0, y0, x0), (z1, y1, x1) = bbox
            region = layer.data[z0:z1, y0:y1, x0:x1]
            region[mask] = label_id
            layer.data[z0:z1, y0:y1, x0:x1] = region
        layer.refresh()
        self._refresh_outlines()
        if out_of_bounds:
            ids_str = ", ".join(str(i) for i in out_of_bounds)
            show_error(
                f"Ovoid(s) {ids_str} fell entirely outside the current volume "
                f"({tuple(layer.data.shape)}) and weren't drawn. If these were "
                "loaded from a JSON file, this usually means the volume was in "
                "a different X-Y/X-Z orientation when the file was saved than "
                "it is now -- try toggling the view to match, or re-fit those "
                "ovoids in the current orientation."
            )
