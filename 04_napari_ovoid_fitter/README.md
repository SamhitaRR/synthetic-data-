# Ovoid Fitter

A napari widget for manually segmenting cells (or other roughly egg/ellipsoid-shaped
objects) in 3D image stacks. You click a handful of boundary points on each cell,
the widget fits a 3D ovoid to them, and rasterizes it into a labels layer. Ovoids can
be adjusted afterward — by sliders, or by adding local "bumps" — and the whole
session can be saved to and reloaded from a JSON file.

An ovoid here is an ellipsoid that's allowed to have a different radius on each end
of one axis (i.e. egg-shaped). Setting those two radii equal gives a plain symmetric
ellipsoid.

## Requirements

- Python 3.9+
- [napari](https://napari.org) with a Qt backend
- numpy, scipy

```bash
pip install "napari[all]" scipy
```

(`napari[all]` pulls in a Qt backend, numpy, and qtpy for you — scipy is the one
extra dependency this script needs, for the ovoid-fitting optimizer.)

## Setup

1. Download `ovoid_fitter.py` and put it in its own folder.
2. Open napari.
3. Open napari's built-in console (the `>_` icon in the bottom-left of the viewer, or **Window → Console**).
4. Paste the following, replacing the path with wherever you saved the file:

    ```python
    import sys
    sys.path.insert(0, r"/path/to/folder/with/python/file")
    from ovoid_fitter import EllipsoidFitterWidget
    widget = EllipsoidFitterWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="Ovoid Fitter")
    ```

5. Load your images — drag and drop them into napari, or **File → Open File(s)**.

## Usage

**1. Start a new ovoid**
Click **Start New Ovoid**. This creates (or re-activates) a `boundary points` layer
in add mode.

**2. Add boundary points**
Pick a cell and click along its outline. You can scroll through slices and add
points on several of them. Spread points across all the extremities of the cell in
x, y, *and* z — you'll need at least 9 points to fit an ovoid, and a few more than
that, well distributed, gives a noticeably better fit.

**3. Switch to the X-Z view**
Click **Switch to X-Z View**. Find the cell you were just annotating — it'll have
the yellow points you already placed on it — and add more points from this angle.
This view is especially useful for the top and bottom of the cell, which are hard
to judge accurately when scrolling through the original view. Click the button
again (**Switch to Original View**) to go back.

**4. Fit the ovoid**
Once you're happy with the points, click **Fit Ovoid From Points**. This adds the
ovoid to the list on the left and rasterizes it into an `Ellipsoid Labels` layer
(created automatically the first time).

**5. Check the fit**
- Scroll through the X-Z view or the original view.
- Toggle napari's 3D display (`Ctrl+Y`, or the ndisplay button) and switch the
  `Ellipsoid Labels` layer on and off to compare the fit against the raw image in 3D.
- Click **See Outlines** to add an `ovoid outlines` layer, then hide
  `Ellipsoid Labels` — looking at just the outline on top of the image often makes
  small inaccuracies easier to spot than the filled label does.

**6. Move on, or fix it up**
- Happy with it → click **Start New Ovoid** and repeat for the next cell.
- Not happy at all → select it in the list and click **Delete Selected**.
- Close, but slightly off → select it in the list, then either:
  - drag the center / radii / rotation sliders — changes update the ovoid's voxels
    in `Ellipsoid Labels` live (set the two long-axis radii equal to make it a plain
    symmetric ellipsoid instead of egg-shaped), or
  - add a local bump (next step).

**7. Local adjustments with bumps**
If an ovoid is mostly right but bulges out (or caves in) in one spot:
1. Select the ovoid and click **Add Bump Points** — this opens a `bump points` layer.
2. Click a point at the region that needs adjusting.
3. Click **Pull Bumps From Points** to apply it.

Use **Delete Selected Bump** to remove one bump, or **Clear All Bumps (Selected
Ovoid)** to remove all bumps from the currently selected ovoid.

**8. Save and reload your work**
- **Save Parameters (JSON)** writes every ovoid (center, radii, rotation, bumps) to
  a JSON file.
- **Load Parameters (JSON)** reads one back in, so you can pause and resume a
  session. If you load a file that was saved in a different X-Y/X-Z orientation
  than the one currently displayed, the widget will warn you — switch the view to
  match, or re-fit the affected ovoids.

**9. Timer**
The widget quietly tracks how long you spend annotating. Click **Pause Timer**
before stepping away, and again to resume.

## Notes

- napari's mouse behavior for placing points on a 3D surface has changed across
  versions — worth double-checking against whichever version you're using.
- The fit is a PCA-seeded nonlinear least-squares solve; the PCA seeding matters —
  without it the optimizer tends to land in a bad local minimum (axes swapped or
  misaligned), especially for egg-shaped objects.
- When ovoids overlap, labels are rebuilt from scratch on every change in ascending
  id order, so a higher id always wins the overlap — and deleting an ovoid
  automatically lets whatever it was covering reappear.
