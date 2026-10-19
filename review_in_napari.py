"""
review_in_napari.py
===================
Open one cut-out and its annotations in napari to review and correct them by hand
(paint, erase, fill), then save. Labels: 0 ToPredict, 1 Matrix, 2 Rocks, 3 Pores.

    pip install "napari[all]"
    python review_in_napari.py LAU-OLY

Corrected annotations overwrite data/annotations/<sample_id>.tif when napari is closed.
Then re-run 03_segment_random_forest.py, 04_calibration_effect.py and 05_figures.py.
"""
import sys

import napari
import tifffile

from config import ANNOT_DIR, COLORS, GRAY_DIR, VOXEL_MM

sid = sys.argv[1] if len(sys.argv) > 1 else "LAU-OLY"
img = tifffile.imread(GRAY_DIR / f"{sid}.tif")
ann = tifffile.imread(ANNOT_DIR / f"{sid}.tif")

viewer = napari.Viewer(title=f"{sid}: review annotations")
viewer.add_image(img, name="CT", colormap="gray", scale=[VOXEL_MM] * 3)
labels = viewer.add_labels(ann, name="annotations", scale=[VOXEL_MM] * 3)
labels.colormap = napari.utils.colormaps.DirectLabelColormap(
    color_dict={k: [c / 255 for c in rgb] + [1.0] for k, rgb in COLORS.items()} | {None: [0, 0, 0, 0]})
napari.run()

tifffile.imwrite(ANNOT_DIR / f"{sid}.tif", labels.data.astype("uint8"), imagej=True,
                 resolution=(1 / VOXEL_MM, 1 / VOXEL_MM),
                 metadata={"spacing": VOXEL_MM, "unit": "mm", "axes": "ZYX"})
print(f"annotations saved: {ANNOT_DIR / (sid + '.tif')}")
