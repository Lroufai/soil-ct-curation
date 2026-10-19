"""
02_prepare_annotations.py
=========================
For each sample:
  1. cut out 300 consecutive slices, central 512 x 512 pixels, saved as one 16-bit ImageJ TIFF
     with the voxel size embedded (data/grayscale/<sample_id>.tif);
  2. calibrate the grey values of the scan on two internal references, the air-filled pores and
     the soil matrix (two-point calibration, one per scan);
  3. place sparse seed annotations on five slices (data/annotations/<sample_id>.tif, label 0 =
     ToPredict), following the nnUNet4SoilXrayCT format;
  4. write dataset_info.json for the nnUNet4SoilXrayCT workflow.

Why a calibration per scan: the six scans share the acquisition settings, but their grey values
drift. The pore peak ranges from about 18 000 to 22 500 and the matrix peak from 32 000 to
36 000 (results/calibration.json). Thresholds common to all scans would therefore label
different materials in different soils. After calibration, n = 0 is air and n = 1 is matrix.

Seed rule on the calibrated, lightly smoothed image n (interiors only, after erosion):
    Pores   n < 0.10        Matrix   0.85 < n < 1.15        Rocks   n > 1.60
Boundaries and partial-volume voxels stay "ToPredict". After calibration the three materials
form three separate clusters in every scan (figures/calibration.png); intermediate values are
mostly boundaries. The seeds are a starting point for interactive correction in napari
(review_in_napari.py).
"""
import json

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.morphology import disk, remove_small_objects

from config import (ANNOT_DIR, ANNOTATED, CLASSES, COLORS, CROP, FIRST_SLICE, GRAY_DIR, LABELS,
                    META_DIR, N_SLICES, RAW_DIR, RESULTS_DIR, ROOT, SAMPLES, VOXEL_MM)

SMOOTH = 1.5        # Gaussian smoothing (pixels) before seeding
ERODE_PX = 2        # erosion radius (pixels): only region interiors are seeded
MIN_REGION_PX = 20  # smallest seed region kept (pixels)
BANDS = {3: (None, 0.10), 1: (0.85, 1.15), 2: (1.60, None)}


def load_cutout(folder, sid):
    """Reuse the saved cut-out if present, else read it from the raw slices."""
    path = GRAY_DIR / f"{sid}.tif"
    if path.exists():
        return tifffile.imread(path)
    stack, c0 = [], None
    for k in range(FIRST_SLICE, FIRST_SLICE + N_SLICES):
        img = tifffile.imread(RAW_DIR / folder / f"slice{k:05d}.tif")
        if c0 is None:
            c0 = ((img.shape[0] - CROP) // 2, (img.shape[1] - CROP) // 2)
        stack.append(img[c0[0]:c0[0] + CROP, c0[1]:c0[1] + CROP])
    return np.stack(stack)


def calibrate(vol):
    """Pore (air) and matrix peaks of the smoothed histogram, from every 15th slice."""
    s = ndi.gaussian_filter(vol[::15].astype(np.float32), (0, SMOOTH, SMOOTH))
    h, e = np.histogram(s, bins=650, range=(0, 65000))
    c = 0.5 * (e[1:] + e[:-1])
    hs = ndi.gaussian_filter1d(h.astype(float), 3)
    peaks, _ = find_peaks(hs, prominence=hs.max() * 0.01)
    mat = [p for p in peaks if 28000 < c[p] < 42000]
    matrix = c[max(mat, key=lambda p: hs[p])]
    air = [p for p in peaks if c[p] < matrix - 8000]
    pore = c[min(air)]
    noise = float(np.std(vol[::15][s < pore + 500]))       # raw grey-value spread inside pores
    return float(pore), float(matrix), noise, c, hs / hs.sum()


def calibrated(img, pore, matrix, smooth=SMOOTH):
    """Grey values mapped to 0 = air-filled pores, 1 = soil matrix."""
    return (ndi.gaussian_filter(img.astype(np.float32), smooth) - pore) / (matrix - pore)


def seed_slice(img, pore, matrix):
    """Confident interiors of the three classes on one slice."""
    n = calibrated(img, pore, matrix)
    lab = np.zeros(img.shape, np.uint8)
    for k, (lo, hi) in BANDS.items():
        msk = np.ones(img.shape, bool)
        if lo is not None:
            msk &= n > lo
        if hi is not None:
            msk &= n < hi
        msk = ndi.binary_erosion(msk, structure=disk(ERODE_PX))
        msk = remove_small_objects(msk, max_size=MIN_REGION_PX)
        lab[msk] = k
    return lab


def save_imagej(path, arr):
    tifffile.imwrite(path, arr, imagej=True, resolution=(1 / VOXEL_MM, 1 / VOXEL_MM),
                     metadata={"spacing": VOXEL_MM, "unit": "mm", "axes": "ZYX"})


if __name__ == "__main__":
    for d in (GRAY_DIR, ANNOT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    calib, seeds, hists = {}, {}, {}
    for folder, info in SAMPLES.items():
        sid = info["id"]
        vol = load_cutout(folder, sid)
        pore, matrix, noise, centres, hist = calibrate(vol)
        calib[sid] = {"pore_peak": round(pore), "matrix_peak": round(matrix),
                      "span": round(matrix - pore), "pore_noise_sd": round(noise)}
        hists[sid] = hist
        ann = np.zeros(vol.shape, np.uint8)
        for z in ANNOTATED:
            ann[z] = seed_slice(vol[z], pore, matrix)
        save_imagej(GRAY_DIR / f"{sid}.tif", vol)
        save_imagej(ANNOT_DIR / f"{sid}.tif", ann)
        meta_path = META_DIR / f"{sid}.json"
        if meta_path.exists():                       # FAIR record: document the derived data
            rec = json.load(open(meta_path, encoding="utf-8"))
            rec["derived_data"] = {
                "cutout": {"slices": [FIRST_SLICE, FIRST_SLICE + N_SLICES - 1],
                           "crop_px": [CROP, CROP], "position": "centre of the slice",
                           "file": f"data/grayscale/{sid}.tif"},
                "grey_value_calibration": {"air_filled_pores": round(pore), "soil_matrix": round(matrix),
                                           "definition": "n = (grey - air) / (matrix - air)"},
                "annotations": {"file": f"data/annotations/{sid}.tif", "slices": ANNOTATED,
                                "labels": {str(k): v for k, v in LABELS.items()},
                                "status": "seed annotations, to be reviewed in napari"},
            }
            json.dump(rec, open(meta_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        on = ann[ANNOTATED]
        seeds[sid] = {"seed_voxels": {LABELS[k]: int((on == k).sum()) for k in CLASSES},
                      "annotated_fraction_of_slices": round(float((on > 0).mean()), 3)}
        print(f"{sid}: pore {pore:.0f}, matrix {matrix:.0f} | seeds {seeds[sid]['seed_voxels']}", flush=True)

    np.savez(RESULTS_DIR / "histograms.npz", centres=centres, **hists)
    json.dump(calib, open(RESULTS_DIR / "calibration.json", "w"), indent=2)
    json.dump({"bands_calibrated": {LABELS[k]: v for k, v in BANDS.items()}, "samples": seeds},
              open(RESULTS_DIR / "seed_summary.json", "w"), indent=2)
    dataset_info = {
        "TaskID": 901,
        "DatasetName": "LausanneUrbanSoils",
        "norm_type": "zscore",
        "labels": {str(k): v for k, v in LABELS.items()},
        "colors": {str(k): v for k, v in COLORS.items()},
    }
    with open(ROOT / "dataset_info.json", "w") as fh:
        json.dump(dataset_info, fh, indent=4)
    print("calibration.json, seed_summary.json and dataset_info.json written")
