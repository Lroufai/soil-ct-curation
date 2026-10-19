"""
04_calibration_effect.py
========================
What the per-scan calibration changes. The same pipeline (seed rule + pooled random forest) is
run once more WITHOUT calibration: one common grey-value scale and one common set of thresholds
for all six scans, read from the pooled histogram (matrix peak m, width sigma):
    Pores  s < m - 3 sigma     Matrix  |s - m| < sigma     Rocks  s > m + 3 sigma
The pore fractions of both variants are compared with the thesis segmentation (Avizo,
thresholds set by eye for each scan), which serves as the independent reference.

Output: results/calibration_effect.json
"""
import importlib
import json

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import multiscale_basic_features
from skimage.morphology import disk, remove_small_objects
from sklearn.ensemble import RandomForestClassifier

from config import ANNOTATED, CLASSES, GRAY_DIR, RESULTS_DIR, SAMPLES

rf03 = importlib.import_module("03_segment_random_forest")
PROFILE, RF_KW, PER_CLASS = rf03.PROFILE, rf03.RF_KW, rf03.PER_CLASS


def pooled_peak(vols):
    hist = np.zeros(1024)
    for vol in vols:
        s = ndi.gaussian_filter(vol[::10].astype(np.float32), (0, 1.5, 1.5))
        h, edges = np.histogram(s, bins=1024, range=(0, 65535))
        hist += h / h.sum()
    hist = ndi.gaussian_filter1d(hist, 2)
    c = 0.5 * (edges[1:] + edges[:-1])
    w = (c > 28000) & (c < 42000)
    i = int(np.flatnonzero(w)[np.argmax(hist[w])])
    half = hist[i] / 2
    left = i - np.argmax(hist[i::-1] < half)
    right = i + np.argmax(hist[i:] < half)
    hwhm = min(c[i] - c[left], c[right] - c[i])
    return float(c[i]), float(hwhm / 1.1774)


def common_seeds(img, m, sigma):
    s = ndi.gaussian_filter(img.astype(np.float32), 1.5)
    out = np.zeros(img.shape, np.uint8)
    for k, msk in {3: s < m - 3 * sigma, 1: np.abs(s - m) < sigma, 2: s > m + 3 * sigma}.items():
        msk = remove_small_objects(ndi.binary_erosion(msk, structure=disk(2)), max_size=20)
        out[msk] = k
    return out


def feats(img, m):
    return multiscale_basic_features(img.astype(np.float32) / m, intensity=True, edges=True,
                                     texture=True, sigma_min=1, sigma_max=8, num_sigma=4)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vols = {i["id"]: tifffile.imread(GRAY_DIR / f"{i['id']}.tif") for i in SAMPLES.values()}
    m, sigma = pooled_peak(vols.values())
    X, y = [], []
    for sid, vol in vols.items():
        F = [feats(vol[z], m) for z in ANNOTATED]
        L = [common_seeds(vol[z], m, sigma) for z in ANNOTATED]
        for k in CLASSES:
            f = np.concatenate([a[b == k] for a, b in zip(F, L)])
            if len(f):
                idx = rng.choice(len(f), size=min(PER_CLASS, len(f)), replace=False)
                X.append(f[idx]); y.append(np.full(len(idx), k))
    rf = RandomForestClassifier(**RF_KW).fit(np.concatenate(X), np.concatenate(y))

    calibrated = json.load(open(RESULTS_DIR / "rf_report.json"))["volume_fractions"]
    out = {"common_scale": {"matrix_peak": round(m), "sigma": round(sigma),
                            "pore_threshold": round(m - 3 * sigma)}, "samples": {}}
    for info in SAMPLES.values():
        sid = info["id"]
        pores = []
        for z in PROFILE:
            f = feats(vols[sid][z], m)
            lab = rf.predict(f.reshape(-1, f.shape[-1]))
            pores.append(float((lab == 3).mean()))
        out["samples"][sid] = {
            "pores_without_calibration": round(float(np.mean(pores)), 4),
            "pores_with_calibration": calibrated[sid]["pores"],
            "thesis_ct_void_fraction": info["thesis_ct_void"] / 100,
        }
        print(sid, out["samples"][sid], flush=True)
    json.dump(out, open(RESULTS_DIR / "calibration_effect.json", "w"), indent=2)
