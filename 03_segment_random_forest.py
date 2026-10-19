"""
03_segment_random_forest.py
===========================
Interactive-style shallow machine learning segmentation (the approach of ilastik, labkit or
Weka): a random forest classifies every voxel from multi-scale filter responses (smoothed
intensity, edges, texture), trained on the sparse seed annotations.

Quality control, without claiming accuracy against seeds (seeds come from an intensity rule,
so a classifier reproduces them trivially; true accuracy needs manual annotations):
  1. transfer between soils: a classifier trained on all samples and one trained without the
     sample are compared voxel by voxel on five held-out slices, over all voxels and over the
     difficult ones (boundaries, partial volume). This is the question a generic classifier
     must answer;
  2. volume fractions on 30 slices of each cut-out, compared with the thesis segmentation,
     and the share of uncertain voxels (class probability below 0.7), which shows where manual
     annotation in napari should go next.

Output: results/rf_report.json, results/predictions/<sample_id>_slices.npz
"""
import importlib
import json
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import multiscale_basic_features
from skimage.morphology import disk, remove_small_objects
from sklearn.ensemble import RandomForestClassifier

from config import ANNOT_DIR, ANNOTATED, CLASSES, GRAY_DIR, HELD_OUT, RESULTS_DIR, SAMPLES

seed_slice = importlib.import_module("02_prepare_annotations").seed_slice

PER_CLASS = 15_000          # training voxels drawn per class and per sample
PROFILE = list(range(5, 300, 10))   # 30 slices used for volume fractions
UNCERTAIN = 0.7
RF_KW = dict(n_estimators=60, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=0)


def features(img, cal):
    """Multi-scale filter bank on one slice, on calibrated grey values (0 = air, 1 = matrix),
    so that one classifier can be shared by all scans."""
    n = (img.astype(np.float32) - cal["pore_peak"]) / cal["span"]
    return multiscale_basic_features(n, intensity=True, edges=True, texture=True,
                                     sigma_min=1, sigma_max=8, num_sigma=4)


def sample_training(feat_slices, lab_slices, rng):
    X, y = [], []
    for k in CLASSES:
        f = np.concatenate([fs[ls == k] for fs, ls in zip(feat_slices, lab_slices)])
        if len(f) == 0:
            continue
        idx = rng.choice(len(f), size=min(PER_CLASS, len(f)), replace=False)
        X.append(f[idx]); y.append(np.full(len(idx), k))
    return np.concatenate(X), np.concatenate(y)




if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(0)
    calib = json.load(open(RESULTS_DIR / "calibration.json"))
    ids = [i["id"] for i in SAMPLES.values() if (GRAY_DIR / f"{i['id']}.tif").exists()]
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    data = {}
    for sid in ids:
        vol = tifffile.imread(GRAY_DIR / f"{sid}.tif")
        ann = tifffile.imread(ANNOT_DIR / f"{sid}.tif")
        cal = calib[sid]
        data[sid] = {
            "train_f": [features(vol[z], cal) for z in ANNOTATED],
            "train_l": [ann[z] for z in ANNOTATED],
            "test_f": [features(vol[z], cal) for z in HELD_OUT],
            "test_l": [seed_slice(vol[z], cal["pore_peak"], cal["matrix_peak"]) for z in HELD_OUT],
            "vol": vol, "cal": cal,
        }
        print(f"features ready: {sid} ({time.time() - t0:.0f} s)", flush=True)

    report = {"transfer_between_soils": {}, "volume_fractions": {}}

    # pooled classifier: all samples
    Xs, ys = zip(*[sample_training(d["train_f"], d["train_l"], rng) for d in data.values()])
    rf = RandomForestClassifier(**RF_KW).fit(np.concatenate(Xs), np.concatenate(ys))

    # 1. transfer between soils: with versus without the sample, all voxels of held-out slices
    for sid in data:
        others = [s for s in data if s != sid]
        Xo, yo = zip(*[sample_training(data[s]["train_f"], data[s]["train_l"], rng) for s in others])
        rf_o = RandomForestClassifier(**RF_KW).fit(np.concatenate(Xo), np.concatenate(yo))
        d = data[sid]
        F = np.concatenate([f.reshape(-1, f.shape[-1]) for f in d["test_f"]])
        hard = np.concatenate([l.ravel() for l in d["test_l"]]) == 0   # not confident by intensity
        p_with, p_without = rf.predict(F), rf_o.predict(F)
        agree = p_with == p_without
        report["transfer_between_soils"][sid] = {
            "agreement_all_voxels": round(float(agree.mean()), 4),
            "agreement_boundary_voxels": round(float(agree[hard].mean()), 4),
            "pores_with": round(float((p_with == 3).mean()), 4),
            "pores_without": round(float((p_without == 3).mean()), 4),
        }
        print(f"transfer {sid}: {report['transfer_between_soils'][sid]}", flush=True)

    # 2. volume fractions and uncertainty on 30 slices per cut-out (pooled classifier)
    for sid, d in data.items():
        fr, unc, keep = [], [], {}
        for z in PROFILE:
            f = features(d["vol"][z], d["cal"])
            p = rf.predict_proba(f.reshape(-1, f.shape[-1])).reshape(*f.shape[:2], -1)
            lab = rf.classes_[np.argmax(p, axis=-1)]
            fr.append([float((lab == k).mean()) for k in CLASSES])
            unc.append(float((p.max(axis=-1) < UNCERTAIN).mean()))
            if z == 145:
                keep = {"image": d["vol"][z], "label": lab.astype(np.uint8), "pmax": p.max(axis=-1).astype(np.float16)}
        fr = np.array(fr)
        np.savez_compressed(RESULTS_DIR / "predictions" / f"{sid}_slices.npz", z=np.array(PROFILE),
                            fractions=fr, uncertain=np.array(unc), **keep)
        report["volume_fractions"][sid] = {
            "matrix": round(float(fr[:, 0].mean()), 4), "rocks": round(float(fr[:, 1].mean()), 4),
            "pores": round(float(fr[:, 2].mean()), 4),
            "pores_min_max_over_slices": [round(float(fr[:, 2].min()), 4), round(float(fr[:, 2].max()), 4)],
            "uncertain_voxels": round(float(np.mean(unc)), 4),
            "thesis_ct_void_fraction": [i for i in SAMPLES.values() if i["id"] == sid][0]["thesis_ct_void"] / 100,
            "lab_total_porosity": [i for i in SAMPLES.values() if i["id"] == sid][0]["lab_porosity"] / 100,
        }
        print(f"fractions {sid}: {report['volume_fractions'][sid]}", flush=True)

    report["settings"] = {"features": "multiscale_basic_features, sigma 1-8, 4 scales",
                          "random_forest": RF_KW, "per_class_training_voxels": PER_CLASS,
                          "uncertainty_threshold": UNCERTAIN}
    with open(RESULTS_DIR / "rf_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"done in {time.time() - t0:.0f} s")
