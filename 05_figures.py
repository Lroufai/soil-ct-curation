"""
05_figures.py
=============
Quality-control figures for the README:
  1. calibration.png      : grey-value histograms of the six scans before and after the
                            two-point calibration (air-filled pores = 0, matrix = 1);
  2. qc_segmentation.png  : for each sample, a CT slice, the seed annotations, the random-forest
                            classes and the uncertainty map (where manual annotation should go next);
  3. qc_pore_fraction.png : pore fraction with and without calibration, next to the thesis
                            segmentation and the laboratory total porosity.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.colors import ListedColormap

import plotstyle
from config import ANNOT_DIR, ANNOTATED, CLASSES, COLORS, FIG_DIR, GRAY_DIR, LABELS, RESULTS_DIR, SAMPLES

FIG_DIR.mkdir(exist_ok=True)
plotstyle.apply()
ids = [i["id"] for i in SAMPLES.values()]
names = {i["id"]: i["park"] for i in SAMPLES.values()}
calib = json.load(open(RESULTS_DIR / "calibration.json"))
rf = json.load(open(RESULTS_DIR / "rf_report.json"))
effect = json.load(open(RESULTS_DIR / "calibration_effect.json"))["samples"]
H = np.load(RESULTS_DIR / "histograms.npz")
class_cmap = ListedColormap([np.array(COLORS[k]) / 255 for k in CLASSES])
seed_cmap = ListedColormap([(0, 0, 0, 0)] + [tuple(np.array(COLORS[k]) / 255) + (0.85,) for k in CLASSES])

# ------------------------------------------------------------ 1. calibration
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
c = H["centres"]
for sid in ids:
    h, cal = H[sid], calib[sid]
    a1.plot(c, h, color=plotstyle.INK_2, lw=1)
    for key, col in (("pore_peak", plotstyle.C1), ("matrix_peak", plotstyle.C2)):
        i = np.argmin(np.abs(c - cal[key]))
        a1.plot(c[i], h[i], "o", color=col, ms=6, mec=plotstyle.SURFACE, mew=1.2)
    n = (c - cal["pore_peak"]) / cal["span"]
    a2.plot(n, h, color=plotstyle.INK_2, lw=1)
a1.plot([], [], "o", color=plotstyle.C1, label="air-filled pores")
a1.plot([], [], "o", color=plotstyle.C2, label="soil matrix")
a1.legend(loc="upper right")
a1.set_xlim(10000, 60000)
a1.set_xlabel("Grey value (16-bit)")
a1.set_ylabel("Share of voxels")
a1.set_title("Six scans as reconstructed: the peaks drift")
a2.axvline(0, color=plotstyle.C1, lw=1.2, ls="--")
a2.axvline(1, color=plotstyle.C2, lw=1.2, ls="--")
a2.set_xlim(-0.8, 2.6)
a2.set_xlabel("Calibrated grey value n (0 = air, 1 = matrix)")
a2.set_title("After two-point calibration: one common scale")
fig.savefig(FIG_DIR / "calibration.png")
plt.close(fig)

# ------------------------------------------------------------ 2. segmentation panel
fig, axes = plt.subplots(4, len(ids), figsize=(2.6 * len(ids), 10.2))
fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.06, wspace=0.06, hspace=0.06)
for j, sid in enumerate(ids):
    pr = np.load(RESULTS_DIR / "predictions" / f"{sid}_slices.npz")
    ann = tifffile.imread(ANNOT_DIR / f"{sid}.tif", key=ANNOTATED[2])
    grey = tifffile.imread(GRAY_DIR / f"{sid}.tif", key=ANNOTATED[2])
    cal = calib[sid]
    lo, hi = cal["pore_peak"] - 0.3 * cal["span"], cal["matrix_peak"] + 1.3 * cal["span"]
    axes[0, j].imshow(pr["image"], cmap="gray", vmin=lo, vmax=hi)
    axes[0, j].set_title(names[sid], fontsize=10)
    axes[1, j].imshow(grey, cmap="gray", vmin=lo, vmax=hi)
    axes[1, j].imshow(ann, cmap=seed_cmap, vmin=0, vmax=len(CLASSES), interpolation="nearest")
    axes[2, j].imshow(pr["label"] - 1, cmap=class_cmap, vmin=0, vmax=len(CLASSES) - 1, interpolation="nearest")
    im = axes[3, j].imshow(1 - pr["pmax"].astype(float), cmap="magma", vmin=0, vmax=0.6)
    for a in axes[:, j]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
rows = ["CT slice (10.2 mm wide)", "Seed annotations", "Random forest classes", "Uncertainty (1 - p max)"]
for i, r in enumerate(rows):
    axes[i, 0].set_ylabel(r, fontsize=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(COLORS[k]) / 255) for k in CLASSES]
fig.legend(handles, [LABELS[k] for k in CLASSES], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
cax = fig.add_axes([0.935, 0.08, 0.012, 0.19])
fig.colorbar(im, cax=cax, label="uncertainty")
fig.suptitle("Six urban park soils, one shared classifier: seeds, classes and uncertainty", y=0.985)
fig.savefig(FIG_DIR / "qc_segmentation.png")
plt.close(fig)

# ------------------------------------------------------------ 3. pore fraction comparison
vf = rf["volume_fractions"]
series = [
    ("Random forest, calibrated", [vf[s]["pores"] * 100 for s in ids], plotstyle.C1),
    ("Random forest, no calibration", [effect[s]["pores_without_calibration"] * 100 for s in ids], plotstyle.C2),
    ("Thesis segmentation (Avizo, by eye)", [vf[s]["thesis_ct_void_fraction"] * 100 for s in ids], plotstyle.C3),
    ("Laboratory total porosity", [vf[s]["lab_total_porosity"] * 100 for s in ids], plotstyle.C4),
]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
x = np.arange(len(ids))
w = 0.2
mm = np.array([vf[s]["pores_min_max_over_slices"] for s in ids]) * 100
for i, (lab, vals, col) in enumerate(series):
    bars = ax.bar(x + (i - 1.5) * w, vals, width=w - 0.025, color=col, label=lab)
    for j, (b, v) in enumerate(zip(bars, vals)):
        top = mm[j, 1] if i == 0 else v          # calibrated bars carry the slice range
        ax.text(b.get_x() + b.get_width() / 2, top + 0.8, f"{v:.0f}", ha="center", va="bottom",
                fontsize=7.5, color=plotstyle.INK_2)
mean = np.array(series[0][1])
ax.errorbar(x - 1.5 * w, mean, yerr=[mean - mm[:, 0], mm[:, 1] - mean], fmt="none",
            ecolor=plotstyle.INK, elinewidth=1, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels([names[s] for s in ids], fontsize=9)
ax.set_ylabel("Pore fraction (%)")
ax.set_title("Pore fraction of each cut-out, with the range over its slices")
ax.legend(loc="upper right", fontsize=8.5, ncol=2)
ax.set_ylim(0, 82)
fig.savefig(FIG_DIR / "qc_pore_fraction.png")
plt.close(fig)
print("figures written to", FIG_DIR)
