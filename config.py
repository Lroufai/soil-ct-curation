"""Shared settings: samples, paths, classes and the slice range used for the cut-outs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Raw reconstructions (one folder of 16-bit TIFF slices per sample). Not part of the repository.
RAW_DIR = Path(os.environ.get("SOILCT_RAW", ROOT / "raw"))
OUT = ROOT / "data"                    # derived data (cut-outs, annotations), not versioned
GRAY_DIR = OUT / "grayscale"           # nnUNet4SoilXrayCT convention: <sample_id>.tif
ANNOT_DIR = OUT / "annotations"        # same name, label 0 = ToPredict
META_DIR = ROOT / "metadata"           # versioned: one JSON record per sample + dataset card
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

VOXEL_MM = 0.0199957                   # isotropic voxel size from unireconstruction.xml

# Six urban park soils, Lausanne (Switzerland), 0-10 cm depth, URBA-SOIL project.
# CT void fraction and laboratory total porosity: MSc thesis (Intchi Roufai, 2023), Table 11.
SAMPLES = {
    "Vidy-theatre": {"id": "LAU-VTHE", "code": "Vthe", "archive": "ID0267", "park": "Vidy Theatre",
                     "lab_porosity": 44.0, "thesis_ct_void": 21.33},
    "Montbenon":    {"id": "LAU-ESPM", "code": "EspM", "archive": "ID0264", "park": "Esplanade de Montbenon",
                     "lab_porosity": 36.2, "thesis_ct_void": 45.44},
    "Eracom":       {"id": "LAU-ERA",  "code": "Era",  "archive": "ID0263", "park": "Eracom",
                     "lab_porosity": 42.1, "thesis_ct_void": 17.41},
    "Olympique":    {"id": "LAU-OLY",  "code": "Oly",  "archive": "ID0265", "park": "Olympique",
                     "lab_porosity": 44.0, "thesis_ct_void": 23.13},
    "Denantou2":    {"id": "LAU-D2",   "code": "D2",   "archive": "ID0266", "park": "Denantou 2",
                     "lab_porosity": 49.1, "thesis_ct_void": 13.85},
    "Signal":       {"id": "LAU-SIG",  "code": "Sigjeu", "archive": "ID0268", "park": "Signal Playground",
                     "lab_porosity": 47.1, "thesis_ct_void": 14.27},
}

# Three material classes. Label 0 is "ToPredict" (not annotated) and label 1 the soil matrix,
# as in the nnUNet4SoilXrayCT convention.
LABELS = {0: "ToPredict", 1: "Matrix", 2: "Rocks", 3: "Pores"}
COLORS = {0: [0, 0, 0], 1: [115, 0, 102], 2: [255, 255, 0], 3: [0, 122, 153]}
CLASSES = (1, 2, 3)

# Cut-out: 300 consecutive slices (the size used for ground truth in nnUNet4SoilXrayCT),
# central 512 x 512 pixels (10.2 x 10.2 mm), fully inside the reconstructed field of view.
FIRST_SLICE = 1577
N_SLICES = 300
CROP = 512
# Slices of the cut-out that receive sparse seed annotations, and slices held out for validation
ANNOTATED = [20, 80, 140, 200, 260]
HELD_OUT = [50, 110, 170, 230, 290]
