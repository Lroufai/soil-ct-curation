"""
01_metadata.py
==============
One machine-readable metadata record per scan (FAIR: findable, accessible, interoperable,
reusable), built from the scanner's own reconstruction file (unireconstruction.xml) and from
the sampling information of the URBA-SOIL campaign. Also writes a human-readable dataset card.

Output: metadata/<sample_id>.json and metadata/DATASET_CARD.md
"""
import json
import xml.etree.ElementTree as ET

import numpy as np
import tifffile

from config import META_DIR, RAW_DIR, SAMPLES, VOXEL_MM


def read_reconstruction(xml_path):
    root = ET.parse(xml_path).getroot()
    cb = root.find("conebeam")
    vol = cb.find("volume")
    acq = cb.find("acquisitioninfo")
    rec = cb.find("reconstruct")
    geo = acq.find("geometry").attrib
    sdd, sod = float(geo["sdd"]), float(geo["sod"])
    return {
        "software": "XactRevision " + root.attrib.get("xactrevision", "").strip(),
        "tube_voltage_kV": float(acq.find("xray").attrib["voltage"]),
        "tube_current_uA": float(acq.find("xray").attrib["current"]),
        "source_detector_distance_mm": sdd,
        "source_object_distance_mm": sod,
        "geometric_magnification": round(sdd / sod, 3),
        "n_projections": int(cb.find("profile").attrib["images"]),
        "frames_averaged_per_projection": int(cb.find("profile/turnParam/avr").attrib["Imager"]),
        "reconstruction": rec.attrib["methode"],
        "reconstruction_filter": rec.attrib["filter"],
        "ring_artefact_filter": cb.find("ring_filter").attrib["ringEnabled"] == "1",
        "voxel_size_mm": float(vol.find("voxelSize").attrib["X"]),
        "volume_size_voxels_xyz": [int(vol.find("size").attrib[k]) for k in ("X", "Z", "Y")],
    }


def build_record(folder, info, recon):
    slices = sorted(p for p in (RAW_DIR / folder).glob("slice*.tif"))
    first = tifffile.imread(slices[0])
    nx, nz, ny = recon["volume_size_voxels_xyz"]
    return {
        "identifier": info["id"],
        "title": f"X-ray micro-CT of an urban park soil core, {info['park']}, Lausanne (Switzerland)",
        "description": ("Undisturbed topsoil core (0-10 cm) from a grassland in a public park, "
                        "scanned by cone-beam X-ray micro-CT. 16-bit reconstruction, "
                        "isotropic voxel of 20 micrometres."),
        "keywords": ["soil structure", "X-ray computed tomography", "urban soil", "macropores",
                     "image segmentation"],
        "creators": [{"name": "Intchi Roufai, Abdoul Latif", "role": "sampling, image processing, curation"}],
        "contributors": [
            {"name": "Taureg, Albert", "role": "DataCollector", "affiliation": "EPFL, PIXE laboratory"},
            {"name": "Perrenoud, Gary", "role": "DataCollector", "affiliation": "EPFL, PIXE laboratory"},
        ],
        "project": "URBA-SOIL (University of Neuchatel, HEIA-FR, EPFL)",
        "sample": {
            "site": info["park"], "site_code": info["code"], "city": "Lausanne", "country": "CH",
            "land_use": "urban park, grassland", "depth_cm": [0, 10],
            "core": "PVC cylinder, 5 cm diameter, 10 cm height",
            "sampling_permit": "Service des parcs et domaines (SPADOM), City of Lausanne",
        },
        "acquisition": {"date": "2023-05-22", "facility": "EPFL, PIXE laboratory", **recon},
        "files": {
            "archive": f"{info['archive']}_230522_{folder}_Recon.7z",
            "format": "TIFF, one file per slice, axis order of files = Y (vertical)",
            "dtype": str(first.dtype),
            "slice_shape_px": list(first.shape),
            "n_slices_full_volume": ny,
            "physical_size_mm": [round(n * VOXEL_MM, 2) for n in (nx, nz, ny)],
        },
        "related_measurements": {
            "laboratory_total_porosity_percent": info["lab_porosity"],
            "ct_void_fraction_percent_thesis": info["thesis_ct_void"],
            "source": "Intchi Roufai (2023), MSc thesis, University of Neuchatel, Table 11",
        },
        "license_proposed": "CC BY 4.0, subject to agreement of all data owners",
        "care_principles": "No Indigenous data; public land sampled under municipal permit.",
    }


if __name__ == "__main__":
    META_DIR.mkdir(exist_ok=True)
    rows = []
    for folder, info in SAMPLES.items():
        xml_path = RAW_DIR / folder / "unireconstruction.xml"
        if not xml_path.exists():
            print(f"skip {folder}: no reconstruction file yet")
            continue
        rec = build_record(folder, info, read_reconstruction(xml_path))
        with open(META_DIR / f"{info['id']}.json", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        a = rec["acquisition"]
        rows.append(f"| {info['id']} | {info['park']} | {a['tube_voltage_kV']:.0f} kV / {a['tube_current_uA']:.0f} uA "
                    f"| {a['n_projections']} | {a['voxel_size_mm'] * 1000:.2f} um "
                    f"| {' x '.join(str(v) for v in a['volume_size_voxels_xyz'])} "
                    f"| {info['thesis_ct_void']:.1f} | {info['lab_porosity']:.1f} |")
        print(f"{info['id']}: {a['tube_voltage_kV']:.0f} kV, {a['n_projections']} projections, "
              f"voxel {a['voxel_size_mm'] * 1000:.2f} um")

    card = [
        "# Dataset card: X-ray micro-CT of six urban park soils, Lausanne",
        "",
        "Six undisturbed topsoil cores (0-10 cm) from grasslands in public parks of Lausanne, Switzerland, "
        "collected for the URBA-SOIL project and scanned by cone-beam X-ray micro-CT at the EPFL PIXE laboratory "
        "on 22 May 2023. Urban park topsoils are often reworked and rich in coarse fragments, "
        "a soil type that broadens training datasets built mostly on agricultural soils.",
        "",
        "| Sample | Site | Source | Projections | Voxel | Volume (x y z) | CT void fraction, thesis (%) | Lab porosity (%) |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "Material classes annotated: matrix (fine earth), rocks (coarse grains), pores. "
        "Label 0 = not annotated (ToPredict), following the nnUNet4SoilXrayCT convention. "
        "Grey values are calibrated per scan on two internal references (air-filled pores = 0, "
        "soil matrix = 1); the reference values of each scan are stored in its record under derived_data.",
        "",
        "Raw reconstructions: 16-bit TIFF slices, about 9 GB per sample, archived at EPFL. "
        "Proposed licence CC BY 4.0, subject to agreement of all data owners.",
    ]
    (META_DIR / "DATASET_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} records written to {META_DIR}")
