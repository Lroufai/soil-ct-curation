# Dataset card: X-ray micro-CT of six urban park soils, Lausanne

Six undisturbed topsoil cores (0-10 cm) from grasslands in public parks of Lausanne, Switzerland, collected for the URBA-SOIL project and scanned by cone-beam X-ray micro-CT at the EPFL PIXE laboratory on 22 May 2023. Urban park topsoils are often reworked and rich in coarse fragments, a soil type that broadens training datasets built mostly on agricultural soils.

| Sample | Site | Source | Projections | Voxel | Volume (x y z) | CT void fraction, thesis (%) | Lab porosity (%) |
|---|---|---|---|---|---|---|---|
| LAU-VTHE | Vidy Theatre | 80 kV / 250 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 21.3 | 44.0 |
| LAU-ESPM | Esplanade de Montbenon | 80 kV / 250 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 45.4 | 36.2 |
| LAU-ERA | Eracom | 80 kV / 249 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 17.4 | 42.1 |
| LAU-OLY | Olympique | 80 kV / 249 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 23.1 | 44.0 |
| LAU-D2 | Denantou 2 | 80 kV / 250 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 13.8 | 49.1 |
| LAU-SIG | Signal Playground | 80 kV / 250 uA | 1312 | 20.00 um | 1650 x 1650 x 1900 | 14.3 | 47.1 |

Material classes annotated: matrix (fine earth), rocks (coarse grains), pores. Label 0 = not annotated (ToPredict), following the nnUNet4SoilXrayCT convention. Grey values are calibrated per scan on two internal references (air-filled pores = 0, soil matrix = 1); the reference values of each scan are stored in its record under derived_data.

Raw reconstructions: 16-bit TIFF slices, about 9 GB per sample, archived at EPFL. Proposed licence CC BY 4.0, subject to agreement of all data owners.
