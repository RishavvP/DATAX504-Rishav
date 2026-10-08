# Nepal Flood Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishavvP/DATAX504-Rishav/blob/main/final-project/nepal-flood-segmentation/01_flood_segmentation.ipynb)

Pixel-level flood mapping from Sentinel-1 radar and Sentinel-2 optical satellite imagery with Keras U-Nets,
trained on the open **Sen1Floods11** dataset and applied to a Nepal monsoon flood as a case study.

Final project for Practical Machine Learning.

## What's here

| File | Purpose |
|---|---|
| `01_flood_segmentation.ipynb` | The whole project: download → explore → baselines → U-Net → training → evaluation → Nepal case study |
| `results/` | The notebook as run on Colab, with all outputs ([view it here](https://nbviewer.org/github/RishavvP/DATAX504-Rishav/blob/main/final-project/nepal-flood-segmentation/results/01_flood_segmentation_executed.ipynb), since GitHub may not render an 8 MB notebook), `results.csv` and the report figures |
| `REPORT.md` | The written report (submitted as a PDF on Moodle) |
| `requirements.txt` | Packages for running locally (Colab already has most of them) |

## Running it

**Real training (Google Colab, free GPU):**
1. Click **Open in Colab** above and pick *Runtime → Change runtime type → T4 GPU*.
2. *Runtime → Run all*. Allow Google Drive access when asked. The dataset (about 1.75 GB) downloads to the Colab machine each session; checkpoints and results go to `MyDrive/nepal-flood-segmentation/`.
3. If Colab disconnects, reconnect and *Run all* again. Training resumes from the last epoch; finished experiments are skipped.
4. For the Nepal case study, export the two Sentinel-1 scenes with the Earth Engine code in notebook §16, put them in `MyDrive/nepal-flood-segmentation/nepal/`, and *Run all* again.

Full training of all 9 experiments takes about 8–10 GPU hours on a T4 (about 4–5 on an L4).

**Smoke test (any laptop, ~5 minutes, synthetic data):** checks that every cell runs end to end. Run from this folder (`cd final-project/nepal-flood-segmentation`).
```bash
pip install -r requirements.txt
FLOOD_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute 01_flood_segmentation.ipynb --output smoke_run.ipynb
```
(PowerShell: `$env:FLOOD_SMOKE_TEST="1"` before the command.)

## Method in one paragraph

Each 512×512 chip has 2 radar bands (VV, VH) and 13 optical bands, with a hand-drawn water mask (`1` water, `0` dry, `-1` unlabelled).
A U-Net is trained on random 256×256 crops with flip/rotation augmentation, a masked BCE + Dice loss and a custom masked IoU metric,
for up to 300 epochs with early stopping on validation IoU. Experiments vary one design choice at a time (input bands, per-pixel MLP vs U-Net,
loss, regularisation, normalisation, random seed) and are compared against three non-learned baselines (all-dry, Otsu threshold on VH, NDWI > 0). Models are ranked by **water-class IoU** on the official test split
and on the held-out Bolivia event. The radar-only model is then run on before/after Sentinel-1 scenes of a Nepal flood to map newly flooded area.

## Results

Water IoU, thresholds chosen on validation. The reference U-Net is the mean ± std of 3 seeds. Full table in `REPORT.md`.

| Method | Test IoU | Bolivia IoU (unseen event) |
|---|---|---|
| All dry | 0.000 | 0.000 |
| Otsu (S1 VH) | 0.229 | 0.384 |
| NDWI > 0 (S2) | 0.758 | 0.832 |
| Pixel MLP (S1+S2) | 0.868 | 0.814 |
| U-Net S1 | 0.667 | 0.693 |
| U-Net S2 | 0.847 | 0.808 |
| U-Net S1+S2 | 0.854 ± 0.001 | 0.812 ± 0.012 |

**Nepal case study:** 34.94 km² newly flooded on the Koshi River floodplain (19 Sept → 1 Oct 2024).

![Nepal flood map](results/figures/06_nepal_flood_map.png)

## Data and citation

Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1.* CVPR Workshops. Data: `gs://sen1floods11`, code: https://github.com/cloudtostreet/Sen1Floods11

Sentinel-1/2 imagery: Copernicus Programme (ESA). Nepal scenes exported from Google Earth Engine (`COPERNICUS/S1_GRD`).
