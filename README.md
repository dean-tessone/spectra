# SPECTRA

SPECTRA is a weakly supervised transformer that learns from all rare circulating cells imaged in a patient's blood sample. It is trained only on patient diagnosis (cancer or non-cancer). Attribution then traces each prediction back to the individual cells that carry it.

The input is fluid-phase whole-slide immunofluorescence of peripheral blood, stained for DAPI, cytokeratin (CK), CD45/CD31 and vimentin. Each slide is reduced to its 10,000 most anomalous events (fewer on 7 slides). Each event is described by a 512-dimensional image embedding from a self-supervised encoder (BLUE v2). SPECTRA reads these events as one bag per patient.

This repository contains the code and final figures for the paper. The data are on Zenodo (see [Data](#data)).

## Cohort

- **Primary analysis:** 2,550 patients with 13 cancer types or non-cancer reference controls.
  - Training: 1,793 patients (1,689 cancers, 104 controls).
  - Held-out test: 757 patients (713 cancers, 44 controls).
- **Reserved controls:** 103 further controls (benign lung nodule work-up and BRCA carriers without cancer). These are used only for the disease-specific adaptation analyses.

## Data

Zenodo: DOI to be added.

| File | Content |
|---|---|
| `samples.csv` | One row per slide: `slide_id`, `tube_id` (first five characters of `slide_id`), `cancer_type` (`Control` for non-cancer), `split` (train/test), `cv_fold` (training folds 0 to 2; -1 for test), `primary_group` (the tube used for the primary endpoint), `population` (cancer, reference control, lung work-up control, BRCA-carrier control) |
| `events_*.parquet` | One row per event: `slide_id`, `tube_id`, `frame_id`, `cell_id`, `cell_type`, `wbcclass`, `junk_score`, `model_input`, `embedding` (512 float32) |
| `cell_types.csv` | The 10 cell-type classes |

Column definitions:

- **`cell_type`:** the class assigned by the frozen BLUE v2 phenotype classifier, a logistic regression on the embedding.
- **`wbcclass`:** the leukocyte classifier score.
- **`junk_score`:** the image-quality model's probability that the event is unusable.
- **`model_input`:** whether the event passed the three filters that select SPECTRA's input cells:
  - `junk_score` ≤ 0.465;
  - `wbcclass` < 0.9;
  - mean CK, CD45/CD31 or vimentin intensity ≥ 5,000.

The release contains no patient identifiers, no clinical information other than cancer type, and no raw images.

## Repository layout

| Folder | Content |
|---|---|
| `src/spectra/` | Library: data loading, bag construction, normalization, evaluation |
| `model/architectures/` | SPECTRA transformer; ABMIL, additive MIL, mean pooling, CLAM-SB and CLAM-MB baselines; LoRA |
| `model/pipeline/` | Cohort preparation, fitting-only normalizers, training runtime and source fits |
| `model/training_cv/` | Three-fold training cross-validation and the final three-seed refit |
| `model/depth_comparison/` | Three- versus six-layer comparison |
| `model/test_evaluation/` | Single evaluation of the locked model on the held-out test set |
| `model/bootstrap_ci/`, `model/batch_effects/`, `model/allgroup_oof/` | Confidence intervals, scanner and batch checks, all-tube supplementary endpoint |
| `robustness/` | Study, site and acquisition checks, leukocyte negative control, cell and cluster removal reporting, cluster 10 image features, CK-gate grid and permutation tests (revision analyses) |
| `baselines/` | Matched MIL baselines, cell-count baseline, CK-gate and CTC-like count comparators |
| `attribution/` | AttnLRP attribution, Leiden atlas of attributed cells, cell removal, attribution sanity checks |
| `adaptation/` | Disease-specific adaptation (lung nodules, BRCA carriers) |
| `validation/` | Imaging mass cytometry and single-cell copy-number analyses of attributed cells |
| `imaging/` | Cell-image extraction and display transform |
| `cohort/` | Control-population policy and manuscript tables |
| `data_release/` | Export of the Zenodo release |
| `figures/` | Final figures (`main/`, `supplementary/`; PDF, SVG and PNG), legends and figure code (`code/`) |
| `manuscript/` | LaTeX source of the manuscript and Supplementary Information (`make` builds the PDF) |

Each analysis folder writes its outputs to `runs/` and `analyses/`. These folders are not part of the repository.

## Installation

```bash
pip install -e .
```

Python 3.10 or later. Training used one NVIDIA GPU.

## Configuration

Machine-specific locations are read from environment variables:

| Variable | Used for |
|---|---|
| `SPECTRA_DATA_ROOT` | Per-slide embedding, junk-score and source caches |
| `SPECTRA_SLIDE_TABLE` | Slide table |
| `SPECTRA_10X_ROOT`, `SPECTRA_40X_ROOT` | Raw 10x and 40x image frames (cell images only) |
| `SPECTRA_IMC_ROOT`, `OPENIMC` | Imaging mass cytometry acquisitions and the OpenIMC executable |
| `SPECTRA_CNV_H5` | Single-cell copy-number profiles |
| `BLUEV2_CLASSIFIER`, `BLUEV2_TRAINING_SHEET` | Frozen BLUE v2 phenotype classifier and its training-slide list |
| `SPECTRA_RELEASE_DIR`, `SPECTRA_SHARED_OUT` | Output folders for the data release and report copies |

The scripts are the code used for the paper. They expect the per-slide caches that the Zenodo release was exported from. To start from the release, map the `embedding`, `junk_score` and `wbcclass` columns of each slide onto that layout.

## Citation

To be added on publication.
