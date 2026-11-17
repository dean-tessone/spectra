# SPECTRA: rare-cell embeddings from whole-slide liquid biopsy images

Per-event image embeddings for the 6,049 peripheral-blood slides analysed in the SPECTRA study.

Each slide was imaged with four-channel immunofluorescence (DAPI, cytokeratin, CD45/CD31, vimentin). It was then reduced to its 10,000 most anomalous events (fewer on 7 slides). Each event has:

- a 512-dimensional embedding from the self-supervised BLUE v2 encoder;
- a cell type from the frozen BLUE v2 phenotype classifier;
- a leukocyte score (`wbcclass`);
- an image-quality junk score.

Code: https://github.com/dean-tessone/spectra

## Files

| File | Rows | Content |
|---|---|---|
| `samples.csv` | 6,049 slides | Slide table (below) |
| `cell_types.csv` | 10 | Cell-type classes |
| `train/events_train_000.parquet` to `_021` | 43,611,627 events (4,363 slides) | Training-set events |
| `test/events_test_000.parquet` to `_008` | 16,859,584 events (1,686 slides) | Held-out test-set events |
| `MANIFEST_shards.csv` | 31 | File name, slide count, row count, size and SHA-256 of each shard |

Each shard holds 200 slides; the last shard of each split holds fewer. Each slide is one Parquet row group.

## `samples.csv`

| Column | Meaning |
|---|---|
| `slide_id` | Slide identifier |
| `tube_id` | Blood tube identifier: the first five characters of `slide_id` |
| `cancer_type` | One of 13 cancer types, or `Control` for non-cancer participants |
| `split` | `train` or `test`; assigned per patient |
| `cv_fold` | Training cross-validation fold (0, 1, 2), assigned per patient; -1 for test |
| `primary_group` | `True` for the slides used in the primary analysis: each patient's earliest eligible tube, first two slides. Each patient has exactly one primary tube. |
| `population` | `cancer`; `reference_control` (primary non-cancer controls); `lung_workup_control` (benign lung-nodule work-up); `brca_carrier_control` (BRCA carriers without cancer). The last two were used only for disease-specific adaptation. |

## `events_*.parquet`

| Column | Type | Meaning |
|---|---|---|
| `slide_id`, `tube_id` | string | As above |
| `frame_id`, `cell_id` | int32 | Event location: image frame and cell number within the slide |
| `cell_type` | string | Class from the frozen BLUE v2 phenotype classifier (logistic regression on the embedding) |
| `wbcclass` | float32 | Leukocyte classifier score (0 to 1) |
| `junk_score` | float32 | Image-quality model's probability that the event is unusable (0 to 1) |
| `model_input` | bool | Event passed all three filters for SPECTRA's input: `junk_score` ≤ 0.46509784932900855; `wbcclass` < 0.9; mean cytokeratin, CD45/CD31 or vimentin intensity ≥ 5,000 |
| `embedding` | list of 512 float32 | BLUE v2 embedding |

Reading one shard in Python:

```python
import pyarrow.parquet as pq
t = pq.read_table('test/events_test_000.parquet')
X = t['embedding'].combine_chunks().flatten().to_numpy().reshape(-1, 512)
meta = t.drop(['embedding']).to_pandas()
```

## Privacy

The release contains no patient identifiers, no clinical data other than cancer type, and no images. Slides and tubes are identified only by laboratory slide and tube numbers.

## Citation

To be added on publication.
