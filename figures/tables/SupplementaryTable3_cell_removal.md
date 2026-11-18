**Supplementary Table 3 | In silico cell removal on the locked model, and CK-gate thresholds.** **a**, Each cell set was removed from every held-out test patient (713 cancers, 44 controls; cancers and controls alike) and the locked three-model ensemble re-scored; matched random removal deletes the same number of the patient's other cells (mean of five draws). AUROC 95% CIs and differences from stratified patient bootstrap (4,000 replicates). Sensitivity is given at the original cut-off (with the number of controls below it) and at a cut-off re-estimated on the altered control scores to keep 40 of 44 controls below. Removal alters the input distribution, so these results show what the model output depends on, not the causal role of the removed cells. **b**, AUROC of per-patient CK-gated cell counts across nine predefined gate thresholds (16-bit intensities).

**a**

| Cells removed | Median per cancer patient | AUROC (95% CI) | AUROC, matched random removal | Difference vs random (95% CI) | Sensitivity at original 90% cut-off (controls below) | Sensitivity at re-estimated 90% cut-off |
|---|---|---|---|---|---|---|
| None (intact model) | – | 0.82 (0.75–0.89) | – | – | 56.5% (40/44) | 56.5% |
| CK-gated cells | 2 | 0.82 (0.75–0.88) | 0.83 | -0.002 (-0.009 to +0.004) | 56.2% (40/44) | 56.4% |
| All CTC-like cells | 618 | 0.73 (0.65–0.80) | 0.80 | -0.067 (-0.157 to +0.025) | 59.3% (36/44) | 32.1% |
| Top-10 attributed cells | 10 | 0.84 (0.78–0.89) | 0.82 | +0.013 (-0.023 to +0.053) | 28.9% (43/44) | 51.6% |
| Top-50 attributed cells | 50 | 0.82 (0.76–0.88) | 0.83 | -0.004 (-0.065 to +0.069) | 14.9% (44/44) | 60.4% |
| Mass-90 attributed cells | 220 | 0.78 (0.71–0.84) | 0.82 | -0.039 (-0.105 to +0.028) | 0.6% (44/44) | 45.7% |
| All cluster 10 cells | 32 | 0.83 (0.77–0.89) | 0.83 | +0.004 (-0.035 to +0.046) | 42.1% (42/44) | 60.4% |
| All cluster 20 cells | 23 | 0.82 (0.74–0.88) | 0.82 | -0.006 (-0.018 to +0.004) | 54.1% (39/44) | 52.5% |
| All cluster 21 cells | 31 | 0.82 (0.75–0.89) | 0.83 | -0.005 (-0.025 to +0.015) | 56.1% (39/44) | 48.9% |

**b**

| CK gate (CK > / CD45/CD31 <) | AUROC (95% CI) | Controls with ≥1 gated cell (of 44) |
|---|---|---|
| 30k / 5k | 0.52 (0.44–0.61) | 30 |
| 30k / 10k | 0.54 (0.45–0.62) | 39 |
| 30k / 15k | 0.56 (0.48–0.64) | 43 |
| 40k / 5k | 0.51 (0.44–0.59) | 25 |
| 40k / 10k (primary) | 0.50 (0.42–0.59) | 32 |
| 40k / 15k | 0.53 (0.45–0.60) | 37 |
| 50k / 5k | 0.50 (0.43–0.56) | 14 |
| 50k / 10k | 0.47 (0.40–0.55) | 21 |
| 50k / 15k | 0.49 (0.41–0.57) | 24 |
