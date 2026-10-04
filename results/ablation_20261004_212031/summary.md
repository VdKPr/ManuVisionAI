| Config | Test accuracy % | Balanced acc % | Macro F1 | Escape rate % | False-reject rate % | Pooled escapes |
|---|---|---|---|---|---|---|
| baseline | 92.5 ± 6.5 | 83.7 ± 7.8 | 0.84 ± 0.11 | 7.4 ± 3.2 | 3.4 ± 5.9 | 4/54 |
| aug | 95.5 ± 2.6 | 87.2 ± 7.8 | 0.88 ± 0.08 | 9.3 ± 6.4 | 0.7 ± 1.2 | 5/54 |
| wce | 92.5 ± 1.5 | 84.3 ± 8.4 | 0.84 ± 0.07 | 9.3 ± 6.4 | 3.4 ± 2.4 | 5/54 |
| aug+wce | 98.0 ± 2.3 | 96.4 ± 3.1 | 0.97 ± 0.03 | 3.7 ± 3.2 | 1.4 ± 2.4 | 2/54 |
| aug+mirror | 99.0 ± 1.7 | 96.7 ± 5.8 | 0.98 ± 0.04 | 3.7 ± 6.4 | 0.0 ± 0.0 | 2/54 |

Seed-paired change vs baseline (mean ± std over seeds, percentage points):

| Config | Δ accuracy | Δ macro F1 (×100) | Δ escape rate | Δ false-reject rate |
|---|---|---|---|---|
| aug | 3.0 ± 5.4 | 3.7 ± 8.2 | 1.9 ± 8.5 | -2.7 ± 6.6 |
| wce | 0.0 ± 5.2 | 0.2 ± 6.4 | 1.9 ± 8.5 | 0.0 ± 7.4 |
| aug+wce | 5.5 ± 7.1 | 12.6 ± 10.7 | -3.7 ± 6.4 | -2.0 ± 7.4 |
| aug+mirror | 6.5 ± 4.8 | 13.6 ± 7.0 | -3.7 ± 8.5 | -3.4 ± 5.9 |

Per-class recall % (mean over seeds):

| Config | good | bent | color | flip | scratch |
|---|---|---|---|---|---|
| baseline | 97 | 100 | 53 | 100 | 68 |
| aug | 99 | 87 | 75 | 100 | 75 |
| wce | 97 | 93 | 55 | 100 | 77 |
| aug+wce | 99 | 100 | 92 | 100 | 92 |
| aug+mirror | 100 | 100 | 100 | 100 | 83 |

Each test split has 67 images; one image is 1.5 accuracy points.
Differences smaller than the seed-to-seed spread are noise, not findings.
