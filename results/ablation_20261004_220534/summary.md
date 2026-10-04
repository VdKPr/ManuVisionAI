| Config | Test accuracy % | Balanced acc % | Macro F1 | Escape rate % | False-reject rate % | Pooled escapes |
|---|---|---|---|---|---|---|
| baseline | 88.6 ± 3.8 | 70.4 ± 7.7 | 0.75 ± 0.10 | 26.3 ± 5.3 | 1.4 ± 2.4 | 15/57 |
| aug+wce | 94.0 ± 1.5 | 83.5 ± 2.5 | 0.86 ± 0.04 | 14.0 ± 6.1 | 0.7 ± 1.2 | 8/57 |
| aug+mirror | 95.0 ± 3.4 | 85.7 ± 10.1 | 0.88 ± 0.08 | 10.5 ± 9.1 | 0.0 ± 0.0 | 6/57 |

Seed-paired change vs baseline (mean ± std over seeds, percentage points):

| Config | Δ accuracy | Δ macro F1 (×100) | Δ escape rate | Δ false-reject rate |
|---|---|---|---|---|
| aug+wce | 5.5 ± 3.1 | 10.4 ± 7.0 | -12.3 ± 3.0 | -0.7 ± 3.2 |
| aug+mirror | 6.5 ± 2.3 | 13.1 ± 6.1 | -15.8 ± 13.9 | -1.4 ± 2.4 |

Per-class recall % (mean over seeds):

| Config | good | bent | color | flip | scratch |
|---|---|---|---|---|---|
| baseline | 99 | 47 | 47 | 100 | 60 |
| aug+wce | 99 | 80 | 52 | 100 | 87 |
| aug+mirror | 100 | 87 | 62 | 100 | 80 |

Operating point tuned on validation (escape cost = 10x a false reject; part flagged defective when P(good) < tau):

| Config | tau (per seed) | Escapes, argmax | Escapes, tuned tau | False rejects, argmax | False rejects, tuned tau |
|---|---|---|---|---|---|
| baseline | 0.958, 0.949, 0.054 | 15/57 | 8/57 | 2/144 | 14/144 |
| aug+wce | 0.747, 0.934, 0.686 | 8/57 | 3/57 | 1/144 | 20/144 |
| aug+mirror | 0.234, 0.648, 0.383 | 6/57 | 7/57 | 0/144 | 0/144 |

Each test split has 67 images; one image is 1.5 accuracy points.
Differences smaller than the seed-to-seed spread are noise, not findings.
