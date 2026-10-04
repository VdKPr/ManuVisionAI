| Config | Test accuracy % | Balanced acc % | Macro F1 | Escape rate % | False-reject rate % | Pooled escapes |
|---|---|---|---|---|---|---|
| aug+mirror | 100.0 | 100.0 | 1.00 | 0.0 | 0.0 | 0/18 |

Seed-paired change vs baseline (mean ± std over seeds, percentage points):

| Config | Δ accuracy | Δ macro F1 (×100) | Δ escape rate | Δ false-reject rate |
|---|---|---|---|---|

Per-class recall % (mean over seeds):

| Config | good | bent | color | flip | scratch |
|---|---|---|---|---|---|
| aug+mirror | 100 | 100 | 100 | 100 | 100 |

Operating point tuned on validation (escape cost = 10x a false reject; part flagged defective when P(good) < tau):

| Config | tau (per seed) | Escapes, argmax | Escapes, tuned tau | False rejects, argmax | False rejects, tuned tau |
|---|---|---|---|---|---|
| aug+mirror | 0.873 | 0/18 | 0/18 | 0/49 | 0/49 |

Each test split has 67 images; one image is 1.5 accuracy points.
Differences smaller than the seed-to-seed spread are noise, not findings.
