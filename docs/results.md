# Validation and time-of-day results

Reproduce (full Kaggle data, about 30 s):

```bash
python3 -m taxi_graph validate --csv data/NYC.csv --out docs/validation_results.json
```

Raw numbers: [validation_results.json](validation_results.json).

## Method

- Cleaned trips are split deterministically: every 5th trip is held out (20%, 291,162 trips), the rest train (1,164,652).
- The location index, graph (V = 3759, E = 105,619) and baseline use **train trips only**.
- Each test trip is predicted as the Dijkstra distance between its origin and destination grid cells, then compared with its actual duration.
- No ML: the estimator is aggregation + shortest path. One multi-target Dijkstra runs per distinct origin, not per trip.
- Defaults: 0.005 degree grid (about 500 m), mean edge weights.

Test-trip accounting: 4,326 same-cell trips excluded (no edge to predict), 224 with an unseen origin/destination, 122 with no path, **286,490 covered (99.96% of predictable trips)**.

## 1. Accuracy (held-out)

| Estimator | MAE | median AE | RMSE | bias | MAPE | within 20% |
|---|---|---|---|---|---|---|
| Shortest path (Dijkstra) | 268.0 s | 159.0 s | 475.8 s | -102.7 s | 36.7% | 39.8% |
| Global mean baseline | 464.7 s | 382.3 s | 659.6 s | -7.1 s | 96.5% | 21.3% |

The graph estimator cuts mean error by 42% against a constant baseline.

### Finding: chaining is not better than a direct edge

On the 273,866 test trips where a direct edge also exists:

| Estimator | MAE | median AE | RMSE | bias | within 20% |
|---|---|---|---|---|---|
| Shortest path | 257.8 s | 155.4 s | 457.4 s | -99.5 s | 39.7% |
| Direct edge only | 238.8 s | 160.5 s | 378.3 s | -0.0 s | 40.0% |

Shortest path **underestimates by about 100 s on average**. Taking the minimum over many noisy chained mean-legs is optimistic: it picks whichever route happens to look cheapest. Where a direct edge exists it has a lower MAE, RMSE and no bias. The multi-leg route earns its place for pairs with no direct edge (about 4% of covered trips) and for explaining routes, not for accuracy where a direct observation exists. That motivated the direct-first estimator below.

### Fix: direct edge first, else shortest path

`Engine.estimate()` (and the `query` command's "best estimate") returns the direct edge when one was observed, and the shortest path otherwise.

| Estimator (286,490 covered trips) | MAE | median AE | RMSE | bias | within 20% |
|---|---|---|---|---|---|
| Shortest path | 268.0 s | **159.0 s** | 475.8 s | -102.7 s | 39.8% |
| Direct edge, else path | **249.8 s** | 164.0 s | **404.0 s** | -7.7 s | **40.0%** |

MAE falls 6.8%, RMSE 15%, and the bias nearly disappears. The **median error gets slightly worse** (159 s to 164 s) and within-20% barely moves: direct edges fix the large errors, not the typical one.

The weak spot is the 12,624 trips (4.4%) with no direct edge, which must use a chained path: MAE 489.6 s, bias -173.1 s, the worst group. We did not try to correct it.

## 2. Time-of-day graphs (FR-15)

One graph per period, built from train trips in that pickup-hour window, compared with the all-day graph on the same test trips (covered by both). The first table uses the plain shortest-path estimator; the direct-first results follow it.

| Period | Test trips | All-day MAE | Period MAE | All-day within 20% | Period within 20% |
|---|---|---|---|---|---|
| night 00-06 | 33,723 | 230.8 s | **192.3 s** | 36.5% | **49.7%** |
| am rush 06-10 | 44,264 | 275.7 s | 271.4 s | 35.1% | 36.0% |
| midday 10-16 | 82,932 | 305.2 s | 285.6 s | 38.6% | 40.3% |
| pm rush 16-20 | 63,191 | 281.7 s | 253.1 s | 42.6% | 44.2% |
| evening 20-24 | 62,502 | 209.9 s | **191.0 s** | 43.8% | **50.7%** |
| **all periods** | 285,620 | 265.9 s | **244.7 s** | 39.8% | **43.8%** |

- Period graphs lower MAE by 8% overall (median AE 158.6 s to 143.2 s, within-20% 39.8% to 43.8%).
- MAE reduction by period: night -17%, pm rush -10%, evening -9%, midday -6%, am rush -2%. The morning-rush graph barely beats the all-day graph. We did not investigate why.
- Period graphs are sparser, so coverage drops slightly (99.4-99.8% versus 99.9-100%). A hybrid that uses the period graph when it covers the pair and the all-day graph otherwise gives MAE 246.8 s on all 286,490 covered trips.
- Period graphs still underestimate (bias -88 s overall), for the same reason as above.

### Time-of-day with direct-first estimation

Same comparison, but each estimator uses a direct edge when it exists (period graph first, then the all-day graph if the period graph cannot answer). All 286,490 covered trips:

| Period | All-day MAE | Period MAE | All-day within 20% | Period within 20% |
|---|---|---|---|---|
| night 00-06 | 283.0 s | **201.9 s** | 30.2% | **48.0%** |
| am rush 06-10 | **258.2 s** | 262.8 s | 35.7% | 36.7% |
| midday 10-16 | **258.5 s** | 266.9 s | 42.0% | 40.9% |
| pm rush 16-20 | **237.0 s** | 246.8 s | 45.2% | 43.9% |
| evening 20-24 | 227.5 s | **196.8 s** | 40.6% | **49.4%** |
| **all periods** | 249.8 s | **238.9 s** | 40.0% | **43.6%** |

Best configuration overall: period graphs with direct-first (MAE 238.9 s, median AE 149.3 s, within-20% 43.6%, bias -3.2 s). But the gain is concentrated at **night and evening**; in the morning rush, midday and evening rush the period graph is slightly *worse* than the all-day graph once direct edges are used (sparser per-period edge means are noisier). The earlier "8% better everywhere" reading only holds for the shortest-path estimator. A practical rule from this data is to use period graphs for night and evening and the all-day graph otherwise; we did not implement or test that selection, so it would need its own held-out check.

## 3. Choices made on a separate validation split

Sections 1-2 looked at the test set when deciding what to try, so the follow-up ideas ("use period graphs only at night and evening", "correct the chained-path bias") got a proper protocol: kept-trip index % 5 == 0 is test (the same 291,162 test trips as above), == 1 is validation, the rest (60%, 873,489 trips) is train. Choices are fitted on **validation only** and scored once on test.

```bash
python3 -m taxi_graph tune --csv data/NYC.csv --out docs/selection_results.json
```

Raw numbers: [selection_results.json](selection_results.json). Graph and location index use the 60% train set, so absolute numbers are slightly worse than in sections 1-2 (all-day direct-first MAE 252.0 s here versus 249.8 s with 80% train).

### Period selection (works)

Chosen per period by lower validation MAE:

| Period | Validation MAE all-day | Validation MAE period | Chosen |
|---|---|---|---|
| night 00-06 | 282.3 s | 206.2 s | period graph |
| am rush 06-10 | 261.2 s | 269.5 s | all-day graph |
| midday 10-16 | 259.4 s | 271.3 s | all-day graph |
| pm rush 16-20 | 241.0 s | 252.9 s | all-day graph |
| evening 20-24 | 229.6 s | 203.7 s | period graph |

The validation split picks the same two periods the test set suggested, so that choice was not a test-set artifact.

| Test estimator (286,405 covered trips) | MAE | median AE | RMSE | bias | within 20% |
|---|---|---|---|---|---|
| All-day graph, direct first | 252.0 s | 164.5 s | 408.4 s | -8.0 s | 39.9% |
| Period graphs everywhere, direct first | 243.9 s | 151.1 s | 414.7 s | -2.3 s | 43.1% |
| **Selected periods (night, evening), direct first** | **237.1 s** | **145.8 s** | **403.2 s** | -44.3 s | **43.7%** |

Selecting periods cuts MAE 5.9% against the all-day graph and 2.8% against using period graphs everywhere, and improves the median error too. Note the overall bias (-44 s) is larger than for either alternative; we did not investigate why.

### Chained-path scale factor (does not help)

One constant, 1.128 = sum(actual) / sum(predicted) over 14,810 validation trips answered by a chained path (no direct edge). On the 14,748 such test trips:

| | MAE | median AE | RMSE | bias | within 20% |
|---|---|---|---|---|---|
| Before scale | **483.0 s** | **294.0 s** | **760.9 s** | -148.3 s | 41.4% |
| After scale | 489.9 s | 311.2 s | 771.2 s | **+19.8 s** | 40.1% |

Scaling removes the bias but every error metric gets slightly worse: the errors on these trips are spread widely, so shifting the center does not reduce them. **Not adopted.** Overall it also changes nothing useful (all-day MAE 252.0 s to 252.4 s; selected periods 237.1 s to 238.4 s).

### Not yet wired into the query path

`Engine` and the `query` command still use a single all-day graph with direct-first estimation. Using period graphs for night and evening at query time needs per-period graph building and a pickup-hour argument; the selection above is evaluation only.

## Caveats

- The split is by row order, not chronological. Dataset rows are not time-ordered, so this approximates random hold-out; a time-based split would be a stricter test.
- Same-cell trips (1.5% of the test set) are excluded because the graph has no self-loop edges, so error figures cover inter-cell trips only.
- Results depend on grid size (0.005 degrees) and the mean statistic; neither was tuned.
- Dataset: Kaggle `yasserh/nyc-taxi-trip-duration` (2016 NYC taxi trips).
