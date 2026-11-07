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

Shortest path **underestimates by about 100 s on average**. Taking the minimum over many noisy chained mean-legs is optimistic: it picks whichever route happens to look cheapest. Where a direct edge exists it has a lower MAE, RMSE and no bias. The multi-leg route earns its place for pairs with no direct edge (about 4% of covered trips) and for explaining routes, not for accuracy where a direct observation exists. A natural next step is to prefer the direct edge when it exists.

## 2. Time-of-day graphs (FR-15)

One graph per period, built from train trips in that pickup-hour window, compared with the all-day graph on the same test trips (covered by both).

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

## Caveats

- The split is by row order, not chronological. Dataset rows are not time-ordered, so this approximates random hold-out; a time-based split would be a stricter test.
- Same-cell trips (1.5% of the test set) are excluded because the graph has no self-loop edges, so error figures cover inter-cell trips only.
- Results depend on grid size (0.005 degrees) and the mean statistic; neither was tuned.
- Dataset: Kaggle `yasserh/nyc-taxi-trip-duration` (2016 NYC taxi trips).
