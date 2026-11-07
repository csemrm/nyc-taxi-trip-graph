# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Python 3.12, **stdlib only** (no dependencies), packaged in `taxi_graph/`, tests in `tests/`, Docker-based. Git initialized on `main`. Planning PDFs in the repo root are the source of truth:
- `CS501_NYC Taxi Algorithmic_Project Proposal_Week 8_Team 5 (1).pdf` — scope, methodology, 8-week schedule
- `Requirements Analysis NYC Taxi Trip Duration Estimation-1.pdf` — FR-01..16, NFR-01..07, user stories, MoSCoW
- `SP3_Project_Architecture_NYC_Taxi (1).pdf` — module design, data structures, query/result contract

Dataset: `data/NYC.csv` (Kaggle `yasserh/nyc-taxi-trip-duration`, 1,458,644 rows; gitignored, mounted into Docker at `/data`). The official competition download returns 403 unless competition rules are accepted on the Kaggle account; the dataset mirror was used instead. `python -m taxi_graph generate` writes deterministic synthetic CSVs.

## Commands

```bash
# Docker (preferred)
docker compose build
docker compose run --rm test                                   # all tests
docker compose run --rm app stats --csv /data/NYC.csv          # build graph + stats
docker compose run --rm app query --csv /data/NYC.csv --from 40.7580 -73.9855 --to 40.6413 -73.7781
docker compose run --rm app bench --csv /data/NYC.csv --sizes 100000 500000 1458644
docker compose run --rm app mst --csv /data/NYC.csv            # optional Kruskal
docker compose run --rm app validate --csv /data/NYC.csv       # held-out accuracy + time-of-day graphs (~30 s)

# Local
python3 -m unittest discover -s tests
python3 -m unittest tests.test_algorithms.DijkstraTests.test_competing_paths   # single test
python3 -m taxi_graph --help
```

Reference numbers (full dataset, default 0.005 deg grid, mean): build ~5 s, V=4001, E=117,785, Dijkstra ~5 ms avg over random pairs (65% of random pairs are unreachable and fail fast), ~20 ms mean for reachable pairs. `complexity --csv ... --out-dir docs/sp4` regenerates the SP4 measurements and SVG plots. `bench --memory` enables tracemalloc and inflates build time.

## Code map

`config` (frozen `Config`) -> `loader` (streaming clean, per-reason counts) -> `location` (grid key + dict key->vertex id) -> `graph` (`EdgeAggregator`, CSR `TravelGraph`) -> `traversal` (BFS, DFS, weak components) / `heap` + `shortest_path` (indexed heap Dijkstra) / `mst` (Kruskal) -> `pipeline` (`build`, `Engine`) -> `cli`. `perf` and `synthetic` are helpers.

`validation` (separate from `pipeline`) does the train/test split (every `fold`-th cleaned trip is test; graph, index and baseline use train only), per-period graphs (`PERIODS`) and error metrics. It uses `Dijkstra.distances()` (multi-target, early stop) so predictions cost one search per distinct origin. `iter_clean_trips(..., with_hour=True)` appends the pickup hour. Results and caveats are in `docs/results.md`; key findings: shortest path is biased about -100 s, so `Engine.estimate()` uses a direct edge when one exists and the shortest path otherwise (MAE 268 s -> 250 s); period graphs help mainly at night/evening and slightly hurt other periods under direct-first.

`selection.run_selection` (CLI `tune`) is the clean protocol for any data-fitted choice: index % 5 == 0 test, == 1 validation, rest train; fit on validation, report once on test. Result: use period graphs only for night and evening (MAE 252 -> 237 s); a chained-path scale factor fixed bias but not error and was not adopted. `query --hour H` uses the selected period graphs (`pipeline.SELECTED_PERIODS`) via `build(cfg, periods=...)` and `Engine.estimate(..., hour=H)`, falling back to the all-day graph.

Performance design choices to preserve:
- One int key per grid cell and one packed int `(u<<32)|v` per edge; single pass aggregation O(T).
- CSR built by counting sort O(V+E); plain lists (not `array`) in hot loops for speed.
- Heap uses the Dijkstra `dist` list as its key store (no duplication); heap size <= V (decrease-key, no lazy duplicates).
- `Dijkstra` workspace allocated once; each query resets only touched vertices, so early-stopped queries are not O(V).
- Self-loop trips (same origin/destination cell) are skipped in the graph and counted; src==dst queries return 0 directly.

## Project goal

Estimate NYC taxi trip duration between a pickup and destination by turning historical trips into a weighted directed graph and running classical graph algorithms. **Hard constraint (NFR-01): no ML/AI models** — no regression, classification, neural nets, or learned predictors. Also out of scope: live traffic, real-time GPS, turn-by-turn navigation, commercial routing APIs. Edge weights come from plain aggregation (mean/median), not learning.

## Architecture (pipeline)

`Historical trips (CSV) → Clean/normalize → Hash & aggregate → Directed weighted graph → BFS/DFS/Dijkstra → Route + performance report`

Planned modules (each separate and individually testable, NFR-07):
- **DataLoader / DataCleaner** — read pickup/drop-off coords, pickup time, passenger count, duration. Drop missing/non-finite values, out-of-NYC-bounds coords, non-positive durations; upper duration cutoff is a *configurable* value. Record exclusion counts per reason. Configurable row limit/subset for dev.
- **LocationNormalizer / LocationIndex** — map coords to a stable zone/grid-cell key (resolution configurable, fixed per run), then hash map `LocationKey → VertexID` with IDs `0..V-1`, no gaps.
- **EdgeAggregator / TravelGraph** — group by ordered (origin, dest) pair; weight = configured mean or median duration (seconds), keep observation count, optional min-trip-count filter. Adjacency lists, **directed** (u→v does not imply v→u), weights non-negative.
- **GraphTraversal** — BFS (reachability + hop levels), DFS (components). Must state whether DFS reports weak connectivity or directed reachability.
- **MinHeap / ShortestPath** — Dijkstra with indexed binary min-heap (decrease-key), distance + parent arrays, early stop when destination is popped. The O((V+E) log V) bound assumes indexed heap; a lazy duplicate-entry heap must report its own bound.
- **PathReconstructor / RouteResult** — follow parent pointers back and reverse. Result carries source, dest, status, vertex path, per-edge durations, total, visited vertices, examined/relaxed edges, exec time, graph size.
- **PerformanceAnalyzer / Reporter** — timings and counters; memory only if measurable consistently, else compare theoretical space.
- Optional (Could Have): per-time-period graphs (FR-15); MST via Kruskal/Prim on a derived undirected graph (FR-16), labeled as connectivity analysis, not a trip estimate.

## Behavior contracts easy to get wrong

- Query statuses: `SUCCESS`, `NO_PATH` ("No path found."), `LOCATION_NOT_IN_GRAPH` ("Location not in graph."). Invalid coordinates are rejected at validation. Pickup == destination → duration 0 with a one-vertex route (never conflate "no path" with a zero-minute trip).
- Determinism (NFR-04): same input + settings must yield identical graph, weights, and routes — use stable key ordering and fixed rules. Record cleaning/grid/aggregation settings as run metadata for every experiment.
- Durations displayed as minutes and seconds.
- Testing is layered: unit checks → hand-built known-answer graphs (competing paths, disconnected/isolated vertices, repeated edges, ties) → full-pipeline integration on a CSV subset → real-data queries. Dijkstra correctness (NFR-02) is verified on hand-built graphs before real data.
- Reporting should compare direct-edge duration vs multi-edge shortest-path duration for selected pairs (FR-14).

## Conventions from the docs

- Complexity targets: hash lookup O(1) avg; BFS/DFS O(V+E); Dijkstra O((V+E) log V); Kruskal O(E log E); graph build O(T+E) over T retained trips.
- Requirement/story IDs (`FR-xx`, `US-SP-01`, …) are used for traceability; reference them in tests and commit messages where relevant.
- Develop on a configurable subset first, then scale the same pipeline to the full retained dataset on a standard laptop (NFR-05).
