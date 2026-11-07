# NYC Taxi Trip Duration Estimation (graph + Dijkstra)

CS501 Fall 2026, Team 5. Estimates trip duration between two NYC locations by turning historical taxi trips into a weighted directed graph and running classical algorithms (hashing, BFS, DFS, Dijkstra with an indexed min-heap, optional Kruskal MST). **No ML/AI models.**

Pure Python 3.12 standard library, no dependencies. See [docs/pipeline.md](docs/pipeline.md) for how it works.

## 1. Get the data

The dataset is not in git (about 200 MB). Download it with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (needs `~/.kaggle/kaggle.json`):

```bash
mkdir -p data
kaggle datasets download -d yasserh/nyc-taxi-trip-duration -p data --unzip
# creates data/NYC.csv (1,458,644 rows)
```

The loader expects the Kaggle columns `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `trip_duration` (and `pickup_datetime` if you use `--hours`).

No dataset handy? Generate a deterministic synthetic one:

```bash
python3 -m taxi_graph generate --out data/synthetic.csv --trips 100000
```

## 2. Run with Docker (recommended)

```bash
docker compose build
docker compose run --rm test                                    # run all tests

# graph + cleaning stats
docker compose run --rm app stats --csv /data/NYC.csv

# estimate Midtown -> JFK (lat lon pairs)
docker compose run --rm app query --csv /data/NYC.csv \
    --from 40.7580 -73.9855 --to 40.6413 -73.7781

# scaling benchmark
docker compose run --rm app bench --csv /data/NYC.csv --sizes 100000 500000 1458644

# optional Kruskal MST analysis
docker compose run --rm app mst --csv /data/NYC.csv
```

`./data` is mounted at `/data` inside the container.

### Web UI

```bash
docker compose up --build web          # then open http://127.0.0.1:8000/
# or locally:
python3 -m taxi_graph serve --csv data/NYC.csv --port 8000
```

Click a pickup point, then a destination on the map; optionally pick a pickup hour. The page shows the estimate, whether it came from a direct edge or a chained shortest path, the route legs, and search statistics. Blue dots are the grid cells that exist in the data; a click in a cell with no trips returns "Location not in graph". The drawn route joins cell centers; it is not a street route.

Notes: the graph is built once at startup (about 5 s on the full data) and held in memory, so each query takes milliseconds. There is no authentication, so the server binds to `127.0.0.1` by default and Docker publishes the port on localhost only. The page loads the Leaflet library (pinned, with integrity hash) from unpkg and map tiles from OpenStreetMap, so the map needs internet access; the API itself does not.

## 3. Run locally

```bash
python3 -m unittest discover -s tests
python3 -m taxi_graph stats --csv data/NYC.csv
python3 -m taxi_graph query --csv data/NYC.csv --from 40.7580 -73.9855 --to 40.6413 -73.7781
```

## Commands and options

| Command | Purpose |
|---|---|
| `generate --out F --trips N --seed S` | Write a synthetic CSV |
| `stats` | Build graph; print cleaning summary, V, E, components, BFS reach |
| `query --from LAT LON --to LAT LON [--hour H]` | Best estimate (direct edge, else shortest path), route, per-leg times. With `--hour`, night (0-6) and evening (20-24) pickups use time-of-day graphs |
| `bench --sizes N... --queries Q --memory` | Build/BFS/DFS/Dijkstra timings per row count (`--memory` adds tracemalloc, which slows builds) |
| `mst` | Kruskal minimum spanning forest (connectivity analysis, not a trip estimate) |
| `validate --fold K --out F.json` | Hold out every K-th trip; accuracy vs actual durations and vs baselines; per time-of-day graph comparison. See [docs/results.md](docs/results.md) |
| `serve --host H --port P` | Web UI with map; same data options as the other commands. See [Web UI](#web-ui) |
| `tune --out F.json` | Train/validation/test (60/20/20): choose which periods use their own graph and test a chained-path scale factor on validation, report on test |

Shared options for `stats`, `query`, `bench`, `mst`, `validate`, `tune`, `serve`:

| Option | Default | Meaning |
|---|---|---|
| `--csv PATH` | required | Trip CSV |
| `--max-rows N` | all | Use first N rows (deterministic subset) |
| `--cell-deg D` | `0.005` | Grid cell size in degrees (about 500 m) |
| `--stat mean\|median` | `mean` | Edge weight statistic |
| `--min-trips N` | `1` | Drop edges seen fewer than N times |
| `--max-duration S` | `21600` | Drop trips longer than S seconds |
| `--hours START END` | off | Only trips with pickup hour in `[START, END)` |

Query statuses: `SUCCESS`, `NO_PATH`, `LOCATION_NOT_IN_GRAPH`, `INVALID_COORDINATES`.

## Reference results

Full dataset, default settings: about 5 s build, V = 4001, E = 117,785, Dijkstra about 5 ms per query, about 39 MB peak memory.

Held-out validation (20% of trips), MAE: constant-mean baseline 465 s, shortest path 268 s, direct edge else shortest path 250 s (shortest path alone is biased about -100 s), period graphs with direct-first 239 s. The period-graph gain comes from night and evening trips; choosing only those periods on a separate validation split gives 237 s versus 252 s for the all-day graph (60% train). A chained-path scale factor removed bias but did not reduce error, so it was not adopted. `query` reports the direct-first estimate. Details and caveats in [docs/results.md](docs/results.md).

## Layout

```
taxi_graph/   loader, location, graph, traversal, heap, shortest_path, mst, pipeline, cli
tests/        unit + pipeline tests (Dijkstra checked against Bellman-Ford)
docs/         pipeline diagram, validation + time-of-day results
Dockerfile, docker-compose.yml
```

Planning documents (proposal, requirements, architecture) are the PDFs in the repo root.
