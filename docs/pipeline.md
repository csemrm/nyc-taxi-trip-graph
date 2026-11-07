# Pipeline

Historical taxi trips become a weighted directed graph. Dijkstra finds the cheapest route through it. No ML.

```
                         BUILD  (once, ~5 s, O(T + V + E))
 ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────────┐
 │  NYC.csv    │──▶│   Clean      │──▶│  Grid cell    │──▶│  Hash index   │
 │ 1.46M trips │   │ bounds,      │   │ lat/lon → int │   │ cell key →    │
 │ (streamed)  │   │ duration     │   │ key, ~500 m   │   │ vertex id     │
 └─────────────┘   │ 1 s–6 h      │   └───────────────┘   │ 0..V-1  O(1)  │
                   └──────────────┘                       └───────┬───────┘
                      loader.py          location.py              │
                                                                  ▼
 ┌───────────────────────────┐   ┌───────────────────────────────────────┐
 │  CSR graph (TravelGraph)  │◀──│  Aggregate per (origin→dest) pair     │
 │  offsets / targets /      │   │  key = (u<<32)|v → [sum, count]       │
 │  weights   O(V+E) space   │   │  weight = mean seconds                │
 │  V=4001  E=117,785        │   │  same-cell trips skipped              │
 └─────────────┬─────────────┘   └───────────────────────────────────────┘
               │                          graph.py
               │
   ────────────┼───────────────────────────────────────────────────────────
               │                  QUERY / ANALYSIS
               │
     ┌─────────┼───────────────────────┬──────────────────────┐
     ▼         ▼                       ▼                      ▼
 ┌────────┐ ┌──────────────────┐ ┌──────────────┐     ┌──────────────┐
 │ BFS    │ │ DFS              │ │ Dijkstra     │     │ Kruskal MST  │
 │ hop    │ │ weak components  │ │ indexed heap │     │ (optional)   │
 │ levels │ │ O(V+E)           │ │ O((V+E)logV) │     │ O(E log E)   │
 └────────┘ └──────────────────┘ └──────┬───────┘     └──────────────┘
 traversal.py                           │               mst.py
                                        │
   pickup lat/lon ─▶ cell key ─▶ vertex │ id        shortest_path.py + heap.py
   dest   lat/lon ─▶ cell key ─▶ vertex │ id
                                        ▼
                      ┌──────────────────────────────────┐
                      │  Dijkstra loop                   │
                      │  1. dist[src]=0, push heap       │
                      │  2. pop min vertex u             │
                      │  3. u == dest?  ── yes ─▶ stop   │
                      │  4. relax edges u→v:             │
                      │     dist[u]+w < dist[v] ?        │
                      │     update dist, parent, heap    │
                      └────────────────┬─────────────────┘
                                       ▼
                      ┌──────────────────────────────────┐
                      │  Follow parent[] back, reverse   │
                      │  → route + total duration        │
                      └────────────────┬─────────────────┘
                                       ▼
        SUCCESS  │  NO_PATH  │  LOCATION_NOT_IN_GRAPH  │  INVALID_COORDINATES
                       RouteResult (visited, relaxed, time, V, E)
```

V, E and timings are for the full Kaggle dataset with the default 0.005° grid and mean weights.

## Why it is fast

- **Indexed heap:** tracks each vertex's position, so decrease-key is O(log V) and the heap never holds more than V entries.
- **Early stop:** Dijkstra stops when the destination is popped; a typical query visits about 700 of 4001 vertices.
- **Workspace reuse:** `dist`, `parent` and the heap are allocated once; each query resets only the vertices it touched.
- **Flat storage:** CSR lists plus one packed int per edge key; about 39 MB peak on the full dataset.
