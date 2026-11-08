# 2026-11-08 · Slice latency on YAGO 4.6 tiny (issue #15, D-015)

**Hypothesis.** Per-query slicing on the memory-mapped index stays within p95 ≤ 50 ms
(single thread, warm page cache). ГИПОТЕЗА for the full index; measured here on tiny only.

**Config.** `scripts/yago/bench_slice.py`, defaults: `max_nodes=1024, hops=2, delta=16,
alpha=0.15, epsilon=1e-4, scan_cap=4096`; seed 0; 2000 random seeds with degree ≥ 1
(`numpy.random.default_rng(0)`, without replacement) and the 100 highest-degree nodes.
One seed per query, mass 1.0. Warm-up pass over the same queries before timing.
Timing = full Python call `Index.slice` (binding overhead included), `time.perf_counter_ns`.

**Data.** YAGO 4.6 tiny (`yago-4.6-tiny.zip`, sha256 `0cb8805e…a0f81`, CC BY 4.0), split by
`scripts/yago/prepare.py` into schema (2176 lines) and data (7 313 006 lines). Index built by
`holm-index build --schema tiny-schema.ttl --facts tiny-data.ttl`:
415 228 nodes, 148 372 edges, 79 relations, 206 694 classes, 601 945 (node, class) pairs;
32.2 MB; build 20.0 s, peak working set ≈ 404 MB (sampled every 200 ms).
Connected nodes 140 621, max degree 8226.

**Commit / hardware.** `59192bd`, Windows 10 Pro 19045, Intel64 Family 6 Model 141 (12 logical
CPUs), Python 3.12.8, Rust 1.99.1 `x86_64-pc-windows-gnu` (D-017), release build.
A network download ran in the background during the measurement.

## Results (ms)

| Workload | Method | n | p50 | p95 | p99 | max | mean nodes | mean edges | mean work |
|---|---|---|---|---|---|---|---|---|---|
| random connected | ball | 2000 | 0.036 | **0.346** | 0.621 | 0.801 | 12.5 | 14.0 | 13.8 |
| random connected | ppr | 2000 | 0.364 | **5.206** | 5.978 | 11.255 | 127.6 | 191.2 | 2161.1 |
| top-degree hubs | ball | 100 | 0.171 | **0.888** | 1.118 | 1.134 | 44.8 | 66.4 | 71.1 |
| top-degree hubs | ppr | 100 | 2.408 | **5.552** | 6.444 | 6.998 | 257.6 | 383.7 | 3993.2 |

Numbers are from the third run (commit `86009b0`, with `--raw-out` for the figures), stored in
`reports/2026-11-08_slice_latency_yago46_tiny.json`; the two earlier runs (commits `fb008af`,
`59192bd`) differed by ≤ 0.3 ms at p95 and ≤ 4.9 ms at max.

## Conclusions

- ФАКТ: on tiny, p95 ≤ 5.6 ms for both methods and both workloads, i.e. about 9× below the 50 ms target.
- ФАКТ: PPR work stays far below its bound M/(α·ε) = 66 667 (max mean 3993).
- ВЫВОД: the tiny index (32 MB) fits in the page cache; the result says nothing about cold
  pages or about the full index. Acceptance of #15 requires the same run on the full
  Wikipedia part of YAGO 4.6.
- ГИПОТЕЗА: PPR cost is dominated by `top_incident` heap merges and hash-map updates;
  profiling is needed before any optimisation (R-L3).
