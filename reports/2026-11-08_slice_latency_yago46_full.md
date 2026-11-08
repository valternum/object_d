# 2026-11-08 · Slice latency on the full YAGO 4.6 index — acceptance of issue #15 (D-015)

**Hypothesis.** Per-query slicing on the memory-mapped index of the full YAGO 4.6 (English-Wikipedia
entities) stays within p95 ≤ 50 ms per query, single thread, warm page cache.

**Data.** `yago-4.6-facts.zip` (sha256 `f9b57f35…1c14b1`, 5.09 GB unpacked), `yago-4.6-meta.zip`
(sha256 `d551720d…`, 0.72 GB), `yago-4.6-schema.zip`, `yago-4.6-taxonomy.zip`
(sha256 `6cc9f409…95278`); CC BY 4.0. Full digests: `sources.json` written by `scripts/yago/prepare.py`.

**Build.** `holm-index build --schema yago-schema.ttl --taxonomy yago-taxonomy.ttl
--facts yago-facts.ttl --meta yago-meta.ttl --chunk-mb 512`: exit 0, **385.5 s**, peak working set
**≈ 2.5 GB** (sampled every 500 ms; limit 12 GB), index **734 MB**.
5 589 647 nodes, 17 786 582 hyperedges, 81 relations, 206 342 classes, 7 012 658 (node, class)
pairs, 446 462 edges with a time interval, **5 556 213 nodes (99.4 %) with a Wikidata QID**,
0 duplicate edges, 5 643 175 meta facts not attached to an entity–entity edge (mostly literal facts).

**Benchmark.** `scripts/yago/bench_slice.py --raw-out …` (params as in the tiny report; seed 0;
2000 random connected seeds + 100 top-degree nodes; warm-up pass). Commit `86009b0`; hardware as in
the tiny report. Run twice (pipeline run and re-run with raw output); numbers below are the re-run,
the first run differed by ≤ 1.3 ms at p95.

| Workload | Method | n | p50 | **p95** | p99 | max | mean nodes | mean edges | mean work |
|---|---|---|---|---|---|---|---|---|---|
| random connected | ball | 2000 | 1.61 | **6.81** | 9.62 | 13.31 | 40.8 | 115.6 | 55.3 |
| random connected | ppr | 2000 | 21.15 | **31.37** | 36.04 | 70.56 | 215.7 | 1294.9 | 5767.8 |
| top-degree hubs | ball | 100 | 7.06 | **13.95** | 18.62 | 19.85 | 92.8 | 413.7 | 201.9 |
| top-degree hubs | ppr | 100 | 25.59 | **32.22** | 34.71 | 36.50 | 209.3 | 1645.5 | 7596.9 |

Figures: `reports/figures/fig2_latency_ecdf.pdf`, `fig3_work_vs_latency.pdf`, `fig4_degree_ccdf.pdf`.

## Verdict

- ФАКТ: **acceptance passed** — p95 ≤ 32.3 ms in all four cells (target 50 ms); build peak memory
  ≈ 2.5 GB ≤ 12 GB.
- ФАКТ: PPR max latency 70.6 ms exceeds 50 ms for single queries (p99 36.0 ms); the criterion is on p95.
- ФАКТ: from tiny to full, PPR mean work grows 2.7× while p50 latency grows ≈ 60×; ball queries with
  identical work (Δ = 16) spread from 0.05 to 10 ms (Fig. 3b).
- ВЫВОД: on the full index latency is dominated by costs outside the counted work — the induced-edge
  scan (rows up to `scan_cap` = 4096) and memory access over a 734 MB mapping. Profiling is required
  before optimising (R-L3); candidate fixes: lower `scan_cap`, induced edges only from traversed
  edges, smaller hash maps in PPR.
- ВЫВОД: 99.4 % of nodes carry a Wikidata QID, so QA datasets built on Wikidata can be joined to the
  index by QID.
- Not measured: cold-cache latency.

## Online memory (measured inside the Python process, `K32GetProcessMemoryInfo`)

Whole benchmark (2100 queries × 2 methods + warm-up): peak working set **591 MB** (includes
file-backed pages of the 734 MB mapping), peak private commit **551 MB**, working set at exit 39 MB.
Stage breakdown (separate run, 300 ball + 300 PPR slices):

| Stage | private commit (MB) | working set (MB) |
|---|---|---|
| Python + NumPy, before opening the index | 370 | 25 |
| after `Index.open` (mmap, header check) | 371 | 25 |
| after `degrees()` (5.59 M × u32) | 392 | 50 |
| after seed selection (NumPy temporaries) | 433 | 90 |
| after 300 ball slices | 433 | 314 |
| after 300 PPR slices | 435 | 415 |

- ФАКТ: slicing itself adds ≈ 2 MB of anonymous memory over 600 queries; the growth of the working
  set is file-backed mmap pages (reclaimable page cache).
- ГИПОТЕЗА: the 370 MB baseline commit comes from NumPy's BLAS per-thread buffers (not from HOLM code);
  to be checked with `OPENBLAS_NUM_THREADS=1`.
- Repeated runs: p95 stayed within 6.8–7.1 ms (ball) and 29.7–32.7 ms (PPR) for random seeds,
  14.0–14.6 ms (ball) and 32.2–39.0 ms (PPR) for hubs — all below 50 ms; PPR max varied 52–94 ms.
