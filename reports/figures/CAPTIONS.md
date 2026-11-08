# Figure and table captions (draft for a paper)

Regenerate everything with `python scripts/figures/make_figures.py` (inputs in `reports/figures/data/`
and `reports/2026-11-08_slice_latency_yago46_{tiny,full}.json`). Vector PDFs embed TrueType fonts;
sizes: single column 89 mm, double column 183 mm; sans-serif 5–7 pt at final size.

**Figure 1 — Architecture of the retrieval layer and its place in the model.**
Offline, the YAGO 4.6 Turtle files are compiled by `hg_index` into a memory-mapped index: an
incidence hypergraph in Elias–Fano CSR form whose rows are grouped into relation runs, so that the
Δ best incidences of a node are selected in O(runs + Δ) independently of its degree. Online, seed
entities are expanded by `hg_slice` (bounded BFS or approximate personalized PageRank by local push,
with total work ≤ M/(αε)) into a local subgraph that feeds the walk, memory and injection stages.
Solid blue boxes are implemented and tested; dashed boxes are planned; grey boxes are data. Measured
numbers refer to the full YAGO 4.6 (English-Wikipedia entities).

**Figure 2 — Per-query slice latency.**
Empirical CDF of latency for bounded BFS (ball) and PPR local push on YAGO 4.6 tiny (a, b) and full
(c, d), for 2000 random seeds with degree ≥ 1 (a, c) and the 100 highest-degree nodes (b, d).
Single thread, warm page cache, latency of the full Python call (binding overhead included). Dots
mark the 95th percentile; the vertical line is the acceptance target p95 = 50 ms. Parameters:
max_nodes = 1024, hops = 2, Δ = 16, α = 0.15, ε = 10⁻⁴, scan_cap = 4096; sampling seed 0.

**Figure 3 — Work versus latency.**
Per-query work (sum of capped degrees over expansions or pushes) against latency for random
connected seeds. On tiny (a), PPR latency grows roughly linearly with work. On the full index (b),
latency per unit of work is several times higher, and ball queries with the same work (Δ = 16)
spread over two orders of magnitude, which indicates costs not counted as work (induced-edge scan,
memory access). The vertical line is the Andersen–Chung–Lang bound M/(αε) for seed mass M = 1.

**Figure 4 — Degree distribution.**
Complementary CDF of node degree (incidences) over connected nodes. The vertical line marks the
fan-out cap Δ = 16 used by both slicing methods; the heavy tail (maximum degree 1 687 162 in the
full index) motivates degree-independent top-Δ selection.

**Table 1 (`table1_index.tex`) — Index statistics** for YAGO 4.6 tiny and full: sizes, build time
and peak memory (single process, Windows 10, Rust 1.99.1 release build).

**Table 2 (`table2_latency.tex`) — Slice latency** p50 / p95 / p99 / max (ms) and mean slice size per
index, seed workload and method; p95 in bold (acceptance criterion).

LaTeX tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
