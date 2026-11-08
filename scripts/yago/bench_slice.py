"""Slice latency benchmark on a ``.holmidx`` index (issue #15, D-015).

Not part of ``make test`` (R-H5). Measures the full Python call
``Index.slice`` (binding overhead included), single thread, after a warm-up
pass over the same queries (warm page cache).

    python scripts/yago/bench_slice.py --index tiny.holmidx --out reports/x.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import holm.holm_native as native  # noqa: E402

from holm.retrieval.types import SliceParams  # noqa: E402


def percentiles(ns: list[int]) -> dict[str, float]:
    ms = np.asarray(ns, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
        "mean_ms": float(ms.mean()),
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--hubs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nodes", type=int, default=1024)
    ap.add_argument("--delta", type=int, default=16)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--scan-cap", type=int, default=4096)
    ap.add_argument(
        "--raw-out", type=Path, default=None, help="optional .npz with per-query data for figures"
    )
    args = ap.parse_args(argv)
    raw: dict[str, np.ndarray] = {}

    t0 = time.perf_counter()
    ix = native.Index.open(str(args.index))
    open_s = time.perf_counter() - t0
    deg = ix.degrees()
    rng = np.random.default_rng(args.seed)
    connected = np.flatnonzero(deg > 0)
    workloads = {
        "random_connected": rng.choice(connected, size=min(args.queries, connected.size), replace=False),
        "top_degree_hubs": np.argsort(-deg.astype(np.int64), kind="stable")[: args.hubs],
    }
    common = dict(
        max_nodes=args.max_nodes,
        hops=args.hops,
        delta=args.delta,
        alpha=args.alpha,
        epsilon=args.epsilon,
        scan_cap=args.scan_cap,
    )
    one = np.ones(1, dtype=np.float64)
    results = []
    for wl, seeds in workloads.items():
        for method in ("ball", "ppr"):
            kw = SliceParams(method=method, **common).native_kwargs()
            for s in seeds:  # warm-up
                ix.slice(np.array([s], dtype=np.uint32), one, **kw)
            ns, nodes, edges, work = [], [], [], []
            for s in seeds:
                a = np.array([s], dtype=np.uint32)
                t = time.perf_counter_ns()
                d = ix.slice(a, one, **kw)
                ns.append(time.perf_counter_ns() - t)
                nodes.append(len(d["nodes"]))
                edges.append(len(d["edges"]))
                work.append(d["work"])
            key = f"{wl}.{method}"
            raw[f"{key}.latency_ns"] = np.asarray(ns, dtype=np.int64)
            raw[f"{key}.work"] = np.asarray(work, dtype=np.int64)
            raw[f"{key}.nodes"] = np.asarray(nodes, dtype=np.int64)
            raw[f"{key}.edges"] = np.asarray(edges, dtype=np.int64)
            row = {
                "workload": wl,
                "method": method,
                "queries": int(len(seeds)),
                **percentiles(ns),
                "mean_nodes": float(np.mean(nodes)),
                "mean_edges": float(np.mean(edges)),
                "mean_work": float(np.mean(work)),
                "max_work": int(np.max(work)),
            }
            results.append(row)
            print(
                f"{wl:17s} {method:4s} n={row['queries']:5d} p50={row['p50_ms']:.3f} ms "
                f"p95={row['p95_ms']:.3f} ms p99={row['p99_ms']:.3f} ms max={row['max_ms']:.3f} ms "
                f"nodes={row['mean_nodes']:.1f} edges={row['mean_edges']:.1f} work={row['mean_work']:.1f}"
            )
    report = {
        "index": str(args.index),
        "index_bytes": args.index.stat().st_size,
        "n_nodes": ix.n_nodes,
        "n_edges": ix.n_edges,
        "connected_nodes": int(connected.size),
        "max_degree": int(deg.max()),
        "open_seconds": open_s,
        "params": common,
        "seed": args.seed,
        "commit": git_commit(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.raw_out is not None:
        values, counts = np.unique(deg, return_counts=True)
        raw["degree.values"] = values.astype(np.int64)
        raw["degree.counts"] = counts.astype(np.int64)
        args.raw_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.raw_out, **raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
