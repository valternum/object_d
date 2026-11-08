"""Figures and tables for the HOLM retrieval-layer report (D-019: matplotlib, reports only).

    python scripts/figures/make_figures.py

Inputs (committed, small): ``reports/figures/data/{build_stats.json, latency_tiny.npz,
latency_full.npz}`` and ``reports/2026-11-08_slice_latency_yago46_{tiny,full}.json``.
Outputs: ``reports/figures/fig{1..4}_*.{pdf,svg,png}`` and ``table{1,2}_*.tex``.

Print conventions: vector PDF with embedded TrueType fonts, sans-serif 6-7 pt at final
size, single column 89 mm / double column 183 mm, colour-blind-checked two-series
palette (ball = blue, PPR = orange; dataviz validator: CVD dE 24.7, normal dE 33.6).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports" / "figures"
DATA = FIG / "data"
REPORTS = ROOT / "reports"
MM = 1 / 25.4
COL1, COL2 = 89 * MM, 183 * MM
TARGET_P95_MS = 50.0  # acceptance criterion of issue #15 (D-015)

C = {
    "ball": "#2a78d6",
    "ppr": "#eb6834",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "tint": "#cde2fb",
    "neutral": "#f0efec",
}
METHOD_LABEL = {"ball": "ball (bounded BFS)", "ppr": "PPR (local push)"}
WORKLOADS = {
    "random_connected": "random connected seeds",
    "top_degree_hubs": "top-degree hubs",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.5,
            "axes.edgecolor": C["ink2"],
            "axes.labelcolor": C["ink"],
            "xtick.color": C["ink2"],
            "ytick.color": C["ink2"],
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def hairline_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, which="major", axis=axis, color=C["grid"], linewidth=0.4)
    ax.set_axisbelow(True)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    return xs, np.arange(1, xs.size + 1) / xs.size


def result_row(report: dict, workload: str, method: str) -> dict:
    return next(r for r in report["results"] if r["workload"] == workload and r["method"] == method)


# --------------------------------------------------------------------------- figure 1


def box(ax, x, y, w, h, text, kind):
    styles = {
        "impl": dict(fc=C["tint"], ec=C["ball"], ls="-", tc=C["ink"]),
        "plan": dict(fc="white", ec=C["muted"], ls=(0, (3, 2)), tc=C["ink2"]),
        "data": dict(fc=C["neutral"], ec=C["ink2"], ls="-", tc=C["ink"]),
    }[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            fc=styles["fc"],
            ec=styles["ec"],
            lw=0.7,
            ls=styles["ls"],
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=5.6,
        color=styles["tc"],
        linespacing=1.25,
    )
    return (x, y, w, h)


def arrow(ax, p, q, dashed=False, text=None, offset=(0, 0)):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>,head_length=1.6,head_width=0.9",
            mutation_scale=1,
            lw=0.6,
            color=C["muted"] if dashed else C["ink2"],
            ls=(0, (3, 2)) if dashed else "-",
            shrinkA=0.5,
            shrinkB=0.5,
        )
    )
    if text:
        ax.text(
            (p[0] + q[0]) / 2 + offset[0],
            (p[1] + q[1]) / 2 + offset[1],
            text,
            fontsize=5.2,
            color=C["ink2"],
            ha="center",
            va="center",
        )


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def fig1_architecture(stats: dict, full: dict) -> None:
    fig = plt.figure(figsize=(COL2, 92 * MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 183)
    ax.set_ylim(0, 92)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        2, 88, "Offline, once per knowledge-graph release", fontsize=6.3, fontweight="bold", color=C["ink"]
    )
    ax.text(2, 55.5, "Online, per query", fontsize=6.3, fontweight="bold", color=C["ink"])
    ax.plot([2, 181], [58.5, 58.5], color=C["grid"], lw=0.5)

    a = box(ax, 2, 66, 36, 15, "YAGO 4.6 Turtle files\nschema · taxonomy\nfacts · meta facts", "data")
    b = box(
        ax,
        46,
        66,
        56,
        15,
        "hg_index (Rust)\nstreaming Turtle lexer · external-sort dictionary\n"
        "SHACL schema + taxonomy · BFS renumbering",
        "impl",
    )
    c = box(
        ax,
        110,
        66,
        71,
        15,
        ".holmidx, memory-mapped, zero-copy\n"
        "Elias–Fano CSR · relation runs (top-Δ in O(runs + Δ))\n"
        "node types · time intervals · Wikidata QIDs",
        "impl",
    )
    arrow(ax, right(a), left(b))
    arrow(ax, right(b), left(c))
    f = stats["full"]
    ax.text(
        145.5,
        62.5,
        f"full YAGO 4.6: {f['nodes'] / 1e6:.2f} M nodes · {f['edges'] / 1e6:.1f} M edges · "
        f"{f['index_mb']:.0f} MB\n"
        f"built in {f['build_s']:.0f} s, peak memory {f['peak_mb'] / 1024:.1f} GB",
        fontsize=5.2,
        color=C["ink2"],
        ha="center",
        va="center",
    )

    y1 = 34
    d = box(ax, 2, y1, 20, 13, "Query\ntext", "data")
    e = box(ax, 28, y1, 26, 13, "G1 entity linker\n(planned)", "plan")
    s = box(
        ax,
        60,
        y1,
        46,
        13,
        "hg_slice (Rust)\nball | PPR local push, work ≤ M/(αε)\nrelation · class · time filters",
        "impl",
    )
    g = box(ax, 112, y1, 30, 13, "Local subgraph\n≤ N_slice nodes,\ninduced edges", "data")
    h = box(ax, 148, y1, 33, 13, "Walk G2/G3 + mask M_O\n(planned)", "plan")
    arrow(ax, right(d), left(e))
    arrow(ax, right(e), left(s), text="seeds", offset=(0, 2.2))
    arrow(ax, right(s), left(g))
    arrow(ax, right(g), left(h))
    arrow(ax, (116, 66), top(s))
    ax.text(104, 55.5, "mmap reads", fontsize=5.2, color=C["ink2"], ha="left", va="center")
    rb, rp = result_row(full, "random_connected", "ball"), result_row(full, "random_connected", "ppr")
    ax.text(
        83,
        29.5,
        f"measured, full index: p95 {rb['p95_ms']:.1f} ms (ball) · {rp['p95_ms']:.1f} ms (PPR)",
        fontsize=5.2,
        color=C["ink2"],
        ha="center",
        va="center",
    )

    y2 = 8
    i = box(
        ax,
        140,
        y2,
        41,
        15,
        "S ≤ 64 facts with labels\nFACT · INFERRED · INDIRECT\n→ memory M (planned)",
        "plan",
    )
    j = box(
        ax,
        84,
        y2,
        48,
        15,
        "Small backbone LM + injection stream\n"
        "S-A cross-attention · S-B simplicial MoE\nS-C gated fusion (planned)",
        "plan",
    )
    k = box(
        ax,
        36,
        y2,
        40,
        15,
        "Answer + supporting subgraph\n+ G5 consistency verdict\n(accept · regenerate · abstain)",
        "plan",
    )
    hh = box(ax, 2, y2, 28, 15, "H2 heat field:\ncache + prefetch\n(planned)", "plan")
    arrow(ax, bottom(h), top(i))
    arrow(ax, left(i), right(j))
    arrow(ax, left(j), right(k))
    arrow(ax, top(hh), (66, y1), dashed=True, text="prefetch", offset=(-9, 1))

    handles = [
        Patch(fc=C["tint"], ec=C["ball"], lw=0.7, label="implemented and tested"),
        Patch(fc="white", ec=C["muted"], lw=0.7, ls=(0, (3, 2)), label="planned"),
        Patch(fc=C["neutral"], ec=C["ink2"], lw=0.7, label="data"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        ncol=3,
        handlelength=1.6,
    )
    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------- figure 2


def fig2_latency(raw: dict, reports: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 105 * MM), sharex=True, sharey=True)
    letters = iter("abcd")
    for r, idx in enumerate(("tiny", "full")):
        for col, wl in enumerate(WORKLOADS):
            ax = axes[r, col]
            hairline_grid(ax, "x")
            ax.axhline(0.95, color=C["grid"], lw=0.5)
            for m in ("ball", "ppr"):
                ms = raw[idx][f"{wl}.{m}.latency_ns"] / 1e6
                xs, ys = ecdf(ms)
                p95 = result_row(reports[idx], wl, m)["p95_ms"]
                label = f"{'ball' if m == 'ball' else 'PPR'} · p95 = {p95:.2g} ms"
                ax.step(xs, ys, where="post", color=C[m], lw=1.0, label=label)
                ax.plot([p95], [0.95], "o", ms=3.2, color=C[m], mec="white", mew=0.6, zorder=5)
            ax.axvline(TARGET_P95_MS, color=C["ink"], lw=0.6)
            ax.text(
                TARGET_P95_MS * 1.12,
                0.32,
                "50 ms target",
                rotation=90,
                fontsize=5.6,
                color=C["ink2"],
                va="center",
            )
            ax.legend(loc="best", frameon=False, handlelength=1.4, borderaxespad=0.3)
            n = raw[idx][f"{wl}.ball.latency_ns"].size
            name = "YAGO 4.6 tiny" if idx == "tiny" else "YAGO 4.6 full"
            ax.set_title(
                f"{next(letters)}   {name} · {WORKLOADS[wl]} (n = {n})", loc="left", fontweight="bold"
            )
            ax.set_xscale("log")
            ax.set_xlim(0.01, 200)
            ax.set_ylim(0, 1.02)
            if col == 0:
                ax.set_ylabel("fraction of queries ≤ t")
            if r == 1:
                ax.set_xlabel("latency per query t (ms, log scale)")
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, "fig2_latency_ecdf")


# --------------------------------------------------------------------------- figure 3


def fig3_work(raw: dict, reports: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 62 * MM), sharey=False)
    params = reports["full"]["params"]
    bound = 1.0 / (params["alpha"] * params["epsilon"])
    for ax, idx, letter in zip(axes, ("tiny", "full"), "ab", strict=True):
        hairline_grid(ax)
        for m in ("ball", "ppr"):
            w = raw[idx][f"random_connected.{m}.work"]
            t = raw[idx][f"random_connected.{m}.latency_ns"] / 1e6
            keep = w > 0
            ax.scatter(
                w[keep],
                t[keep],
                s=2.5,
                color=C[m],
                alpha=0.35,
                linewidths=0,
                rasterized=True,
                label=METHOD_LABEL[m],
            )
        ax.axvline(bound, color=C["ink"], lw=0.6)
        ax.text(
            bound * 0.85,
            0.012,
            f"ACL bound\nM/(αε) = {bound:,.0f}",
            fontsize=5.6,
            color=C["ink2"],
            ha="right",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, bound * 3)
        ax.set_ylim(0.008, 200)
        name = "YAGO 4.6 tiny" if idx == "tiny" else "YAGO 4.6 full"
        ax.set_title(f"{letter}   {name} · random connected seeds", loc="left", fontweight="bold")
        ax.set_xlabel("work per query (Σ d(u) over expansions / pushes)")
        ax.set_ylabel("latency (ms)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=3.5, color=C[m], label=METHOD_LABEL[m])
        for m in ("ball", "ppr")
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_work_vs_latency")


# --------------------------------------------------------------------------- figure 4


def fig4_degree(raw: dict, reports: dict, stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(COL1, 62 * MM))
    hairline_grid(ax)
    delta = reports["full"]["params"]["delta"]
    for idx, color, lw in (("tiny", C["muted"], 0.9), ("full", C["ink"], 1.0)):
        v = raw[idx]["degree.values"]
        cnt = raw[idx]["degree.counts"]
        keep = v > 0
        v, cnt = v[keep], cnt[keep]
        ccdf = np.cumsum(cnt[::-1])[::-1] / cnt.sum()
        frac = cnt[v > delta].sum() / cnt.sum()
        n = stats[idx]["nodes"]
        label = (
            f"YAGO 4.6 {idx}, {n / 1e6:.2f} M nodes\n{100 * frac:.1f} % with degree > Δ · max {int(v[-1]):,}"
        )
        ax.step(v, ccdf, where="post", color=color, lw=lw, label=label)
    ax.axvline(delta, color=C["ball"], lw=0.7)
    ax.text(delta * 1.15, 0.35, f"fan-out cap Δ = {delta}", fontsize=5.6, color=C["ink2"], va="center")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[::-1],
        labels[::-1],
        loc="lower left",
        frameon=True,
        fontsize=5.4,
        handlelength=1.4,
        framealpha=1.0,
    )
    leg.get_frame().set_edgecolor("none")
    leg.set_zorder(10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("node degree d (incidences, log scale)")
    ax.set_ylabel("P(degree ≥ d), connected nodes")
    ax.set_title("Degree distribution of the index", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_degree_ccdf")


# --------------------------------------------------------------------------- tables


def tables(stats: dict, reports: dict) -> None:
    rows = [
        ("Nodes (entities)", "nodes", "{:,}"),
        ("Hyperedges (entity--entity facts)", "edges", "{:,}"),
        ("Relation types", "relations", "{:,}"),
        ("Classes (taxonomy)", "classes", "{:,}"),
        ("(node, class) pairs", "type_pairs", "{:,}"),
        ("Edges with a time interval", "intervals", "{:,}"),
        ("Nodes with a Wikidata QID", "qids", "{:,}"),
        ("Index size (MB)", "index_mb", "{:,.1f}"),
        ("Build time (s)", "build_s", "{:,.1f}"),
        ("Peak memory during build (MB)", "peak_mb", "{:,}"),
    ]
    t1 = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & YAGO 4.6 tiny & YAGO 4.6 full \\",
        r"\midrule",
    ]

    def num(x: float, fmt: str) -> str:
        return fmt.format(x).replace(",", "{,}")

    for label, key, fmt in rows:
        t1.append(f"{label} & {num(stats['tiny'][key], fmt)} & {num(stats['full'][key], fmt)} \\\\")
    t1 += [r"\bottomrule", r"\end{tabular}", ""]
    (FIG / "table1_index.tex").write_text("\n".join(t1), encoding="utf-8")

    t2 = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Index & Seeds & Method & $n$ & p50 & p95 & p99 & max & mean nodes \\",
        r" & & & & \multicolumn{4}{c}{latency (ms)} & \\",
        r"\midrule",
    ]
    for idx in ("tiny", "full"):
        for wl, wname in WORKLOADS.items():
            for m in ("ball", "ppr"):
                r = result_row(reports[idx], wl, m)
                t2.append(
                    f"{idx} & {wname} & {m} & {r['queries']} & {r['p50_ms']:.2f} & "
                    f"\\textbf{{{r['p95_ms']:.2f}}} & "
                    f"{r['p99_ms']:.2f} & {r['max_ms']:.2f} & {r['mean_nodes']:.1f} \\\\"
                )
        if idx == "tiny":
            t2.append(r"\midrule")
    t2 += [r"\bottomrule", r"\end{tabular}", ""]
    (FIG / "table2_latency.tex").write_text("\n".join(t2), encoding="utf-8")


def main() -> int:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    stats = json.loads((DATA / "build_stats.json").read_text(encoding="utf-8"))
    reports = {
        idx: json.loads((REPORTS / f"2026-11-08_slice_latency_yago46_{idx}.json").read_text(encoding="utf-8"))
        for idx in ("tiny", "full")
    }
    raw = {idx: dict(np.load(DATA / f"latency_{idx}.npz")) for idx in ("tiny", "full")}
    fig1_architecture(stats, reports["full"])
    fig2_latency(raw, reports)
    fig3_work(raw, reports)
    fig4_degree(raw, reports, stats)
    tables(stats, reports)
    for p in sorted(FIG.glob("*.*")):
        print(p.relative_to(ROOT), p.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
