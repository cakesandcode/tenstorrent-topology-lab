"""
Sweep cluster size N across 2D torus, 3D torus, and fat-tree. For each
(topology, N, workload) triple, compute completion latency in nanoseconds.
Plot to output/latency_vs_N.png with three subplots, one per workload.

Run:
    python3 run.py

Produces:
    output/latency_vs_N.png
    stdout: markdown table of results suitable for paste into discussion-summary.md

Implementation note: distances are computed analytically (closed-form) for
each topology rather than via all-pairs Dijkstra. This keeps the runtime
laptop-friendly even at N=16384, where an all-pairs approach would not finish.
The analytical formulas live in workloads.py; we still build the full graphs
in topologies.py so readers can inspect / extend them if they want.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from topologies import make_2d_torus, make_3d_torus, make_fat_tree
from workloads import torus_2d_metrics, torus_3d_metrics, fat_tree_metrics


# Cluster sizes to sweep. Sized to make the all-to-all latency crossover visible.
# 2D torus diameter scales as sqrt(N) per hop; fat-tree diameter is fixed at
# 6 switch hops. The structural-latency crossover where fat-tree's flat
# 1500 ns ceiling beats torus's growing diameter happens around N=2500-3000.
SIZES = [
    # (label, 2D torus n, 3D torus (a,b,c) or None, fat-tree k or None)
    ("16",     4,    (2, 2, 4),     4),
    ("64",     8,    (4, 4, 4),     None),
    ("128",    None, None,           8),
    ("256",    16,   (4, 8, 8),     None),
    ("432",    None, None,           12),
    ("1024",   32,   (8, 8, 16),     16),   # fat-tree(16) = 1024 hosts
    ("4096",   64,   (16, 16, 16),  None),
    ("16384",  128,  None,           None),
    ("65536",  256,  None,           None),
]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "output")
    os.makedirs(output_dir, exist_ok=True)

    results: dict[str, dict[str, list[tuple[int, float]]]] = {
        "2D torus":  {"neighbor_exchange": [], "all_reduce_ring": [], "all_to_all": []},
        "3D torus":  {"neighbor_exchange": [], "all_reduce_ring": [], "all_to_all": []},
        "fat-tree":  {"neighbor_exchange": [], "all_reduce_ring": [], "all_to_all": []},
    }

    print(f"{'topology':<12} {'N':>7}  {'neighbor':>10}  {'all-reduce':>12}  {'all-to-all':>11}")
    print("-" * 60)

    def record(topo: str, n: int, m: dict) -> None:
        for w, v in m.items():
            results[topo][w].append((n, v))
        print(f"{topo:<12} {n:>7}  {m['neighbor_exchange']:>10.0f}  "
              f"{m['all_reduce_ring']:>12.0f}  {m['all_to_all']:>11.0f}")

    for label, t2d, t3d, ft_k in SIZES:
        if t2d is not None:
            record("2D torus", t2d * t2d, torus_2d_metrics(t2d))
        if t3d is not None:
            a, b, c = t3d
            record("3D torus", a * b * c, torus_3d_metrics(a, b, c))
        if ft_k is not None:
            g, hosts = make_fat_tree(ft_k)
            record("fat-tree", len(hosts), fat_tree_metrics(g, hosts, ft_k))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    workload_titles = {
        "neighbor_exchange": "Neighbor exchange\n(torus's home turf)",
        "all_reduce_ring":   "Ring all-reduce\n(Tenstorrent's target workload)",
        "all_to_all":        "All-to-all\n(fat-tree wins at scale)",
    }
    colors = {"2D torus": "#1f77b4", "3D torus": "#2ca02c", "fat-tree": "#d62728"}
    markers = {"2D torus": "o", "3D torus": "s", "fat-tree": "^"}

    for ax, (workload, title) in zip(axes, workload_titles.items()):
        for topo, color in colors.items():
            data = sorted(results[topo][workload])
            if not data:
                continue
            xs = [n for n, _ in data]
            ys = [v for _, v in data]
            ax.plot(xs, ys, marker=markers[topo], color=color, label=topo,
                    linewidth=2, markersize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cluster size N (hosts)")
        ax.set_ylabel("Latency (ns, log scale)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="best")

    # Annotate the all-to-all crossover. The fat-tree all-to-all is flat at
    # 1500 ns; the 2D torus crosses it where N satisfies sqrt(N)*30ns >= 1500ns,
    # i.e., N >= ~2500.
    ax_a2a = axes[2]
    ax_a2a.axhline(y=1500, color="#d62728", linestyle=":", alpha=0.6, linewidth=1.2)
    ax_a2a.annotate(
        "fat-tree ceiling: 1500 ns,\nflat in N",
        xy=(20, 1500), xytext=(20, 600),
        fontsize=9, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.6),
    )
    ax_a2a.annotate(
        "crossover ~ N=2500",
        xy=(2500, 1500), xytext=(8000, 350),
        fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#333", alpha=0.6),
    )

    fig.suptitle(
        "Switch-less torus vs switched fat-tree: where each topology wins",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "Structural latency only — does NOT model bandwidth contention or congestion. "
        "Per-hop costs: chip-to-chip 30 ns, switch hop 250 ns.",
        ha="center", fontsize=9, style="italic", color="#666",
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, "latency_vs_N.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out_path}")

    # Summary observations to paste into the post
    print("\n## Summary observations\n")
    print("- Neighbor exchange: 2D torus is a 30 ns floor, constant in N.")
    print("- Ring all-reduce: torus is 4-8x faster per ring step (chip hop 30 ns")
    print("  vs switch hop 250 ns). Both grow linearly with N.")
    print("- All-to-all: 2D torus diameter scales as sqrt(N); fat-tree diameter")
    print("  is fixed at 6 hops = 1500 ns. Structural-latency crossover where")
    print("  fat-tree beats 2D torus on all-to-all appears around N=2500 in")
    print("  this model. Below that, torus is still ahead on pure latency, but")
    print("  fat-tree's bisection-bandwidth advantage (NOT modeled here) is the")
    print("  practical reason switched fabrics dominate at scale.")


if __name__ == "__main__":
    main()
