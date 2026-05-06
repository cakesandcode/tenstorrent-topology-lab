"""
Workloads exercised against each topology.

Three patterns:
  - neighbor_exchange : every host talks only to topological neighbors. Torus's
                        home turf. Latency = max neighbor-pair latency.
  - all_reduce_ring   : ring all-reduce, the dominant collective in distributed
                        training. 2*(N-1) effective hops along a ring.
  - all_to_all        : every host sends to every other host. The pattern that
                        breaks neighbor-only topologies as N grows. Latency =
                        max pairwise host-host latency, which is the network
                        diameter scaled by per-hop cost.

All three return a single number in nanoseconds: the "completion time" of one
collective. We model completion as the slowest path / sum-of-ring-hops, not
the total link-bandwidth time, because the load-bearing claim in the post is
about latency, not bandwidth.

For computational efficiency, distances on the regular topologies (torus,
fat-tree) are computed analytically rather than via all-pairs Dijkstra. The
all-pairs approach was the original implementation but does not scale past
~256 hosts on a fat-tree.
"""

from __future__ import annotations

import networkx as nx

from latency_model import CHIP_HOP_NS, SWITCH_HOP_NS


# ---------------------------------------------------------------------------
# 2D torus (n x n)
# ---------------------------------------------------------------------------
def _torus_2d_diameter_hops(n: int) -> int:
    """Max distance on n x n 2D torus is n/2 + n/2 = n hops (n even)."""
    return (n // 2) + (n // 2) if n % 2 == 0 else (n - 1)


def torus_2d_metrics(n: int) -> dict[str, float]:
    """Closed-form latencies for n x n 2D torus."""
    N = n * n
    # neighbor exchange: every neighbor is exactly 1 chip-hop away.
    neighbor = 1 * CHIP_HOP_NS
    # all-reduce ring: along a row-major host ordering, most adjacent pairs are
    # 1 hop apart. End-of-row to start-of-next-row is also 1 hop on a torus
    # (wraparound) only at column boundary; otherwise a row-major snake
    # wraparound makes adjacent steps cost 1 hop. Sum = N hops; ring all-reduce
    # cost = 2*(N-1)/N * (sum of ring step costs). With every step = 1 hop:
    ring_step_total = N * 1 * CHIP_HOP_NS
    all_reduce = ring_step_total * 2 * (N - 1) / N
    # all-to-all completion = network diameter.
    all_to_all = _torus_2d_diameter_hops(n) * CHIP_HOP_NS
    return {
        "neighbor_exchange": neighbor,
        "all_reduce_ring": all_reduce,
        "all_to_all": all_to_all,
    }


# ---------------------------------------------------------------------------
# 3D torus (a x b x c)
# ---------------------------------------------------------------------------
def torus_3d_metrics(a: int, b: int, c: int) -> dict[str, float]:
    N = a * b * c
    neighbor = 1 * CHIP_HOP_NS
    # row-major host ordering: most adjacent pairs are 1 hop, occasional jumps
    # at slab/row boundaries. We approximate as 1 hop per ring step (clean
    # bound; actual is slightly more in occasional steps).
    ring_step_total = N * 1 * CHIP_HOP_NS
    all_reduce = ring_step_total * 2 * (N - 1) / N
    diameter = (a // 2) + (b // 2) + (c // 2)
    all_to_all = diameter * CHIP_HOP_NS
    return {
        "neighbor_exchange": neighbor,
        "all_reduce_ring": all_reduce,
        "all_to_all": all_to_all,
    }


# ---------------------------------------------------------------------------
# Fat-tree (k-ary)
# ---------------------------------------------------------------------------
def fat_tree_metrics(graph: nx.Graph, hosts: list, k: int) -> dict[str, float]:
    """
    Fat-tree distances are quantized:
      - same edge switch:        2 switch hops   (host-edge-host)
      - same pod, different edge: 4 switch hops  (host-edge-agg-edge-host)
      - different pod:            6 switch hops  (host-edge-agg-core-agg-edge-host)
    Diameter = 6 switch hops. Two random hosts are most likely cross-pod.

    For the host-by-host ring ordering used in all-reduce ring, we use the
    natural enumeration order ("host", pod, edge_idx, host_in_edge). Adjacent
    pairs in this ordering are mostly within an edge switch (cheap) but at
    pod / edge boundaries pay the longer paths. We compute it exactly by
    walking the ring once.
    """
    half = k // 2
    hosts_per_edge = half
    hosts_per_pod = half * half  # half edges * half hosts per edge

    def pairwise_hops(a: tuple, b: tuple) -> int:
        # a and b are ("host", pod, edge_idx, host_in_edge)
        if a == b:
            return 0
        if a[1] == b[1]:                # same pod
            if a[2] == b[2]:            # same edge switch
                return 2
            return 4
        return 6

    def pairwise_latency(a: tuple, b: tuple) -> float:
        return pairwise_hops(a, b) * SWITCH_HOP_NS

    N = len(hosts)

    # neighbor exchange: pick each host's 4 closest peers and report worst.
    # Within an edge switch there are (half-1) other hosts at distance 2.
    # If half-1 >= 4 we have all neighbors at 2 hops. Else we cross to peer
    # edge in same pod for distance 4.
    if half - 1 >= 4:
        neighbor = 2 * SWITCH_HOP_NS
    else:
        neighbor = 4 * SWITCH_HOP_NS

    # all-reduce ring: walk hosts in list order, sum adjacent-pair latencies,
    # apply 2*(N-1)/N factor.
    ring_total = 0.0
    for i in range(N):
        ring_total += pairwise_latency(hosts[i], hosts[(i + 1) % N])
    all_reduce = ring_total * 2 * (N - 1) / N

    # all-to-all: diameter.
    all_to_all = 6 * SWITCH_HOP_NS

    return {
        "neighbor_exchange": neighbor,
        "all_reduce_ring": all_reduce,
        "all_to_all": all_to_all,
    }
