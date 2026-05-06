"""
Latency model for the topology hypothesis test.

We assign a per-edge latency cost based on the edge "kind":
  - "chip"   : ~30 ns   (chip-to-chip direct, Tenstorrent style)
  - "switch" : ~250 ns  (cut-through switch hop, ~Broadcom Tomahawk Ultra)

These are deliberately optimistic-but-defensible numbers for cut-through
switches; older / store-and-forward designs are 1-2 microseconds. The 30 ns
chip-to-chip number is the right order of magnitude for high-speed SerDes
plus a short DAC cable; the actual on-die routing inside Tenstorrent's NoC is
faster, but we're modeling chip-to-chip hops on the public network here.

The model captures only structural latency (hop count x per-hop cost). It
explicitly does NOT capture:
  - link-level bandwidth saturation
  - congestion under load
  - software stack / driver overhead
  - serialization / deserialization delays for full packet sizes

This is intentional: the post is making a *topology* argument, and a
topology-level model is the right level of abstraction. The README documents
these limitations because hostile readers will probe them.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


CHIP_HOP_NS: float = 30.0
SWITCH_HOP_NS: float = 250.0


def edge_weight(kind: str) -> float:
    if kind == "chip":
        return CHIP_HOP_NS
    if kind == "switch":
        return SWITCH_HOP_NS
    raise ValueError(f"unknown edge kind: {kind}")


def weight_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy with 'weight' attribute set per-edge from edge kind."""
    g = graph.copy()
    for u, v, data in g.edges(data=True):
        data["weight"] = edge_weight(data["kind"])
    return g


def pairwise_host_latencies(graph: nx.Graph, hosts: list) -> np.ndarray:
    """
    Returns an N x N matrix of host-to-host latencies in nanoseconds, where N
    is the host count. latency[i, j] is the shortest weighted path from
    host[i] to host[j]; latency[i, i] is zero.
    """
    weighted = weight_graph(graph)
    host_index = {h: i for i, h in enumerate(hosts)}
    n = len(hosts)
    latency = np.zeros((n, n), dtype=np.float64)

    # Dijkstra from each host. networkx's all-pairs would also work, but we
    # only care about host-host pairs and there can be many switches in the
    # graph for fat-trees. Per-host single-source is fine at our scales.
    for src in hosts:
        lengths = nx.single_source_dijkstra_path_length(weighted, src)
        i = host_index[src]
        for dst, dist in lengths.items():
            if dst in host_index:
                latency[i, host_index[dst]] = dist

    return latency
