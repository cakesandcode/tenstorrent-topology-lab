"""
Topology builders for the Tenstorrent networking hypothesis test.

Three topologies are built as NetworkX graphs:
  - 2D torus: n x n grid, 4 neighbors each, wraparound. Tenstorrent's choice.
  - 3D torus: a x b x c grid, 6 neighbors each, wraparound. Google TPU's choice.
  - Fat-tree: 2-level Clos with k pods (Al-Fares et al. canonical form). The
    switched fabric Tenstorrent argues against.

Edges carry an attribute "kind" in {"chip", "switch"} so the latency model can
weight chip-to-chip hops differently from switch hops. In the torus topologies
every edge is "chip"; in the fat-tree, host->leaf and leaf->aggregation->core
hops are all "switch".

Every builder returns (graph, host_nodes) so callers know which nodes carry
end-to-end traffic versus which are switches in the middle. For a torus,
host_nodes == all nodes.
"""

from __future__ import annotations

import networkx as nx


def make_2d_torus(n: int) -> tuple[nx.Graph, list]:
    """n x n 2D torus. n^2 hosts, each with 4 chip-to-chip neighbors."""
    g = nx.Graph()
    hosts = [(i, j) for i in range(n) for j in range(n)]
    g.add_nodes_from(hosts, role="host")
    for i in range(n):
        for j in range(n):
            # right neighbor (wraparound)
            g.add_edge((i, j), (i, (j + 1) % n), kind="chip")
            # down neighbor (wraparound)
            g.add_edge((i, j), ((i + 1) % n, j), kind="chip")
    return g, hosts


def make_3d_torus(a: int, b: int, c: int) -> tuple[nx.Graph, list]:
    """a x b x c 3D torus. a*b*c hosts, each with 6 chip-to-chip neighbors."""
    g = nx.Graph()
    hosts = [(i, j, k) for i in range(a) for j in range(b) for k in range(c)]
    g.add_nodes_from(hosts, role="host")
    for i in range(a):
        for j in range(b):
            for k in range(c):
                g.add_edge((i, j, k), ((i + 1) % a, j, k), kind="chip")
                g.add_edge((i, j, k), (i, (j + 1) % b, k), kind="chip")
                g.add_edge((i, j, k), (i, j, (k + 1) % c), kind="chip")
    return g, hosts


def make_fat_tree(k: int) -> tuple[nx.Graph, list]:
    """
    Standard k-ary fat-tree (Al-Fares et al., SIGCOMM 2008).

    For pod size k:
      - k pods
      - each pod has k/2 edge switches and k/2 aggregation switches
      - each edge switch connects to k/2 hosts
      - (k/2)^2 core switches, each connected to one aggregation switch per pod

    Total hosts = k^3 / 4. k must be even.

    Every edge in this graph is kind="switch" because every hop traverses a
    switching device. Hosts are leaves.
    """
    if k % 2 != 0:
        raise ValueError(f"fat-tree pod size k must be even, got {k}")

    g = nx.Graph()
    half = k // 2
    hosts: list = []

    # core switches
    core = [("core", i, j) for i in range(half) for j in range(half)]
    g.add_nodes_from(core, role="core")

    for pod in range(k):
        # aggregation switches in this pod
        agg = [("agg", pod, i) for i in range(half)]
        # edge switches in this pod
        edge = [("edge", pod, i) for i in range(half)]
        g.add_nodes_from(agg, role="agg")
        g.add_nodes_from(edge, role="edge")

        # full bipartite between agg and edge within a pod
        for a in agg:
            for e in edge:
                g.add_edge(a, e, kind="switch")

        # each aggregation switch i connects to all core switches with first index i
        for i, a in enumerate(agg):
            for j in range(half):
                g.add_edge(a, ("core", i, j), kind="switch")

        # hosts: each edge switch hangs k/2 hosts
        for i, e in enumerate(edge):
            for h in range(half):
                host = ("host", pod, i, h)
                g.add_node(host, role="host")
                g.add_edge(host, e, kind="switch")
                hosts.append(host)

    return g, hosts


def topology_summary(graph: nx.Graph, hosts: list) -> dict:
    """Cheap diagnostic: node count, edge count, host count, diameter."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "hosts": len(hosts),
        # diameter on the host-induced subgraph would be more accurate but is
        # expensive; full-graph diameter is good enough for sanity checks.
        "diameter": nx.diameter(graph) if nx.is_connected(graph) else -1,
    }
