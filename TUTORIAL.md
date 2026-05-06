# Tutorial — testing the simulator step by step

A hands-on walkthrough that does two things at once:

1. **Sets up the simulator** so you can run it on your own machine.
2. **Validates the model** by checking the simulator's output against
   graph-theoretic facts you can compute by hand. If you're going to
   reference the chart in your own analysis, you should know the model's
   limits and where its numbers come from.

Each step is structured the same way: run a command → see expected output →
short note explaining what that output proves about the model.

If anything in the expected output diverges from what you actually see, the
simulator is broken in some way and the chart is no longer trustworthy.

---

## Step 0 — Setup

Clone the repo and install the three dependencies:

```bash
git clone https://github.com/cakesandcode/tenstorrent-topology-lab.git
cd tenstorrent-topology-lab
python3 -m pip install --quiet networkx numpy matplotlib
```

A `pip` version warning is harmless; the install still succeeds. On macOS,
you'll be using the system Python (Xcode's), which is fine.

---

## Step 1 — Confirm the topology builders make the graphs they claim

Drop into Python from inside the repo directory:

```bash
python3
```

```python
from topologies import make_2d_torus, make_3d_torus, make_fat_tree, topology_summary

g, hosts = make_2d_torus(4)
print(topology_summary(g, hosts))
```

**Expected:** `{'nodes': 16, 'edges': 32, 'hosts': 16, 'diameter': 4}`

**What this proves:** a 4×4 2D torus has 16 hosts, each with 4 neighbors
(so 16×4/2 = 32 edges), and the worst-case host pair is 2 hops in each
dimension = 4 hops total. If diameter ≠ 4, the wraparound is broken.

```python
g, hosts = make_3d_torus(4, 4, 4)
print(topology_summary(g, hosts))
```

**Expected:** `{'nodes': 64, 'edges': 192, 'hosts': 64, 'diameter': 6}`

**What this proves:** 64 hosts, 6 neighbors each (192 edges = 64×6/2),
diameter 6 = 2+2+2 hops across three wrapped dimensions. The cube-root
behavior the chart depends on.

```python
g, hosts = make_fat_tree(4)
print(topology_summary(g, hosts))
```

**Expected:** `{'nodes': 36, 'edges': 64, 'hosts': 16, 'diameter': 6}`

**What this proves:** k=4 fat-tree has k³/4 = 16 hosts, plus the switch
infrastructure (4 pods × 2 edge + 4 pods × 2 agg + 4 core = 20 switches +
16 hosts = 36 nodes). Diameter 6 = host→edge→agg→core→agg→edge→host.

If all three match, the graph builders are sound.

---

## Step 2 — Confirm the per-hop latencies are what you think they are

```python
from latency_model import CHIP_HOP_NS, SWITCH_HOP_NS
print(f"chip:   {CHIP_HOP_NS} ns")
print(f"switch: {SWITCH_HOP_NS} ns  (ratio: {SWITCH_HOP_NS/CHIP_HOP_NS:.1f}x)")
```

**Expected:**
```
chip:   30 ns
switch: 250 ns  (ratio: 8.3x)
```

**What this proves:** these are the only numerical inputs to the entire
simulator. The 8.3× ratio is what produces the "torus wins at small N,
fat-tree wins at large N for all-to-all" curve. Change either number and
re-run, and the crossover N moves predictably. This is your sensitivity
lever (Step 5).

---

## Step 3 — Verify each workload metric on a known-small case

```python
from workloads import torus_2d_metrics, torus_3d_metrics, fat_tree_metrics
from topologies import make_fat_tree

# 4x4 2D torus = 16 hosts, diameter 4 hops
print(torus_2d_metrics(4))
```

**Expected:** `{'neighbor_exchange': 30.0, 'all_reduce_ring': 900.0, 'all_to_all': 120.0}`

**What this proves:**
- `neighbor_exchange = 30 ns` — one chip-hop, exactly as expected.
- `all_to_all = 4 hops × 30 ns = 120 ns` — diameter × per-hop. ✓
- `all_reduce_ring = 16 × 30 × 2(15)/16 = 900 ns` — N steps × per-step × ring formula. ✓

```python
# 4x4x4 3D torus = 64 hosts, diameter 6 hops
print(torus_3d_metrics(4, 4, 4))
```

**Expected:** `{'neighbor_exchange': 30.0, 'all_reduce_ring': 3780.0, 'all_to_all': 180.0}`

**What this proves:**
- `all_to_all = 6 × 30 = 180 ns` — diameter is half-of-each-dimension × 3 = 6.
  The cube-root scaling that lets 3D torus stay below the fat-tree ceiling
  at large N.

```python
# k=4 fat-tree = 16 hosts
g, hosts = make_fat_tree(4)
print(fat_tree_metrics(g, hosts, 4))
```

**Expected:** `{'neighbor_exchange': 1000.0, 'all_reduce_ring': ~26250, 'all_to_all': 1500.0}`

**What this proves:**
- `all_to_all = 6 × 250 = 1500 ns` — 6 switch hops, diameter constant in N. ✓
- `neighbor_exchange = 1000 ns = 4 switch hops` — k/2 = 2, so only 2 hosts
  share an edge switch. The 4 closest hosts include cross-edge hosts in the
  same pod (4 switch hops). The "1000 ns vs 500 ns" branch in
  `fat_tree_metrics` triggers for k < 10.

If any of these mismatch, the workload formulas are wrong.

---

## Step 4 — Run the full sweep and see the chart

Exit Python (`Ctrl-D` or `exit()`), back at the shell:

```bash
python3 run.py
```

Verify in the printed table:

1. **Neighbor exchange row for every torus is exactly 30 ns**, regardless of
   N. Confirms the "constant in N" claim.
2. **All-to-all row for fat-tree is exactly 1500 ns**, regardless of N
   (when fat-tree is in the row). Confirms the flat ceiling.
3. **2D torus all-to-all crosses 1500 ns somewhere between N=1024 (960 ns)
   and N=4096 (1920 ns).** This is the crossover the chart annotates at
   N ≈ 2500.

Open the generated chart:

```bash
open output/latency_vs_N.png   # macOS
xdg-open output/latency_vs_N.png   # Linux
```

---

## Step 5 — Sensitivity: move the per-hop costs, watch the crossover move

The crossover N for 2D torus all-to-all is computable in closed form:

```
N_crossover = (SWITCH_HOP_NS × 6 / CHIP_HOP_NS)²
```

With defaults: (250 × 6 / 30)² = 50² = 2500. ✓

Stress-test it both ways:

```bash
# What if switches were 2.5x faster (250 -> 100 ns)?
python3 -c "
SWITCH = 100.0; CHIP = 30.0
print(f'New crossover N ≈ {(SWITCH * 6 / CHIP) ** 2:.0f}')
"
```

**Expected:** `New crossover N ≈ 400`

**What this proves:** if switches got 2.5× faster, the 2D torus advantage on
all-to-all latency would erode much sooner — at ~400 hosts instead of ~2500.
The crossover is *driven by* the switch-vs-chip ratio. This is exactly the
kind of parameter a critical reader might push on.

The opposite stress test:

```bash
# What about typical (not cutting-edge) switches at ~1 us per hop?
python3 -c "
SWITCH = 1000.0; CHIP = 30.0
print(f'New crossover N ≈ {(SWITCH * 6 / CHIP) ** 2:.0f}')
"
```

**Expected:** `New crossover N ≈ 40000`

**What this proves:** against typical (not cutting-edge) Ethernet switches,
the 2D torus stays ahead on all-to-all latency to ~40,000 hosts. The
default 250 ns is the *most generous* assumption for fat-tree; using
realistic 1 µs would make the torus look much better. The chart picks
250 ns deliberately as the toughest test for the torus argument.

---

## Step 6 — Check the analytical diameter formula matches NetworkX's diameter

This is the test that "I trust the closed-form math" rests on. The
simulator uses analytical formulas (not Dijkstra) for performance; this
step verifies the formulas agree with brute force.

```python
import networkx as nx
from topologies import make_2d_torus, make_3d_torus

for n in [4, 8, 16, 32]:
    g, hosts = make_2d_torus(n)
    nx_diameter = nx.diameter(g)
    formula = n  # n/2 + n/2 = n for even n
    print(f"2D torus {n}x{n}:  NetworkX={nx_diameter}, formula={formula},  match={nx_diameter == formula}")
```

**Expected:** all four rows say `match=True`.

```python
for a, b, c in [(4, 4, 4), (4, 8, 8), (8, 8, 16)]:
    g, hosts = make_3d_torus(a, b, c)
    nx_diameter = nx.diameter(g)
    formula = (a // 2) + (b // 2) + (c // 2)
    print(f"3D torus {a}x{b}x{c}: NetworkX={nx_diameter}, formula={formula}, match={nx_diameter == formula}")
```

**Expected:** all three say `match=True`.

If both blocks pass, the analytical formulas are sound and the chart is
trustworthy at the structural level.

---

## Step 7 — Where the model is most likely wrong

If you cite this chart in your own analysis, be ready to defend or concede on:

1. **Ring all-reduce on fat-tree** assumes hosts are visited in enumeration
   order. A real all-reduce library would pick a smarter ordering. Fat-tree's
   all-reduce numbers are pessimistic; even with a smarter ordering they'd be
   2-4× worse than torus per ring step (every step is still a switch hop),
   but not 18-19× worse.
2. **Neighbor exchange for fat-tree** uses "4 closest hosts" because fat-tree
   has no native notion of geometric neighbor. This penalizes fat-tree on a
   workload that doesn't really apply to it. In real systems nobody would
   deploy fat-tree for stencil-style workloads.
3. **No bandwidth model.** The big one. The all-to-all completion time you'd
   measure on real hardware is dominated by link bandwidth, not per-hop
   latency, for any non-trivial message size.
4. **Static topology for fat-tree.** Real fat-trees have ECMP and adaptive
   routing; this model assumes shortest-path always.

Defensive posture if a reader pushes on any of these: the model shows a
*structural-latency* trade-off; bandwidth and routing are separate axes that
this model deliberately doesn't capture. The README documents these limits
explicitly.

---

## Quick reference — sanity checks at a glance

| Topology | N | neighbor (ns) | all-to-all (ns) | Why |
|---|---:|---:|---:|---|
| 2D torus 4×4 | 16 | 30 | 120 | 1 chip-hop · diameter 4 hops |
| 3D torus 4×4×4 | 64 | 30 | 180 | 1 chip-hop · diameter 6 hops |
| 2D torus 32×32 | 1024 | 30 | 960 | constant · diameter 32 hops |
| 2D torus 64×64 | 4096 | 30 | 1920 | constant · diameter 64 hops (crossed!) |
| fat-tree k=4 | 16 | 1000 | 1500 | 4-hop · 6-hop diameter |
| fat-tree k=16 | 1024 | 500 | 1500 | 2-hop · 6-hop diameter (constant) |

If your run produces numbers different from these, the simulator broke
somewhere between when it was committed and when you ran it.
