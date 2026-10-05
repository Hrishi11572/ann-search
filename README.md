# ANN Search

### From KNN Graphs to Navigable Small Worlds

This repository is a hands-on exploration of **Approximate Nearest Neighbor (ANN) Search**, with the goal of understanding how graph-based indexes such as **NSW (Navigable Small World)** and eventually **HNSW (Hierarchical Navigable Small World)** work from first principles.

Rather than treating ANN libraries as black boxes, this project builds the ideas incrementally — starting from a simple KNN graph and studying why naive graph search fails, how long-range connections improve navigability, and how NSW uses these ideas to perform efficient approximate nearest-neighbor search.

> **The goal is to understand the algorithm, not just use the library.**

---

## Motivation

For a query vector (q), the exact nearest-neighbor problem requires comparing (q) against every point in the dataset:

$$
[
x^\* = \arg\min\_{x_i \in X} d(q,x_i)
]
$$

For (N) points, this requires approximately (O(N)) distance computations per query.

This becomes expensive for large vector datasets.

Approximate Nearest Neighbor methods trade a small amount of accuracy for substantially faster search by constructing an **index** over the dataset.

This project explores one particularly interesting approach:

**represent the dataset as a navigable graph.**

Instead of comparing the query with every point, we start from one or more points and navigate through the graph toward points increasingly closer to the query.

---

## Learning Path

The notebooks follow the conceptual development of graph-based ANN algorithms:

```text
KNN Graph
   │
   ▼
Greedy Search
   │
   ▼
Why Greedy Search Fails
   │
   ▼
Long-Range Connections
   │
   ▼
Navigable Small World (NSW)
   │
   ▼
NSW Experiments
   │
   ▼
Hierarchical NSW (HNSW)
```

Each notebook isolates one idea before moving to the next.

---

## Repository Structure

```text
ann-search/
│
└── mini-ann/
    │
    └── notebooks/
        ├── 01_knn_graph.ipynb
        ├── 02_greedy_failure.ipynb
        ├── 03_long-range-benefits.ipynb
        ├── 04_nsw.ipynb
        └── 05_nsw_experiments.ipynb
```

### `01_knn_graph.ipynb`

Introduces the basic **K-Nearest Neighbor graph**.

Given a collection of vectors, every point is connected to its (k) nearest neighbors.

The notebook constructs and visualizes the resulting graph.

Conceptually:

```text
             x2
            /  \
           /    \
         x1 ---- x3
          \       \
           \       x4
            \     /
              x5
```

The KNN graph provides the basic data structure that later ANN algorithms build upon.

---

### `02_greedy_failure.ipynb`

A graph alone does not guarantee efficient search.

This notebook studies **greedy graph search** and demonstrates situations where greedy navigation can become trapped in a poor local region.

At every step, we move to the neighbor that is closest to the query:

```text
current node
     │
     ▼
inspect neighbors
     │
     ▼
choose closest neighbor
     │
     ▼
repeat
```

The important observation is that:

> A graph can have good local connectivity while still being difficult to navigate globally.

This motivates the need for **long-range connections**.

---

### `03_long-range-benefits.ipynb`

This notebook investigates why long-range edges can dramatically improve graph navigability.

Local edges allow us to move efficiently within a neighborhood, while long-range edges allow us to quickly move between distant regions of the dataset.

The basic intuition is:

```text
Without long-range edges:

A ─ A ─ A ─ A ─ A ─ B ─ B ─ B ─ B


With long-range edges:

A ─ A ─ A ─ A
 \          \
  \          ────────── B ─ B ─ B
   \
    B
```

This provides the intuition behind **small-world networks** and eventually NSW.

---

### `04_nsw.ipynb`

This notebook implements a basic **Navigable Small World (NSW)** graph.

The graph is constructed incrementally. When a new point is inserted, a modified nearest-neighbor search is used to identify candidate neighbors.

The implementation contains the main components needed for an ANN graph:

- distance computation
- graph construction
- incremental insertion
- approximate graph search
- multi-start search
- exact nearest-neighbor search for comparison
- graph visualization
- search-path visualization
- recall evaluation

The approximate search can then be compared against brute-force exact search.

---

### `05_nsw_experiments.ipynb`

The final notebook contains experiments on the NSW implementation.

The purpose is to investigate the trade-off between:

- search accuracy
- number of distance evaluations / search steps
- graph connectivity
- search parameters

For example, the experiments vary the number of search candidates / starting points and evaluate how this affects ANN performance.

A representative evaluation looks at:

```text
k = 2   → recall
k = 4   → recall
k = 8   → recall
k = 16  → recall
k = 32  → recall
```

This makes the **accuracy–computation trade-off** visible rather than treating ANN as simply "fast but approximate."

---

## Core Idea

The central idea behind the project can be summarized as:

```text
Dataset
   │
   ▼
Construct graph
   │
   ▼
Each point → nearby points
   │
   ▼
Add navigable connections
   │
   ▼
Start from an entry point
   │
   ▼
Greedily move toward query
   │
   ▼
Reach approximate nearest neighbor
```

For a query (q), the search repeatedly examines the neighborhood of the current vertex (v):

$$
[
N(v) = {u_1,u_2,\ldots,u_k}
]
$$

and moves toward a neighbor that reduces the distance

$$
[
d(q,u) < d(q,v).
]
$$

Unlike brute-force search, we do not need to evaluate the distance from (q) to every point in the dataset.

---

## Exact vs Approximate Search

The repository also maintains an **exact nearest-neighbor search** implementation to provide ground truth.

### Exact Search

```python
distances = np.linalg.norm(X - query, axis=1)
nearest = np.argmin(distances)
```

Complexity:
$$
[
O(N)
]
$$

distance evaluations per query.

### Approximate Graph Search

Instead of scanning all (N) points:

```text
query
  │
  ▼
entry point
  │
  ▼
neighbor
  │
  ▼
better neighbor
  │
  ▼
...
  │
  ▼
approximate NN
```

The number of visited nodes can be substantially smaller than (N).

The cost is that the returned point is not guaranteed to be the exact nearest neighbor.

---

## Evaluation

The main metric used in the experiments is **Recall\@k**.

For a query (q), let:

- (E_k(q)) = exact (k)-nearest neighbors
- (A_k(q)) = approximate (k)-nearest neighbors

Then:

$$
\frac{|E_k(q)\cap A_k(q)|}{k}
$$
averaged over the query set.

For example:

```text
Recall = 1.0
```

means the approximate search recovered all true nearest neighbors.

The project also examines the amount of search performed, allowing us to study the fundamental ANN trade-off:

$$
[
\boxed{\text{Search Cost} \quad \leftrightarrow \quad \text{Recall}}
]
$$
---

## Why NSW?

The interesting part of NSW is that it combines two seemingly competing requirements:

### Local connectivity

Nearby points should be connected so that once the search reaches the correct region, it can efficiently refine the answer.

### Global navigability

The graph should also contain connections that allow the search to travel quickly across the space.

This gives the graph a **small-world structure**:

```text
Local edges
    ↓
Good local search

Long-range edges
    ↓
Fast global navigation

Together
    ↓
Navigable graph
    ↓
Efficient ANN search
```

---

## From NSW to HNSW

This repository is designed to eventually progress from **NSW to HNSW**.

The key limitation of a single-layer NSW graph is that the search must navigate the entire graph at one level.

HNSW introduces a hierarchy:

```text
Layer 2:              ●──────●
                     /        \
                    /          \
Layer 1:       ●───●────●──────●───●
                \       \      /
                 \       \    /
Layer 0:    ●─●─●─●─●─●─●─●─●─●─●─●─●
```

Higher layers contain fewer nodes and longer-range connections.

The search can therefore proceed in two stages:

```text
Coarse navigation
       │
       ▼
Higher layers
       │
       ▼
Move rapidly toward query
       │
       ▼
Lower layers
       │
       ▼
Fine-grained search
       │
       ▼
Nearest neighbors
```

The next step in this project is therefore to understand and implement **HNSW from the NSW implementation developed here**.

---

## What This Project Is About

This is primarily a **learning and implementation project**.

The objective is to understand:

- why exact nearest-neighbor search becomes expensive
- the curse of dimensionality
- KNN graphs
- greedy graph search
- failure modes of greedy search
- graph navigability
- long-range connections
- small-world networks
- NSW construction
- approximate graph search
- multi-start search
- recall and search-efficiency trade-offs
- and eventually HNSW

The implementation is intentionally simple and educational rather than optimized for production use.

---

## References

The implementation is inspired by the literature on navigable small-world graphs and hierarchical graph-based ANN search.

A major reference for the NSW/HNSW line of algorithms is:

> Malkov, Y. A., & Yashunin, D. A. (2018).
> **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.**
> IEEE Transactions on Pattern Analysis and Machine Intelligence.

---

## Experiments

The experiments use synthetic vector datasets so that the behavior of the graph can be visualized directly.

This is deliberate: before applying ANN algorithms to millions of high-dimensional embeddings, it is useful to understand **why the graph works geometrically**.

The notebooks therefore emphasize:

1. visualization,
2. controlled experiments,
3. search trajectories,
4. exact-vs-approximate comparisons,
5. and understanding the algorithmic mechanisms.

---

## Future Work

Planned extensions include:

- [ ] Complete HNSW implementation
- [ ] Hierarchical graph construction
- [ ] `ef` search parameter
- [ ] `M` graph connectivity parameter
- [ ] Better neighbor-selection heuristics
- [ ] Search-path analysis
- [ ] Distance-computation benchmarks
- [ ] Recall vs latency experiments
- [ ] Higher-dimensional datasets
- [ ] Comparison with brute-force KNN
- [ ] Comparison with existing ANN libraries
- [ ] C++ implementation for performance experimentation

---

## Author

**Hrishikesh Tiwari**

This project is part of a broader effort to understand algorithms and systems by implementing them from first principles rather than treating existing libraries as black boxes.
