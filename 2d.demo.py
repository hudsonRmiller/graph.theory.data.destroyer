import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict, deque

np.random.seed(0)

# =====================================================
# Datasets
# =====================================================

def noisy_circle(n=300, noise=0.08):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(t) + noise*np.random.randn(n)
    y = np.sin(t) + noise*np.random.randn(n)
    return np.column_stack([x, y])

def noisy_sine(n=300, noise=0.15):
    x = np.linspace(0, 4*np.pi, n)
    y = np.sin(x) + noise*np.random.randn(n)
    return np.column_stack([x, y])

def line_with_loop(n=400, noise=0.05):
    t = np.linspace(-2, 2, n//2)
    line = np.column_stack([t, noise*np.random.randn(len(t))])

    u = np.linspace(0, 2*np.pi, n//2, endpoint=False)
    loop = np.column_stack([
        0.7*np.cos(u),
        0.7*np.sin(u)
    ]) + np.array([0.8, 0.0])

    X = np.vstack([line, loop])
    X += noise*np.random.randn(*X.shape)
    return X

import numpy as np

# =====================================================
# Low-noise / control datasets
# =====================================================

def clean_circle(n=300):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(t)
    y = np.sin(t)
    return np.column_stack([x, y])

def clean_sine(n=300):
    x = np.linspace(0, 4*np.pi, n)
    y = np.sin(x)
    return np.column_stack([x, y])


# =====================================================
# Moderate-noise datasets
# =====================================================

def noisy_circle(n=300, noise=0.08):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(t) + noise*np.random.randn(n)
    y = np.sin(t) + noise*np.random.randn(n)
    return np.column_stack([x, y])

def noisy_sine(n=300, noise=0.15):
    x = np.linspace(0, 4*np.pi, n)
    y = np.sin(x) + noise*np.random.randn(n)
    return np.column_stack([x, y])


# =====================================================
# High-volatility / extreme noise
# =====================================================

def very_noisy_circle(n=400, noise=0.25):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(t) + noise*np.random.randn(n)
    y = np.sin(t) + noise*np.random.randn(n)
    return np.column_stack([x, y])

def noisy_random_walk(n=400):
    steps = np.random.randn(n, 2)
    X = np.cumsum(steps, axis=0)
    X -= X.mean(axis=0)
    X /= np.std(X)
    return X


# =====================================================
# Loop + line hybrid
# =====================================================

def line_with_loop(n=400, noise=0.05):
    t = np.linspace(-2, 2, n // 2)
    line = np.column_stack([t, noise*np.random.randn(len(t))])

    u = np.linspace(0, 2*np.pi, n // 2, endpoint=False)
    loop = np.column_stack([
        0.7 * np.cos(u),
        0.7 * np.sin(u)
    ]) + np.array([0.8, 0.0])

    X = np.vstack([line, loop])
    X += noise * np.random.randn(*X.shape)
    return X


# =====================================================
# Cluster-based datasets
# =====================================================

def gaussian_clusters():
    centers = [(-2, 0), (0, 0), (2, 0)]
    X = []
    for c in centers:
        X.append(np.random.randn(150, 2) * 0.2 + np.array(c))
    return np.vstack(X)

def clusters_with_bridge():
    left = np.random.randn(120, 2) * 0.2 + np.array([-2, 0])
    right = np.random.randn(120, 2) * 0.2 + np.array([2, 0])
    bridge_x = np.linspace(-2, 2, 120)
    bridge = np.column_stack([bridge_x, 0.1 * np.random.randn(120)])
    return np.vstack([left, bridge, right])


# =====================================================
# Branching / graph-like geometry
# =====================================================

def y_shape(n=450, noise=0.05):
    t = np.linspace(0, 1, n // 3)
    a = np.column_stack([t, t])
    b = np.column_stack([t, -t])
    c = np.column_stack([t, np.zeros_like(t)])
    X = np.vstack([a, b, c])
    return X + noise * np.random.randn(*X.shape)


# =====================================================
# Almost-loops / topological edge cases
# =====================================================

def almost_circle_gap(n=400, noise=0.05):
    t = np.linspace(0, 1.8 * np.pi, n)  # intentional gap
    x = np.cos(t) + noise * np.random.randn(n)
    y = np.sin(t) + noise * np.random.randn(n)
    return np.column_stack([x, y])


# =====================================================
# Parallel manifolds (hard case)
# =====================================================

def parallel_sines(n=400, noise=0.05):
    x = np.linspace(0, 4*np.pi, n // 2)
    y1 = np.sin(x)
    y2 = np.sin(x) + 1.0

    X = np.vstack([
        np.column_stack([x, y1]),
        np.column_stack([x, y2])
    ])

    return X + noise * np.random.randn(*X.shape)

def sine_cosine (n=400, noise=0.05):
    x = np.linspace(0, 4 * np.pi, n // 2)
    y1 = np.sin(x)
    y2 = np.cos(x) + 1.0

    X = np.vstack([
            np.column_stack([x, y1]),
            np.column_stack([x, y2])
    ])

    return X + noise * np.random.randn(*X.shape)

# =====================================================
# Crossing manifolds (expected failure mode)
# =====================================================

def cross(n=400, noise=0.05):
    t = np.linspace(-1, 1, n // 2)
    a = np.column_stack([t, np.zeros_like(t)])
    b = np.column_stack([np.zeros_like(t), t])
    X = np.vstack([a, b])
    return X + noise * np.random.randn(*X.shape)

# =====================================================
# Nearest-neighbor graph
# =====================================================

def nearest_neighbor_graph(X):
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)

    adj = defaultdict(set)
    for i, j in enumerate(nn):
        adj[i].add(j)
        adj[j].add(i)

    return adj

# =====================================================
# Connected components
# =====================================================

def connected_components(adj, n):
    visited = np.zeros(n, dtype=bool)
    components = []

    for i in range(n):
        if visited[i]:
            continue

        q = deque([i])
        visited[i] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        components.append(comp)

    return components

# =====================================================
# One geometric coarsening step
# =====================================================

def geometric_coarsening(X):
    adj = nearest_neighbor_graph(X)
    comps = connected_components(adj, len(X))
    centers = np.array([X[c].mean(axis=0) for c in comps])
    return adj, centers

# =====================================================
# Iterated process
# =====================================================

def iterated_geometric_centers(X, n_iters):
    levels = []
    edges = []

    current = X
    for _ in range(n_iters):
        adj, centers = geometric_coarsening(current)
        levels.append(current)
        edges.append(adj)
        current = centers

    levels.append(current)  # final centers
    return levels, edges

# =====================================================
# Plotting
# =====================================================

def plot_dataset_iterations(X, name, n_iters):
    levels, edges = iterated_geometric_centers(X, n_iters)

    fig, axes = plt.subplots(1, n_iters + 1, figsize=(4*(n_iters+1), 4))
    fig.suptitle(name, fontsize=14)

    for i in range(n_iters + 1):
        ax = axes[i]

        if i < n_iters:
            Xi = levels[i]
            adj = edges[i]

            ax.scatter(Xi[:,0], Xi[:,1], s=15, color="black", alpha=0.5)

            for u in adj:
                for v in adj[u]:
                    ax.plot(
                        [Xi[u,0], Xi[v,0]],
                        [Xi[u,1], Xi[v,1]],
                        color="gray",
                        lw=0.6,
                        alpha=0.4
                    )

            ax.set_title(f"Iteration {i}")

        else:
            # Final: original + final centers
            ax.scatter(
                levels[0][:,0],
                levels[0][:,1],
                s=10,
                color="black",
                alpha=0.3,
                label="Original"
            )
            ax.scatter(
                levels[-1][:,0],
                levels[-1][:,1],
                s=120,
                color="red",
                edgecolor="white",
                zorder=5,
                label="Final centers"
            )
            ax.set_title("Final")

        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    n_iters = 2  # user-controlled

    datasets = {
        # Original
        "Noisy Circle": noisy_circle(),
        "Noisy Sine": noisy_sine(),
        "Line with Loop": line_with_loop(),

        # Low noise
        "Clean Circle": clean_circle(),
        "Clean Sine": clean_sine(),

        # High volatility
        "Very Noisy Circle": very_noisy_circle(),
        "Random Walk": noisy_random_walk(),
        # Clusters
        "Gaussian Clusters": gaussian_clusters(),
        "Clusters with Bridge": clusters_with_bridge(),

        # Geometry stress tests
        "Y-shape": y_shape(),
        "Almost Circle (Gap)": almost_circle_gap(),
        "Parallel Sines": parallel_sines(),

        # Known hard case
        "Cross": cross(),
        "sine-cosine": sine_cosine(),
    }

    for name, X in datasets.items():
        plot_dataset_iterations(X, name, n_iters)
