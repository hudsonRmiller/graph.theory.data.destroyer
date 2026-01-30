import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict, deque
from mpl_toolkits.mplot3d import Axes3D  # noqa

np.random.seed(0)

# =====================================================
# 3D DATASETS
# =====================================================

def noisy_helix(n=600, noise=0.08):
    t = np.linspace(0, 6*np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2*np.pi)
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def noisy_swiss_roll(n=800, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n)
    z = t * np.sin(t)
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def line_with_loop_3d(n=800, noise=0.06):
    # Line
    t = np.linspace(-2, 2, n//2)
    line = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])

    # Loop
    u = np.linspace(0, 2*np.pi, n//2, endpoint=False)
    loop = np.column_stack([
        0.7*np.cos(u),
        0.7*np.sin(u),
        0.7*np.sin(2*u)
    ]) + np.array([1.2, 0.0, 0.0])

    X = np.vstack([line, loop])
    X += noise * np.random.randn(*X.shape)
    return X

def clustered_planes(n=900, noise=0.05):
    centers = [
        np.array([0, 0, 0]),
        np.array([2, 2, 0]),
        np.array([-2, 2, 1])
    ]
    X = []
    for c in centers:
        pts = np.random.randn(n//3, 3)
        pts[:, 2] *= 0.2  # flatten
        X.append(c + pts)
    X = np.vstack(X)
    X += noise * np.random.randn(*X.shape)
    return X

def noisy_plane(n=800, noise=0.05):
    x = np.random.uniform(-2, 2, n)
    y = np.random.uniform(-2, 2, n)
    z = 0.3 * x + 0.2 * y
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def noisy_plane(n=800, noise=0.05):
    x = np.random.uniform(-2, 2, n)
    y = np.random.uniform(-2, 2, n)
    z = 0.3 * x + 0.2 * y
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def sinusoidal_surface(n=1000, noise=0.05):
    x = np.random.uniform(-3, 3, n)
    y = np.random.uniform(-3, 3, n)
    z = np.sin(x) * np.cos(y)
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def noisy_sphere(n=900, noise=0.06):
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2*np.pi, n)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def thick_swiss_roll(n=1000, noise=0.08):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    h = np.random.uniform(-1, 1, n)

    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)

    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X

def intersecting_planes(n=900, noise=0.05):
    x1 = np.random.uniform(-2, 2, n//2)
    y1 = np.random.uniform(-2, 2, n//2)
    z1 = np.zeros_like(x1)

    x2 = np.random.uniform(-2, 2, n//2)
    y2 = np.zeros_like(x2)
    z2 = np.random.uniform(-2, 2, n//2)

    X = np.vstack([
        np.column_stack([x1, y1, z1]),
        np.column_stack([x2, y2, z2])
    ])

    X += noise * np.random.randn(*X.shape)
    return X

def noisy_saddle(n=900, noise=0.05):
    x = np.random.uniform(-2, 2, n)
    y = np.random.uniform(-2, 2, n)
    z = x**2 - y**2
    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(*X.shape)
    return X


# =====================================================
# NEAREST-NEIGHBOR GRAPH
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
# CONNECTED COMPONENTS
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
# ONE GEOMETRIC COARSENING STEP
# =====================================================

def geometric_coarsening(X):
    adj = nearest_neighbor_graph(X)
    comps = connected_components(adj, len(X))
    centers = np.array([X[c].mean(axis=0) for c in comps])
    return adj, centers

# =====================================================
# ITERATED PROCESS
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

    levels.append(current)
    return levels, edges

# =====================================================
# 3D PLOTTING
# =====================================================

def plot_iteration_3d(X, adj=None, title="", show_edges=True, highlight=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:,0], X[:,1], X[:,2],
               s=12, color="black", alpha=0.5)

    if show_edges and adj is not None:
        for u in adj:
            for v in adj[u]:
                ax.plot(
                    [X[u,0], X[v,0]],
                    [X[u,1], X[v,1]],
                    [X[u,2], X[v,2]],
                    color="gray",
                    lw=0.6,
                    alpha=0.4
                )

    if highlight is not None:
        ax.scatter(
            highlight[:,0], highlight[:,1], highlight[:,2],
            s=120, color="red", edgecolor="white", zorder=5
        )

    ax.set_title(title)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()

# =====================================================
# RUN PIPELINE
# =====================================================

if __name__ == "__main__":
    n_iters = 2  # user-controlled

    datasets = {
        "Noisy Helix": noisy_helix(),
        "Swiss Roll": noisy_swiss_roll(),
        "3D Line with Loop": line_with_loop_3d(),
        "Clustered Planes": clustered_planes(),

        # NEW SURFACES
        "Noisy Plane": noisy_plane(),
        "Saddle Surface": noisy_saddle(),
        "Sinusoidal Surface": sinusoidal_surface(),
        "Noisy Sphere": noisy_sphere(),
        "Thick Swiss Roll": thick_swiss_roll(),
        "Intersecting Planes": intersecting_planes(),
    }

    for name, X in datasets.items():
        print(f"Running: {name}")
        levels, edges = iterated_geometric_centers(X, n_iters)

        # Iteration plots
        for i in range(n_iters):
            plot_iteration_3d(
                levels[i],
                adj=edges[i],
                title=f"{name} – Iteration {i}",
                show_edges=True
            )

        # Final: original + final centers
        plot_iteration_3d(
            levels[0],
            title=f"{name} – Final Centers",
            show_edges=False,
            highlight=levels[-1]
        )
