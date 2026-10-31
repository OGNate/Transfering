"""
Plot contour lines where the local hit rate reaches a chosen level (e.g. 70%).

Idea: you can't draw a "70% hit" line through raw 0/1 points directly, so we first
estimate a smooth hit-probability surface P(hit | x, y) over a grid, then ask
Plotly to draw the contour where that surface equals your threshold.

Two estimators are included:
  - "kernel":   local, Gaussian-weighted hit rate (no model assumptions)
  - "logistic": logistic regression with polynomial features (smoother, parametric)
"""

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# 1. Estimate P(hit) on a grid
# ---------------------------------------------------------------------------
def kernel_hit_rate(x, y, hit, gx, gy, bandwidth=None):
    """
    Nadaraya-Watson estimate: at each grid point, the hit rate of nearby points,
    weighted by a Gaussian on distance. Also returns the effective number of
    points behind each estimate so sparse areas can be masked out.
    """
    pts = np.column_stack([x, y]).astype(float)
    # Scale axes so one bandwidth works even when x and y have different units
    scale = pts.std(axis=0)
    scale[scale == 0] = 1.0
    pts = pts / scale

    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()]) / scale

    if bandwidth is None:
        # Scaled-down Scott's rule. Larger = smoother but pulls contours inward
        # toward the bulk of the data; smaller = more faithful but noisier.
        bandwidth = 0.6 * len(pts) ** (-1 / 6)

    # Squared distances, grid x points (chunked to keep memory reasonable)
    rate = np.empty(len(grid))
    n_eff = np.empty(len(grid))
    hit = np.asarray(hit, dtype=float)
    chunk = 5000
    for i in range(0, len(grid), chunk):
        d2 = ((grid[i:i + chunk, None, :] - pts[None, :, :]) ** 2).sum(-1)
        w = np.exp(-0.5 * d2 / bandwidth**2)
        wsum = w.sum(1)
        rate[i:i + chunk] = (w @ hit) / np.maximum(wsum, 1e-12)
        n_eff[i:i + chunk] = wsum**2 / np.maximum((w**2).sum(1), 1e-12)

    return rate.reshape(GX.shape), n_eff.reshape(GX.shape)


def logistic_hit_rate(x, y, hit, gx, gy, degree=3):
    """Logistic regression on polynomial features of (x, y)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X = np.column_stack([x, y])
    model = make_pipeline(
        StandardScaler(), PolynomialFeatures(degree), StandardScaler(),
        LogisticRegression(max_iter=5000),
    )
    model.fit(X, hit)
    GX, GY = np.meshgrid(gx, gy)
    p = model.predict_proba(np.column_stack([GX.ravel(), GY.ravel()]))[:, 1]
    return p.reshape(GX.shape)


# ---------------------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------------------
def plot_hit_contours(x, y, hit, levels=(0.7,), method="kernel",
                      bandwidth=None, min_points=5, grid_size=150,
                      show_surface=True):
    """
    x, y       : coordinates of each data point
    hit        : 1/True for hit, 0/False for no hit
    levels     : hit-rate threshold(s) to draw, e.g. (0.7,) or (0.5, 0.7, 0.9)
    method     : "kernel" or "logistic"
    min_points : (kernel only) hide areas backed by fewer effective points
    """
    x, y, hit = np.asarray(x), np.asarray(y), np.asarray(hit).astype(int)
    pad_x = 0.05 * np.ptp(x)
    pad_y = 0.05 * np.ptp(y)
    gx = np.linspace(x.min() - pad_x, x.max() + pad_x, grid_size)
    gy = np.linspace(y.min() - pad_y, y.max() + pad_y, grid_size)

    if method == "kernel":
        Z, n_eff = kernel_hit_rate(x, y, hit, gx, gy, bandwidth)
        Z = np.where(n_eff >= min_points, Z, np.nan)  # don't extrapolate into empty space
    elif method == "logistic":
        Z = logistic_hit_rate(x, y, hit, gx, gy)
    else:
        raise ValueError("method must be 'kernel' or 'logistic'")

    fig = go.Figure()

    # Optional: faint background of the full estimated hit-rate surface
    if show_surface:
        fig.add_trace(go.Heatmap(
            x=gx, y=gy, z=Z, zmin=0, zmax=1, colorscale="RdBu",
            opacity=0.35, colorbar=dict(title="Est. hit rate"),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>hit rate=%{z:.0%}<extra></extra>",
        ))

    # One contour line per requested level
    dash_styles = ["solid", "dash", "dot", "dashdot"]
    for i, lvl in enumerate(sorted(levels)):
        fig.add_trace(go.Contour(
            x=gx, y=gy, z=Z,
            contours=dict(start=lvl, end=lvl, size=1, coloring="none",
                          showlabels=True,
                          labelfont=dict(size=12, color="black")),
            line=dict(width=3, color="black", dash=dash_styles[i % 4]),
            showscale=False, name=f"{lvl:.0%} hit rate",
            showlegend=True, hoverinfo="skip",
        ))

    # Shade the region where the hit rate is at or above the lowest requested level
    lvl = min(levels)
    fig.add_trace(go.Contour(
        x=gx, y=gy, z=Z,
        contours=dict(type="constraint", operation=">=", value=lvl),
        fillcolor="rgba(34,139,34,0.18)", line=dict(width=0),
        name=f"≥ {lvl:.0%} region", showlegend=True, hoverinfo="skip",
    ))

    # Raw data points on top
    fig.add_trace(go.Scatter(
        x=x[hit == 1], y=y[hit == 1], mode="markers", name="Hit",
        marker=dict(color="#1f77b4", size=6, line=dict(width=0.5, color="white")),
    ))
    fig.add_trace(go.Scatter(
        x=x[hit == 0], y=y[hit == 0], mode="markers", name="No hit",
        marker=dict(color="#d62728", size=6, symbol="x"),
    ))

    fig.update_layout(
        title=f"Hit-rate contours ({method} estimate)",
        xaxis_title="x", yaxis_title="y", template="plotly_white",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Demo with synthetic data — replace with your own x, y, hit arrays
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-3, 3, n)
    y = rng.uniform(-3, 3, n)
    # True hit probability: high near the center, falling off with distance
    p_true = 1 / (1 + np.exp(2.0 * (np.hypot(x, y) - 1.8)))
    hit = rng.random(n) < p_true

    fig = plot_hit_contours(x, y, hit, levels=(0.5, 0.7, 0.9), method="kernel")
    fig.write_html("hit_rate_contours.html")
    fig.show()
