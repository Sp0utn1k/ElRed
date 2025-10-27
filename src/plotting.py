"""
This module provides various plotting utilities for visualizing geospatial data, including ellipses, bounding boxes, and clustered points.
"""

from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection
import numpy as np
from pyproj import Geod
import numba
from offline_folium import offline  # pylint: disable=unused-import
import folium
import plotly
import plotly.graph_objs as go

from .utils import min_enclosing_cap

CUSTOM_TILES_ADDRESSES = {
    "osm": "http://[::1]:8080/styles/osm-bright/512/{z}/{x}/{y}.png"
}


@numba.njit()
def adaptive_theta_sampling(a, b, num_points):
    """Return theta array of shape (len(a), num_points) for each ellipse defined by a, b.

    Args:
        a (array-like): Semi-major axes of the ellipses.
        b (array-like): Semi-minor axes of the ellipses.
        num_points (int): Number of points to sample along each ellipse.

    Returns:
        numpy.ndarray: Array of theta values for each ellipse.
    """
    n = len(a)
    u = np.linspace(0, 2 * np.pi, num_points * 5)
    phi = np.linspace(0, 1, num_points)
    theta_all = np.empty((n, num_points))
    for i in range(n):
        curv = (a[i] * b[i]) / ((a[i] * np.sin(u)) ** 2 + (b[i] * np.cos(u)) ** 2) ** (
            3 / 2
        )
        f = np.cumsum(np.sqrt(curv))
        f = f / f[-1]
        theta_all[i] = np.interp(phi, f, u)
    return theta_all


def create_ellipse_points_vectorized(
    lat, lon, semi_major, semi_minor, bearing, num_points=100
):
    """Create ellipse points for multiple ellipses given center, axes, and bearing arrays.

    Args:
        lat (array-like): Latitudes of ellipse centers.
        lon (array-like): Longitudes of ellipse centers.
        semi_major (array-like): Semi-major axes of the ellipses (meters).
        semi_minor (array-like): Semi-minor axes of the ellipses (meters).
        bearing (array-like): Bearings of the ellipses (degrees clockwise from north).
        num_points (int, optional): Number of points to sample along each ellipse. Defaults to 100.

    Returns:
        numpy.ndarray: Array of shape (n_ellipses, num_points, 2) containing lat/lon points for each ellipse.
    """
    # Convert inputs to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    semi_major = np.asarray(semi_major)
    semi_minor = np.asarray(semi_minor)
    bearing = np.asarray(bearing)

    n_ellipses = len(lat)
    bearing_rad = np.radians(-bearing)

    theta = adaptive_theta_sampling(semi_major, semi_minor, num_points)

    # Reshape for broadcasting: (n_ellipses, num_points)
    y = semi_major.reshape(-1, 1) * np.cos(theta)
    x = semi_minor.reshape(-1, 1) * np.sin(theta)

    # Rotate x,y by bearing_rad (clockwise rotation)
    # Reshape bearing for broadcasting: (n_ellipses, 1)
    bearing_rad = bearing_rad.reshape(-1, 1)
    cos_bearing = np.cos(bearing_rad)
    sin_bearing = np.sin(bearing_rad)

    x_rot = x * cos_bearing - y * sin_bearing
    y_rot = x * sin_bearing + y * cos_bearing

    # Compute distance and bearing (in degrees) vectorized
    dist_m = np.sqrt(x_rot**2 + y_rot**2)
    bearing_deg = np.degrees(np.arctan2(x_rot, y_rot))

    geod = Geod(ellps="WGS84")

    # Create arrays of the starting coordinates
    # Reshape and repeat for all points of each ellipse
    lons0 = np.repeat(
        lon.reshape(-1, 1), num_points, axis=1
    )  # (n_ellipses, num_points)
    lats0 = np.repeat(
        lat.reshape(-1, 1), num_points, axis=1
    )  # (n_ellipses, num_points)

    # Flatten for geod.fwd (which expects 1D arrays)
    lons0_flat = lons0.flatten()
    lats0_flat = lats0.flatten()
    bearing_deg_flat = bearing_deg.flatten()
    dist_m_flat = dist_m.flatten()

    # Compute the destination points in one go
    lons_flat, lats_flat, _ = geod.fwd(
        lons0_flat, lats0_flat, bearing_deg_flat, dist_m_flat
    )

    # Reshape back to (n_ellipses, num_points, 2)
    lons_result = lons_flat.reshape(n_ellipses, num_points)
    lats_result = lats_flat.reshape(n_ellipses, num_points)

    # Stack lat, lon for each ellipse
    return np.stack((lats_result, lons_result), axis=2)  # (n_ellipses, num_points, 2)


def generate_folium_map(
    data,
    mode="ellipse",
    tiles="osm",
    ellipse_num_points=100,
    ellipse_alpha=0.1,
):
    """Generate an interactive folium map to visualize geospatial data.

    Args:
        data: Data object containing lat/lon, ellipses, and labels.
        mode (str, optional): Visualization mode ('ellipse', 'bbox', or 'points').
            Defaults to 'ellipse'.
        tiles (str, optional): Tiles style. Defaults to 'osm' (offline tileserver).
        ellipse_num_points (int, optional): Number of points for ellipses polygon. Defaults to 100.
        ellipse_alpha (float, optional): Transparency level for ellipses. Defaults to 0.1.

    Returns:
        folium.Map: Folium map object.
    """
    center, radius = min_enclosing_cap(data.latlons)

    zoom_start = 9 - int(np.log2(radius + 1e-6))
    zoom_start = max(0, min(18, zoom_start))

    tiles_url = None
    for url in CUSTOM_TILES_ADDRESSES.keys():
        if tiles == url:
            tiles_url = url
            tiles = None

    m = folium.Map(
        location=[center[0], center[1]],
        tiles=tiles,
        zoom_start=zoom_start,
        control_scale=True,
    )  # adds a dynamic scale bar in km

    if tiles_url is not None:
        folium.TileLayer(
            tiles=tiles_url,
            attr="Local Tiles Server",
            name="Local Tiles",
            overlay=False,
            control=True,
        ).add_to(m)

    if mode == "ellipse":
        # Use LatLonData properties for vectorized ellipse points creation
        ellipse_pts_all = create_ellipse_points_vectorized(
            data.lats,
            data.lons,
            data.majors,
            data.minors,
            data.bearings,
            num_points=ellipse_num_points,
        )
        for i in range(len(data.latlons)):
            ellipse_pts = ellipse_pts_all[i]
            folium.Polygon(
                locations=ellipse_pts,
                color="blue",
                weight=1,
                opacity=ellipse_alpha,
                fill=True,
                fillOpacity=0,
                interactive=False,
            ).add_to(m)
    elif mode == "bbox":
        for i in range(len(data.latlons)):
            bbox_min = data.bbox_min[i]
            bbox_max = data.bbox_max[i]
            # Rectangle corners: SW, NW, NE, SE, SW
            sw = [bbox_min[0], bbox_min[1]]
            nw = [bbox_max[0], bbox_min[1]]
            ne = [bbox_max[0], bbox_max[1]]
            se = [bbox_min[0], bbox_max[1]]
            bbox_pts = [sw, nw, ne, se, sw]
            folium.Polygon(
                locations=bbox_pts,
                color="green",
                weight=1,
                fill=True,
                fillOpacity=0.15,
                interactive=False,
            ).add_to(m)
    elif mode == "points":
        # Only plot points, no ellipses or bounding boxes
        pass
    else:
        raise ValueError("mode must be 'ellipse', 'bbox', or 'points'")

    for i, (lat, lon) in enumerate(data.latlons):
        if data.labels is None:
            label = f"Index: {data.idx[i]}"
        else:
            label = f"""Cluster: {data.labels[i]}
        <br>Index: {data.idx[i]}"""

        folium.CircleMarker(
            location=[lat, lon], radius=2, color="red", fill=True, tooltip=label
        ).add_to(m)
    return m


def plot_data(
    data, show_ellipses=True, figsize=(10, 10), ellipses_alpha=0.2, skip_labels=False
):
    """Plot geospatial data using matplotlib.

    Args:
        data: Data object containing positions, ellipses, and labels.
        show_ellipses (bool, optional): Whether to display ellipses. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        ellipses_alpha (float, optional): Transparency level for ellipses. Defaults to 0.2.
        skip_labels (bool, optional): Whether to skip plotting labels. Defaults to False.

    Returns:
        tuple: Matplotlib figure and axis objects.
    """
    pos = data.pos
    major = data.semi_major
    minor = data.semi_minor
    phi = data.phi
    labels = None if skip_labels else data.labels
    point_colors = get_colors(labels) if labels is not None else None

    fig, ax = plt.subplots(figsize=figsize)

    if show_ellipses and ellipses_alpha > 0:
        # Prepare EllipseCollection parameters
        widths = 2 * major
        heights = 2 * minor
        angles = np.degrees(phi)
        offsets = pos

        if point_colors is not None:
            # EllipseCollection expects RGBA tuples for edgecolors
            edgecolors = point_colors
        else:
            edgecolors = ["blue"] * len(pos)

        ec = EllipseCollection(
            widths,
            heights,
            angles,
            units="xy",
            offsets=offsets,
            transOffset=ax.transData,
            edgecolors=edgecolors,
            facecolors="none",
            linewidths=0.5,
            alpha=ellipses_alpha,
        )
        ax.add_collection(ec)

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=1,
        color="red" if labels is None else point_colors,
        label="Positions",
    )

    if hasattr(data, "mu") and data.mu is not None:
        cluster_colors = (
            get_colors(np.arange(len(data.mu))) if point_colors is not None else None
        )
        ax.scatter(
            data.mu[:, 0],
            data.mu[:, 1],
            s=50,
            color="green" if cluster_colors is None else cluster_colors,
            edgecolors="black",
            linewidths=1,
            marker="o",
            label="Cluster Centers",
        )

    ax.axis("equal")
    ax.set_title(
        f"""Projected data ({len(data)} points,
        {len(np.unique(data.labels)) if data.labels is not None else 0} labels)"""
    )
    ax.grid()

    return fig, ax


def plot_with_plotly(dataset):
    """Visualize clustered data using Plotly.

    Args:
        dataset: Dataset object containing x, y coordinates and labels.

    Returns:
        plotly.graph_objs.Figure: Plotly figure object.
    """
    plotly.offline.init_notebook_mode()

    # Main scatter: points colored by label
    scatter_points = go.Scattergl(
        x=dataset.x,
        y=dataset.y,
        mode="markers",
        marker=dict(color=dataset.labels, colorscale="Viridis", size=3, line_width=0),
    )
    scatter_centers = go.Scattergl()  # empty

    if hasattr(dataset, "mu") and dataset.mu is not None:
        scatter_centers = go.Scattergl(
            x=dataset.mu[:, 0],
            y=dataset.mu[:, 1],
            mode="markers",
            marker=dict(
                color=np.arange(len(dataset.mu)),
                colorscale="Viridis",
                size=14,
                line=dict(width=2, color="black"),
            ),
        )

    fig = go.Figure(data=[scatter_points, scatter_centers])
    fig.update_layout(
        dragmode="pan",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="Clustered points and centers (equal axis scale)",
        xaxis_title="X [meters]",
        yaxis_title="Y [meters]",
        width=900,
        height=900,
        showlegend=False,
    )
    return fig


def get_colors(labels):
    """Generate distinct colors for each unique label.

    Args:
        labels (array-like): Array of labels.

    Returns:
        list: List of colors corresponding to each label.
    """
    if labels is None:
        return None
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.get_cmap("tab20", n_labels)  # Use a colormap with enough distinct colors
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    return colors
