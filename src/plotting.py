import folium
from matplotlib import pyplot as plt
import numpy as np
from pyproj import Geod
import numba

from .utils import min_enclosing_cap

@numba.njit(cache=True)
def adaptive_theta_sampling(a, b, num_points):
    """Return theta array of shape (len(a), num_points) for each ellipse defined by a, b."""
    n = len(a)
    u = np.linspace(0, 2 * np.pi, num_points * 5)
    phi = np.linspace(0, 1, num_points)
    theta_all = np.empty((n, num_points))
    for i in range(n):
        curv = (a[i] * b[i]) / ((a[i] * np.sin(u))**2 + (b[i] * np.cos(u))**2)**(3/2)
        f = np.cumsum(np.sqrt(curv))
        f = f / f[-1]
        theta_all[i] = np.interp(phi, f, u)
    return theta_all

def create_ellipse_points_vectorized(lat, lon, semi_major, semi_minor, bearing, num_points=100):
    """Create ellipse points for multiple ellipses given center, axes, and bearing arrays"""
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

    geod = Geod(ellps='WGS84')

    # Create arrays of the starting coordinates
    # Reshape and repeat for all points of each ellipse
    lons0 = np.repeat(lon.reshape(-1, 1), num_points, axis=1)  # (n_ellipses, num_points)
    lats0 = np.repeat(lat.reshape(-1, 1), num_points, axis=1)  # (n_ellipses, num_points)

    # Flatten for geod.fwd (which expects 1D arrays)
    lons0_flat = lons0.flatten()
    lats0_flat = lats0.flatten()
    bearing_deg_flat = bearing_deg.flatten()
    dist_m_flat = dist_m.flatten()

    # Compute the destination points in one go
    lons_flat, lats_flat, _ = geod.fwd(lons0_flat, lats0_flat, bearing_deg_flat, dist_m_flat)
    
    # Reshape back to (n_ellipses, num_points, 2)
    lons_result = lons_flat.reshape(n_ellipses, num_points)
    lats_result = lats_flat.reshape(n_ellipses, num_points)
    
    # Stack lat, lon for each ellipse
    return np.stack((lats_result, lons_result), axis=2)  # (n_ellipses, num_points, 2)

def generate_folium_map(data, mode='ellipse'):
    """Plot your data class with latlons (Nx2) and ellipses (Nx3) arrays.
    mode: 'ellipse' (default), 'bbox' to plot bounding boxes, or 'points' to plot only points.
    """
    center, radius = min_enclosing_cap(data.latlons)

    zoom_start = 9 - int(np.log2(radius+1e-6))
    zoom_start = max(0, min(18, zoom_start))

    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom_start,
                   control_scale=True)  # adds a dynamic scale bar in km

    if mode == 'ellipse':
        # Use LatLonData properties for vectorized ellipse points creation
        ellipse_pts_all = create_ellipse_points_vectorized(
            data.lats, data.lons, data.majors, data.minors, data.bearings
        )
        for i in range(len(data.latlons)):
            ellipse_pts = ellipse_pts_all[i]
            folium.Polygon(
                locations=ellipse_pts,
                color='blue',
                weight=1,
                fill=True,
                fillOpacity=0.2,
                interactive=False
            ).add_to(m)
    elif mode == 'bbox':
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
                color='green',
                weight=1,
                fill=True,
                fillOpacity=0.15,
                interactive=False
            ).add_to(m)
    elif mode == 'points':
        # Only plot points, no ellipses or bounding boxes
        pass
    else:
        raise ValueError("mode must be 'ellipse', 'bbox', or 'points'")

    for i in range(len(data.latlons)):
        lat, lon = data.latlons[i]
        label = f'''Cluster: {data.labels[i]} 
        <br>Index: {data.idx[i]}'''

        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color='red',
            fill=True,
            tooltip=label
        ).add_to(m)

    return m

def plot_data(data, 
              show_ellipses=True, 
              figsize=(10, 10),
              ellipses_alpha=0.2, 
              skip_labels=False):
    pos = data.pos/1000
    major = data.semi_major/1000
    minor = data.semi_minor/1000
    phi = data.phi 
    labels = None if skip_labels else data.labels
    point_colors = get_colors(labels) if labels is not None else None
    plt.figure(figsize=figsize)
    if show_ellipses and ellipses_alpha > 0:
        for i in range(len(pos)):
            c = 'blue' if point_colors is None else point_colors[i]
            ellipse = plt.matplotlib.patches.Ellipse(
                xy=pos[i],
                width=2*major[i],
                height=2*minor[i],
                angle=np.degrees(phi[i]),   
                edgecolor=c,
                facecolor='none',
                lw=0.5,
                alpha=ellipses_alpha
            )
            plt.gca().add_patch(ellipse)
    plt.scatter(pos[:, 0], pos[:, 1], 
                s=1, 
                color = 'red' if labels is None else point_colors,
                label='Positions')
    
    if hasattr(data, 'mu') and data.mu is not None:
        cluster_colors = get_colors(np.arange(len(data.mu))) if point_colors is not None else None
        plt.scatter(data.mu[:, 0]/1000, data.mu[:, 1]/1000, 
                    s=50, 
                    color='green' if cluster_colors is None else cluster_colors,
                    edgecolors='black',
                    linewidths=1,
                    marker='o', 
                    label='Cluster Centers')

    plt.axis('equal')
    plt.title(f"Projected data ({len(data)} points)")
    plt.xlabel('X coordinate [km]')
    plt.ylabel('Y coordinate [km]')
    plt.grid()
    
    # return svg plot
    return plt

def get_colors(labels):
    """Generate distinct colors for each unique label."""
    if labels is None:
        return None
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.get_cmap('tab20', n_labels)  # Use a colormap with enough distinct colors
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]
    return colors