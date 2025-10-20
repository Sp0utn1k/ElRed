import numpy as np
from .matrices import norm_2d

def latlon_to_unitvec(lat, lon):
    """
    Convert latitude and longitude (in radians) to 3D unit vectors.
    lat, lon can be scalars or 1D arrays of the same shape.
    """
    clat = np.cos(lat)
    return np.vstack((clat * np.cos(lon),
                      clat * np.sin(lon),
                      np.sin(lat))).T

def unitvec_to_latlon(u):
    """
    Convert 3D unit vector (or array of them) to (lat, lon) in radians.
    """
    u = np.asarray(u)
    lat = np.arcsin(u[..., 2])
    lon = np.arctan2(u[..., 1], u[..., 0])
    return lat, lon

def angular_distances(c, U):
    """
    Cosines of angles between center c and all unit vectors U (N×3).
    Returns cosines clipped into [-1,1].
    """
    cosines = U.dot(c)
    return np.clip(cosines, -1.0, 1.0)

def _sph_min_enclosing_cap(U, eps=1e-12, maxiter=2000):
    """
    Find the center c (unit 3‐vector) and radius r (in radians)
    of the minimal enclosing spherical cap of points U (N×3 unit vectors).
    """
    # initial center: normalized average
    c = U.sum(axis=0)
    c /= norm_2d(c)
    # initial worst‐angle
    cosines = angular_distances(c, U)
    idx = np.argmin(cosines)
    loss = np.arccos(cosines[idx])
    step = loss  # initial step‐size ~ current radius

    for _ in range(maxiter):
        cosines = angular_distances(c, U)
        idx = np.argmin(cosines)
        min_cos = cosines[idx]
        loss = np.arccos(min_cos)
        # gradient of arccos(c·u) w.r.t. c is -u / sqrt(1-(c·u)^2)
        u_w = U[idx]
        denom = np.sqrt(max(1e-16, 1.0 - min_cos * min_cos))
        g = -u_w / denom
        # project gradient onto tangent plane at c
        g -= c * (c.dot(g))
        ng = norm_2d(g)
        if ng < 1e-16:
            break
        g /= ng
        # attempt an update: rotate c a small 'step' toward -g
        # on the sphere, that is c_new = c*cos(step) + g*sin(step)
        c_new = c * np.cos(step) + g * np.sin(step)
        c_new /= norm_2d(c_new)
        cos_new = angular_distances(c_new, U)
        loss_new = np.arccos(np.min(cos_new))
        if loss_new < loss:
            c = c_new
            continue
        else:
            step *= 0.5
            if step < eps:
                break

    # final radius
    final_cos = np.min(angular_distances(c, U))
    r = np.arccos(final_cos)
    return c, r

def min_enclosing_cap(latlon):
    """
    Find the minimal enclosing spherical cap for given lat/lon points.
    
    Parameters:
    latlon : array-like, shape (N, 2)
        Array of [latitude, longitude] pairs in degrees
    
    Returns:
    center : numpy array, shape (2,)
        [latitude, longitude] of cap center in degrees
    radius : float
        Radius of cap in degrees
    """
    lats, lons = latlon.T
    
    # Convert input from degrees to radians
    lats_rad = np.deg2rad(lats)
    lons_rad = np.deg2rad(lons)

    U = latlon_to_unitvec(lats_rad, lons_rad)

    c3, radius_rad = _sph_min_enclosing_cap(U)
    center_rad = unitvec_to_latlon(c3)
    
    # Convert output from radians to degrees
    center_deg = np.rad2deg(np.array(center_rad))
    radius_deg = np.rad2deg(radius_rad)
    
    return center_deg, radius_deg
