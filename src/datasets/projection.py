import warnings
import numpy as np
from pyproj import Proj
from scipy.stats import chi2

from ..utils import min_enclosing_cap
from . import LatLonData, XYData

def get_projection(lats, lons):
    latlons = np.column_stack((lats, lons))
    center, _ = min_enclosing_cap(latlons)
    lat0, lon0 = center

    proj = Proj(proj="ortho", 
		lat_0=lat0, 
		lon_0=lon0, 
		datum="WGS84")

    return proj

def convert_latlon_to_xy_angles(latlon_data, proj):
    """Project angles from lat/lon to x/y coordinates."""
    # Correct meridian convergence in bearings
    facts = proj.get_factors(latlon_data.lons, latlon_data.lats)
    meridian_convergence = facts.meridian_convergence # degrees
    bearings = latlon_data.bearings + meridian_convergence

    # Correct bearings from (degrees, clockwise from north) to angles (radians, counter-clockwise from east)
    angles_rad = np.radians(90 - bearings)  # Convert to radians and adjust
    angles_rad = np.mod(angles_rad, 2 * np.pi)  # Ensure angles are in [0, 2*pi)

    return angles_rad

def compute_covariances(latlon_data, proj, alpha=0.95):
    """ Compute covariance matrices for latlon data projected to XY coordinates.
    majors and minors are the semi-major and semi-minor axes of the 95% confidence ellipses.
    Args:
        latlon_data (LatLonData): Input data with lat/lon coordinates and ellipse parameters.
        proj (pyproj.Proj): Projection object for converting lat/lon to XY coordinates and compute meridian convergence.
    Returns:
        tuple: (covariance matrices, semi_major, semi_minor, phi) for each point in XY coordinates.
    """

    k = chi2.ppf(alpha, df=2)

    lam2 = latlon_data.majors**2 / k
    lam1 = latlon_data.minors**2 / k
    phi = convert_latlon_to_xy_angles(latlon_data, proj)

    c = np.cos(phi).ravel()
    s = np.sin(phi).ravel()

    covs = np.empty((len(latlon_data),2,2))
    covs[:,0,0] = c**2 * lam2 + s**2 * lam1
    covs[:,1,1] = s**2 * lam2 + c**2 * lam1
    covs[:,0,1] = c*s * (lam2-lam1)
    covs[:,1,0] = covs[:,0,1]

    # Return the ellipse parameters we already computed
    semi_major = latlon_data.majors
    semi_minor = latlon_data.minors

    return covs, semi_major, semi_minor, phi

def validate_projection(latlon_data, proj, tol=.03):
    """Validate that the projection is suitable for the lat/lon data."""
    lats = latlon_data.lats
    lons = latlon_data.lons
    facts = proj.get_factors(lons, lats)
    meridional_error_pct = np.max(np.abs(1-facts.meridional_scale))
    parallel_error_pct = np.max(np.abs(1-facts.parallel_scale))
    angular_error_pct = np.max(np.abs(facts.angular_distortion/90))

    info_msg = (f"Projection validation:\n"
                f" - Max meridional scale error: {meridional_error_pct:.2%}\n"
                f" - Max parallel scale error: {parallel_error_pct:.2%}\n"
                f" - Max angular distortion: {angular_error_pct:.2%}\n")

    warning_msg = []
    if meridional_error_pct > tol:
        warning_msg.append(f"Meridional error {meridional_error_pct:.2%} exceeds tolerance {tol:.2%}. ")
    if parallel_error_pct > tol:
        warning_msg.append(f"Parallel error {parallel_error_pct:.2%} exceeds tolerance {tol:.2%}. ")
    if angular_error_pct > tol:
        warning_msg.append(f"Angular error {angular_error_pct:.2%} exceeds tolerance {tol:.2%}. ")

    if len(warning_msg) > 0:
        warning_msg = "Projection validation failed: " + '\n - '.join(warning_msg)
        return False, warning_msg
    return True, info_msg

def project_latlon_data(latlon_data: LatLonData, verbose: bool = False) -> XYData:
    """Project LatLonData to XYData using orthographic projection."""
    proj = get_projection(latlon_data.lats, latlon_data.lons)
    is_valid, out_msg = validate_projection(latlon_data, proj)
    if not is_valid:
        # raise warning
        warnings.warn(out_msg)

    if verbose:
        print(out_msg)

    x, y = proj(latlon_data.lons, latlon_data.lats)
    covs, semi_major, semi_minor, phi = compute_covariances(latlon_data, proj)
    
    return XYData(
        pos=np.column_stack((x, y)),
        covs=covs,
        idx=latlon_data.idx,
        proj=proj,
        timestamps=latlon_data.timestamps,
        metadata=latlon_data.metadata,
        semi_major=semi_major,
        semi_minor=semi_minor,
        phi=phi,
        parent=latlon_data.parent or latlon_data
    )

