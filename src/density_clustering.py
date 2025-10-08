from collections import defaultdict
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import List, Dict
from scipy.ndimage import gaussian_filter
from scipy.stats import chi2
from skimage.feature import peak_local_max
import smallestenclosingcircle

from .datasets import XYData

def density_clustering(data,
                        cell_size_multiplier=0.5,
                        sigma_in_meters=1000,
                        sigma_min=2,
                        sigma_max=10,
                        threshold_count_per_km2=1000,
                        blob_min_count_per_km2=100,
                        min_dist_between_peaks=800,
                        blob_lower_tol=0.05,
                        blob_patience_in_meters=500,
                        blob_patience_tol=0.05,
                        blob_merge_offset=0,
                        outliers_subgroup_tol=0.1,
                        outliers_alpha=0.9999,
                        verbose=False):
    
    counts = data.get_grid_count(k=cell_size_multiplier)

    if verbose:
        print(f"Grid shape: {counts.shape}, max count: {counts.max()}")
        print(f"Grid cell size: {data.g} meters")

    sigma = sigma_in_meters / data.g
    sigma_clipped = np.clip(sigma, sigma_min, sigma_max)
    if sigma != sigma_clipped:
        print(f"Warning: sigma {sigma:.2f} out of bounds, clipped to {sigma_clipped:.2f}")


    blurred = gaussian_filter(counts, sigma=sigma_clipped)

    threshold_count = threshold_count_per_km2 * (data.g/1000)**2
    peak_threshold = threshold_count / (2 * np.pi * sigma_clipped**2)
    min_cells_between_peaks = int(min_dist_between_peaks / data.g)
    if min_cells_between_peaks < 1:
        print(f"Warning: min_distance {min_cells_between_peaks} less than 1 cell, setting to 1")
        min_cells_between_peaks = 1

    if verbose:
        print(f"Gaussian sigma: {sigma_clipped:.2f} cells")
        print(f"Count threshold: {threshold_count:.1f}")
        print(f"Min distance between peaks: {min_cells_between_peaks} cells ({(min_cells_between_peaks * data.g):.1f} meters)")

    # if min_distance < sigma_clipped:
    #     print(f"Warning: min_distance {min_distance} less than sigma {sigma_clipped}, setting to sigma")
    #     min_distance = int(sigma_clipped)


    peaks = peak_local_max(blurred,
                           threshold_abs=peak_threshold,
                           min_distance=min_cells_between_peaks)
    
    # Remove peaks where counts is below threshold_count
    threshold_count = int(np.round(threshold_count))
    peaks = [peak for peak in peaks if max_neighbor_count(counts, peak, window=min_cells_between_peaks) >= threshold_count]

    peaks = np.array(sorted(peaks, key= lambda x: blurred[tuple(x)], reverse=True))
    

    blob_threshold = blob_min_count_per_km2 * (data.g/1000)**2
    blob_patience = np.ceil(blob_patience_in_meters / data.g)
    blobs = find_blobs(blurred, 
                       blob_threshold, 
                       peaks, 
                       tol=blob_lower_tol, 
                       patience=int(blob_patience), 
                       patience_tol=blob_patience_tol)
    
    blobs = fill_holes(blobs)


    overlapping_blobs = get_neighbours_idx(blobs, offset=blob_merge_offset)
    blobs = merge_neighbours(blobs, 
                             overlapping_blobs)

    subsets, outliers = process_outliers(data, blobs,
                            tol=outliers_subgroup_tol,
                            max_dist_alpha=outliers_alpha)
    
    peaks_m = (peaks * data.g + np.array([data.x_min, data.y_min]))
    if verbose:
        print(f"Found {len(blobs)} blobs, {len(subsets)} subsets, {len(outliers)} outliers")

    return subsets, outliers, peaks_m

def max_neighbor_count(counts, pos, window=1):
    i, j = pos
    Ni, Nj = counts.shape
    i_min = max(0, i - window)
    i_max = min(Ni, i + window + 1)
    j_min = max(0, j - window)
    j_max = min(Nj, j + window + 1)
    return counts[i_min:i_max, j_min:j_max].max()

def process_outliers(data, blobs,
                     tol=0.05,
                     max_dist_alpha=.9999):
    blob_edges = get_blob_edges(blobs)
    blob_centers, radii = compute_blobs_min_circle(blob_edges)
    subsets, outliers = data.split_blobs(blobs)

    max_dist = np.percentile(outliers.semi_major, 99)
    blobs_centers_m = (blob_centers * data.g + np.array([data.x_min, data.y_min]))
    radii_m = radii * data.g

    dists = np.linalg.norm(blobs_centers_m[:,None,:] - outliers.pos[None,:,:], axis=2)
    radii_expanded = radii_m[:,None] + max_dist
    mask = dists < radii_expanded

    inv_covs = outliers.inv_covs

    dists = np.ones_like(mask, dtype=np.float64) * np.inf

    for i in range(len(blobs)):
        maski = mask[i]
        pos = outliers.pos[maski]
        if len(pos) == 0:
            continue
        blob = blob_edges[i]
        blob = (blob * data.g + np.array([data.x_min, data.y_min]))

        dx = blob[:,None] - pos[None,:]
        mahal = np.einsum('nki,kij,nkj->nk', 
                    dx, 
                    inv_covs[maski], 
                    dx)

        mahal_mins = np.min(mahal, axis=0)
        dists[i, maski] = mahal_mins

    dists_mins = dists.min(axis=0, keepdims=True)
    mask = dists <= np.minimum(dists_mins * (1+tol)**2, chi2.ppf(max_dist_alpha, df=2))

    for i,subset in enumerate(subsets):
        subset = XYData.merge([subset, outliers[mask[i]]])
        subset.reset_labels()
        subsets[i] = subset
    outliers = outliers[~mask.any(axis=0)]

    return subsets, outliers

def get_neighbours_idx(blobs, offset=0):
    blob_edges = get_blob_edges(blobs)
    blob_centers, radii = compute_blobs_min_circle(blob_edges)
    mask = compute_mask_close_blobs(blob_centers, radii, offset=offset)
    return get_neighboring_blobs(blob_edges, mask)

@njit(cache=True)
def get_neighboring_blobs(blobs, overlap_matrix):
    neighboring_blobs = []
    for idx1 in range(len(blobs)):
        for idx2 in range(idx1+1, len(blobs)):
            blob1 = blobs[idx1]
            blob2 = blobs[idx2]
            found = False
            for i1, j1 in blob1:
                if found or not overlap_matrix[idx1, idx2]:
                    break
                for i2, j2 in blob2:
                    if found:
                        break
                    di = abs(i1 - i2)
                    dj = abs(j1 - j2)
                    if di <= 1 and dj <= 1:
                        neighboring_blobs.append((idx1, idx2))
                        found = True
    return neighboring_blobs

def merge_neighbours(blobs, pairs):
    if len(pairs) == 0:
        return blobs
    
    n = len(blobs)
    parents = list(range(n))
    def find(i):
        if parents[i] == i:
            return i
        parents[i] = find(parents[i])
        return parents[i]

    def union(i,j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parents[root_j] = root_i
            
    for i,j in pairs:
        union(i,j)
        
    merged_groups = defaultdict(list)
    for i, arr in enumerate(blobs):
        root = find(i)
        merged_groups[root].append(arr)
        
    final_merged = {root:np.concatenate(arrays, axis=0)
                    for root, arrays in merged_groups.items()}
    
    new_blobs = []
    visited_roots = set()
    for i in range(n):
        root = find(i)
        if root not in visited_roots:
            new_blobs.append(final_merged[root])
            visited_roots.add(root)
            
    return new_blobs
    
def fill_holes(blobs):
    for i, blob in enumerate(blobs):
        holes = detect_holes(blob)
        for hole in holes:
            blob = np.vstack((blob, hole))
        blobs[i] = blob
    return blobs

def get_blob_edges(blobs):
    blob_edges = []
    for blob in blobs:
        edges = compute_blob_edges(blob)
        blob_edges.append(edges)
    return blob_edges

def compute_blobs_min_circle(blobs):
    blob_centers = []
    radii = []
    for blob in blobs:
        center_x, center_y, radius = smallestenclosingcircle.make_circle(blob)
        blob_centers.append((center_x, center_y))
        radii.append(radius)
    
    return np.array(blob_centers), np.array(radii)

def compute_mask_close_blobs(blob_centers, radii, offset=0):
    dists = np.linalg.norm(blob_centers[:,None,:] - blob_centers[None,:,:], axis=2)
    np.fill_diagonal(dists, np.inf)
    radii_sum = radii[:,None] + radii[None,:]
    return dists < (radii_sum+offset)


@njit
def find_blobs(blurred, threshold, peaks, tol=.05, patience=1, patience_tol=.05):
    Ni, Nj = blurred.shape
    visited = set()
    blobs = []

    directions = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1), (0,1),
        (1,-1), (1,0), (1,1)
    ]
    
    for peak in peaks:
        i0, j0 = peak[0], peak[1]
        if (i0, j0) in visited:
            continue
        
        peak_val = blurred[i0, j0]
        peak_threshold = max(threshold, tol*peak_val)
        q = [(i0, j0, peak_val, 0)]
        current_blob = []
        while len(q):
            i, j, prev_val, streak = q[0]
            q = q[1:]
            
            val = blurred[i, j]
            if val/prev_val > (1+patience_tol):
                streak += 1
            else:
                streak = 0

            if tol*val > peak_threshold:
                peak_threshold = tol*val
                # Remove element below new threshold
                for idx in range(len(current_blob)-1, -1, -1):
                    ii, jj = current_blob[idx]
                    if blurred[ii, jj] < peak_threshold:
                        current_blob.pop(idx)
                        visited.discard((ii, jj))
            
            current_blob.append([i, j])
            for di, dj in directions:
                i2 = i + di
                j2 = j + dj
                if 0 <= i2 < Ni and 0 <= j2 < Nj:
                    if (i2, j2) not in visited and val > peak_threshold and streak <= patience:
                        visited.add((i2, j2))
                        q.append((i2, j2, val, streak))
        blobs.append(np.array(current_blob))
    return blobs

# Define the coordinate type for reuse
coord_type = types.UniTuple(types.int64, 2)

@njit
def compute_blob_edges(blob):
    """
    Computes the edge pixels of a blob of coordinates using Numba-optimized structures.
    """
    if not len(blob):
        return np.empty((0, 2), dtype=np.int64)


    # # N8
    # directions = (
    #     (-1, -1), (-1, 0), (-1, 1),
    #     (0, -1),          (0, 1),
    #     (1, -1), (1, 0), (1, 1)
    # )

    # N4
    directions = (
                (-1, 0),
        (0, -1),         (0, 1),
                (1, 0)
    )

    blob_dict = Dict.empty(key_type=coord_type, value_type=types.uint8)
    for i in range(blob.shape[0]):
        key = (blob[i, 0], blob[i, 1])
        # FIX for Warning: Use explicit uint8 cast
        blob_dict[key] = np.uint8(1)

    edges_dict = Dict.empty(key_type=coord_type, value_type=types.uint8)

    for i in range(blob.shape[0]):
        i_val, j_val = blob[i, 0], blob[i, 1]
        for di, dj in directions:
            ni, nj = i_val + di, j_val + dj
            if (ni, nj) not in blob_dict:
                # FIX for Warning: Use explicit uint8 cast
                edges_dict[(i_val, j_val)] = np.uint8(1)
                break

    n_edges = len(edges_dict)
    if n_edges == 0:
        return np.empty((0, 2), dtype=np.int64)

    edge_array = np.empty((n_edges, 2), dtype=np.int64)
    for i, (r, c) in enumerate(edges_dict.keys()):
        edge_array[i, 0] = r
        edge_array[i, 1] = c

    return edge_array

@njit
def _flood_fill(grid, r, c, fill_value):
    """
    Performs an iterative flood fill on a grid. Modifies the grid in place.
    """
    if grid[r, c] != 0:
        return List.empty_list(coord_type)

    h, w = grid.shape
    stack = List.empty_list(coord_type)
    stack.append((r, c))
    filled_coords = List.empty_list(coord_type)

    while len(stack) > 0:
        r_curr, c_curr = stack.pop()
        if grid[r_curr, c_curr] == 0:
            grid[r_curr, c_curr] = fill_value
            filled_coords.append((r_curr, c_curr))
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r_curr + dr, c_curr + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                    stack.append((nr, nc))
    return filled_coords

@njit
def detect_holes(blob):
    """
    Detects holes within a blob of coordinates using a flood fill algorithm.
    """
    if blob.shape[0] < 3:
        return List.empty_list(types.int64[:, :])

    # --- 1. Setup Grid ---
    # MAJOR FIX: Replace np.min/max with a manual loop to find the bounding box.
    # This is fully supported by all Numba versions.
    min_r = max_r = blob[0, 0]
    min_c = max_c = blob[0, 1]
    for i in range(1, blob.shape[0]):
        r, c = blob[i, 0], blob[i, 1]
        min_r = min(min_r, r)
        min_c = min(min_c, c)
        max_r = max(max_r, r)
        max_c = max(max_c, c)

    offset_r, offset_c = min_r - 1, min_c - 1
    grid_h = max_r - min_r + 3
    grid_w = max_c - min_c + 3
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for i in range(blob.shape[0]):
        r, c = blob[i, 0], blob[i, 1]
        grid[r - offset_r, c - offset_c] = 1

    _flood_fill(grid, 0, 0, 2)

    holes = List.empty_list(types.int64[:, :])
    for r_idx in range(grid_h):
        for c_idx in range(grid_w):
            if grid[r_idx, c_idx] == 0:
                hole_coords_list = _flood_fill(grid, r_idx, c_idx, 3)
                n_hole_pts = len(hole_coords_list)
                if n_hole_pts > 0:
                    hole_array = np.empty((n_hole_pts, 2), dtype=np.int64)
                    for i in range(n_hole_pts):
                        r_hole, c_hole = hole_coords_list[i]
                        hole_array[i, 0] = r_hole + offset_r
                        hole_array[i, 1] = c_hole + offset_c
                    holes.append(hole_array)
    return holes
