import numpy as np
from typing import List, Optional, Tuple
from collections import defaultdict

from .datasets import LatLonData

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ----------------- Utilities: packing cell keys into int64 -----------------

INT32_BIAS = np.int64(1 << 31)

@njit(cache=True, fastmath=True)
def pack_key(ix: np.int64, iy: np.int64) -> np.int64:
    # Pack two signed 32-bit ints into one signed 64-bit key (order-preserving).
    return ((ix + INT32_BIAS) << 32) | (iy + INT32_BIAS)

# ----------------- Numba DSU (Union-Find) -----------------

@njit(cache=True, fastmath=True)
def dsu_find(parent: np.ndarray, x: int) -> int:
    # Path compression
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit(cache=True, fastmath=True)
def dsu_union(parent: np.ndarray, rank: np.ndarray, a: int, b: int) -> None:
    ra = dsu_find(parent, a)
    rb = dsu_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    if rank[ra] == rank[rb]:
        rank[ra] += 1

# ----------------- Overlap tests -----------------

@njit(cache=True, fastmath=True)
def overlaps_box(lat0_a, lat1_a, lon0_a, lon1_a,
                 lat0_b, lat1_b, lon0_b, lon1_b) -> bool:
    # Inclusive on touching
    return not (lat1_a <= lat0_b or lat1_b <= lat0_a or
                lon1_a <= lon0_b or lon1_b <= lon0_a)

# ----------------- Build cell touches -----------------
# We do a two-pass build:
# 1) Count how many cells each rectangle touches (with antimeridian handling).
# 2) Prefix-sum to allocate flat arrays (keys, rect_ids), then fill them.
# These two passes are parallel over rectangles.

@njit(cache=True, fastmath=True, parallel=True)
def count_cell_touches(lat0: np.ndarray, lat1: np.ndarray,
                       lon0: np.ndarray, lon1: np.ndarray,
                       s: float) -> np.ndarray:
    n = lat0.shape[0]
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        iy0 = int(np.floor(lat0[i] / s))
        iy1 = int(np.floor(lat1[i] / s))
        if lon0[i] <= lon1[i]:
            ix0 = int(np.floor(lon0[i] / s))
            ix1 = int(np.floor(lon1[i] / s))
            counts[i] = (iy1 - iy0 + 1) * (ix1 - ix0 + 1)
        else:
            # Antimeridian wrap: split into two ranges
            ix0a = int(np.floor(lon0[i] / s))
            ix1a = int(np.floor(180.0 / s))
            ix0b = int(np.floor(-180.0 / s))
            ix1b = int(np.floor(lon1[i] / s))
            c1 = (iy1 - iy0 + 1) * (ix1a - ix0a + 1)
            c2 = (iy1 - iy0 + 1) * (ix1b - ix0b + 1)
            counts[i] = c1 + c2
    return counts

@njit(cache=True, fastmath=True, parallel=True)
def fill_cell_touches(lat0: np.ndarray, lat1: np.ndarray,
                      lon0: np.ndarray, lon1: np.ndarray,
                      s: float,
                      offsets: np.ndarray,
                      keys_out: np.ndarray,
                      ids_out: np.ndarray) -> None:
    n = lat0.shape[0]
    for i in prange(n):
        base = offsets[i]
        iy0 = int(np.floor(lat0[i] / s))
        iy1 = int(np.floor(lat1[i] / s))
        if lon0[i] <= lon1[i]:
            ix0 = int(np.floor(lon0[i] / s))
            ix1 = int(np.floor(lon1[i] / s))
            k = 0
            for ix in range(ix0, ix1 + 1):
                pkey = pack_key(np.int64(ix), np.int64(0))  # reuse var for speed (iy differs)
                for iy in range(iy0, iy1 + 1):
                    keys_out[base + k] = pkey + (np.int64(iy) + INT32_BIAS)
                    ids_out[base + k] = i
                    k += 1
        else:
            # Two longitude segments
            ix0a = int(np.floor(lon0[i] / s))
            ix1a = int(np.floor(180.0 / s))
            ix0b = int(np.floor(-180.0 / s))
            ix1b = int(np.floor(lon1[i] / s))
            k = 0
            for ix in range(ix0a, ix1a + 1):
                pkey = pack_key(np.int64(ix), np.int64(0))
                for iy in range(iy0, iy1 + 1):
                    keys_out[base + k] = pkey + (np.int64(iy) + INT32_BIAS)
                    ids_out[base + k] = i
                    k += 1
            for ix in range(ix0b, ix1b + 1):
                pkey = pack_key(np.int64(ix), np.int64(0))
                for iy in range(iy0, iy1 + 1):
                    keys_out[base + k] = pkey + (np.int64(iy) + INT32_BIAS)
                    ids_out[base + k] = i
                    k += 1

# ----------------- Union inside groups (cells) -----------------

@njit(cache=True, fastmath=True)
def union_in_groups(keys_sorted: np.ndarray,
                    ids_sorted: np.ndarray,
                    group_starts: np.ndarray,
                    lat_min: np.ndarray, lat_max: np.ndarray,
                    lon_min: np.ndarray, lon_max: np.ndarray,
                    parent: np.ndarray, rank: np.ndarray,
                    check_overlap: bool) -> None:
    ng = group_starts.shape[0]
    n = ids_sorted.shape[0]
    # Add sentinel end
    for gi in range(ng):
        start = group_starts[gi]
        end = group_starts[gi + 1] if gi + 1 < ng else n
        m = end - start
        if m <= 1:
            continue
        if not check_overlap:
            # Star union: connect all to first
            base_id = ids_sorted[start]
            for k in range(start + 1, end):
                dsu_union(parent, rank, base_id, ids_sorted[k])
        else:
            # Quadratic in-group, but m expected small (tuned by grid size)
            for aidx in range(start, end):
                ia = ids_sorted[aidx]
                la0 = lat_min[ia]; la1 = lat_max[ia]
                lo0 = lon_min[ia]; lo1 = lon_max[ia]
                for bidx in range(aidx + 1, end):
                    ib = ids_sorted[bidx]
                    if overlaps_box(la0, la1, lo0, lo1,
                                    lat_min[ib], lat_max[ib], lon_min[ib], lon_max[ib]):
                        dsu_union(parent, rank, ia, ib)

# ----------------- Public API -----------------

def precluster_latlon_boxes(
    data,
    cell_size_deg: Optional[float] = None,
    epsilon_frac: float = 1e-6,
    check_overlap: bool = False,
    return_labels: bool = False,
    max_cell_entries: int = 20_000_000,  # memory guard
):
    """
    High-speed preclustering with Numba. Falls back to streaming hashed grid if
    the number of cell entries would be too large or if Numba is unavailable.

    Returns: (subsets: List[LatLonData], labels: np.ndarray[int])
    """
    if data.labels is not None:
        raise ValueError(
            "The input data already has assigned labels."
        )
    n = len(data)
    if n == 0:
        return [], (np.empty(0, dtype=np.int32) if return_labels else None)

    # Extract bboxes in degrees
    bb_min = data.bbox_min
    bb_max = data.bbox_max
    lat_min = bb_min[:, 0].astype(np.float64, copy=False)
    lon_min = bb_min[:, 1].astype(np.float64, copy=False)
    lat_max = bb_max[:, 0].astype(np.float64, copy=False)
    lon_max = bb_max[:, 1].astype(np.float64, copy=False)

    # Auto cell size: median max side
    if cell_size_deg is None:
        h = (lat_max - lat_min)
        w = (lon_max - lon_min)
        s = float(np.median(np.maximum(h, w)))
        s = max(s, 1e-7)  # keep indices within signed 32-bit
    else:
        s = float(cell_size_deg)

    eps = s * max(1e-12, float(epsilon_frac))
    lat0 = lat_min - eps
    lat1 = lat_max + eps
    lon0 = lon_min - eps
    lon1 = lon_max + eps

    # Fallback if numba unavailable
    if not NUMBA_AVAILABLE:
        # Use the streaming hashed-grid version (previous answer)
        return precluster_latlon_boxes_streaming(
            data, cell_size_deg=s, epsilon_frac=epsilon_frac,
            check_overlap=check_overlap, return_labels=return_labels
        )

    # 1) Count touches
    counts = count_cell_touches(lat0, lat1, lon0, lon1, s)
    total = int(counts.sum())
    if total <= 0:
        # No cells? Degenerate; each is its own component
        labels = np.arange(n, dtype=np.int32)
        subsets = [data[np.array([i]) ] for i in range(n)]
        return subsets, labels

    # Memory guard
    if total > max_cell_entries:
        # Fall back to streaming hashed-grid (O(total) but sparse memory)
        return precluster_latlon_boxes_streaming(
            data, cell_size_deg=s, epsilon_frac=epsilon_frac,
            check_overlap=check_overlap, return_labels=return_labels
        )

    # 2) Prefix sum and fill
    offsets = np.empty(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    offsets[0] = 0
    keys = np.empty(total, dtype=np.int64)
    ids = np.empty(total, dtype=np.int32)
    fill_cell_touches(lat0, lat1, lon0, lon1, s, offsets, keys, ids)

    # 3) Sort by key, then group by equal keys
    order = np.argsort(keys, kind='quicksort')
    keys_sorted = keys[order]
    ids_sorted = ids[order]

    # group starts where key changes
    # Avoid Python loops: do it in NumPy
    change = np.empty_like(keys_sorted, dtype=np.bool_)
    change[0] = True
    change[1:] = keys_sorted[1:] != keys_sorted[:-1]
    group_starts = np.flatnonzero(change).astype(np.int64)

    # 4) DSU arrays
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.uint8)

    # 5) Unions inside groups (JIT-compiled)
    union_in_groups(keys_sorted, ids_sorted, group_starts,
                    lat_min, lat_max, lon_min, lon_max,
                    parent, rank, check_overlap)

    # 6) Extract compact labels
    # Path compress all
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = dsu_find(parent, i)
    # Relabel 0..k-1
    uniq, inv = np.unique(roots, return_inverse=True)
    labels = inv.astype(np.int32, copy=False)

    # 7) Build subsets
    subsets: List['LatLonData'] = []
    for cid in range(len(uniq)):
        idx = np.nonzero(labels == cid)[0]
        subsets.append(data[idx])

    # Sort subsets by size descending
    subsets = sorted(subsets, key=lambda g: len(g), reverse=True)

    # Assign labels to data
    data.labels = labels
    
    if return_labels:
        return subsets, labels
    else:
        return subsets, None

# ----------------- Streaming fallback (dict-based) -----------------

def precluster_latlon_boxes_streaming(
    data,
    cell_size_deg: float,
    epsilon_frac: float = 1e-6,
    check_overlap: bool = True,
    return_labels: bool = True,
):
    """
    Sparse hashed-grid streaming version (pure Python + NumPy).
    Used as fallback when Numba isn't available or memory guard triggers.
    """
    n = len(data)
    if n == 0:
        return [], (np.empty(0, dtype=np.int32) if return_labels else None)

    bb_min = data.bbox_min; bb_max = data.bbox_max
    lat_min = bb_min[:, 0].astype(np.float64, copy=False)
    lon_min = bb_min[:, 1].astype(np.float64, copy=False)
    lat_max = bb_max[:, 0].astype(np.float64, copy=False)
    lon_max = bb_max[:, 1].astype(np.float64, copy=False)

    s = float(cell_size_deg)
    eps = s * max(1e-12, float(epsilon_frac))

    from math import floor
    def pk(ix, iy):
        return ((int(ix) + (1<<31)) << 32) | (int(iy) + (1<<31))

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.uint8)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]: ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]: rank[ra] += 1

    cells = defaultdict(list)
    for i in range(n):
        la0 = lat_min[i] - eps; la1 = lat_max[i] + eps
        lo0 = lon_min[i] - eps; lo1 = lon_max[i] + eps
        iy0 = int(floor(la0 / s)); iy1 = int(floor(la1 / s))
        def handle(l0, l1):
            ix0 = int(floor(l0 / s)); ix1 = int(floor(l1 / s))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    key = pk(ix, iy)
                    vec = cells[key]
                    if vec:
                        if check_overlap:
                            for j in vec:
                                if not (lat_max[j] <= la0 or la1 <= lat_min[j] or
                                        lon_max[j] <= lo0 or lo1 <= lon_min[j]):
                                    union(i, j)
                        else:
                            for j in vec:
                                union(i, j)
                    vec.append(i)
        if lo0 <= lo1:
            handle(lo0, lo1)
        else:
            handle(lo0, 180.0)
            handle(-180.0, lo1)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    uniq, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int32)

    subsets: List['LatLonData'] = []
    for cid in range(len(uniq)):
        idx = np.nonzero(labels == cid)[0]
        subsets.append(data[idx])

    return (subsets, labels) if return_labels else (subsets, None)
