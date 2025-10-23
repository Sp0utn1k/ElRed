import numpy as np
from typing import Optional, Iterator, List, Tuple
from pyproj import Proj
from scipy.stats import chi2
from sklearn.cluster import KMeans
from numba import njit, types
from numba.typed import Dict

from ..utils.matrices import (
    inverse_2x2_matrices,
    determinant_2x2_matrices,
    eigh_2x2_matrices,
    solve_2x2_matrices,
    mahalanobis_2x2_matrices,
)


@njit
def _assign_points_to_blobs_njit(
    ix: np.ndarray, iy: np.ndarray, coord_to_blob_id: Dict
) -> np.ndarray:
    """
    Assigns each point (ix, iy) to a blob ID using a pre-built map.
    Points not in any blob are assigned -1.
    """
    num_points = len(ix)
    assignments = np.full(num_points, -1, dtype=np.int64)
    for i in range(num_points):
        coord = (ix[i], iy[i])
        if coord in coord_to_blob_id:
            assignments[i] = coord_to_blob_id[coord]
    return assignments


class XYData:
    def __init__(
        self,
        pos: np.ndarray,
        covs: np.ndarray,
        idx: np.ndarray,
        proj: Proj,
        timestamps: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
        ellipse_alpha: float = 0.95,
        semi_major: Optional[np.ndarray] = None,
        semi_minor: Optional[np.ndarray] = None,
        phi: Optional[np.ndarray] = None,
        K: Optional[int] = None,
        pi: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        gamma: Optional[np.ndarray] = None,
        cell_size: Optional[float] = None,
        parent=None,
    ):
        """
        Initialize XYData with positions, covariance matrices, indices, projection (pyproj.Proj),
        optional timestamps, metadata, confidence level ellipse_alpha for ellipse,
        optional pre-computed ellipse parameters, and optional clustering parameters.

        """
        self.pos = pos  # x, y in meters
        self.covs = covs  # 2x2 covariance matrices
        self.idx = idx  # Original indices preserved
        self.proj = proj  # pyproj.Proj instance
        self.timestamps = timestamps
        self.metadata = metadata or {}
        self.ellipse_alpha = ellipse_alpha
        self._cache = {}  # initialize cache

        # Store clustering parameters
        self.K = K
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.labels = labels
        self.gamma = gamma

        # Store pre-computed ellipse parameters if provided
        if semi_major is not None:
            self._cache["semi_major"] = semi_major
        if semi_minor is not None:
            self._cache["semi_minor"] = semi_minor
        if phi is not None:
            self._cache["phi"] = phi

        self.g = cell_size

        self.parent = parent

        # if self.pos.ndim == 2:
        #     N = len(self.pos)
        #     self.pos = self.pos.reshape(N, -1, 1)

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, key) -> "XYData":
        """Enable indexing/slicing: data[mask] or data[0:10]"""
        # Extract cached ellipse parameters for the subset
        semi_major = self._cache.get("semi_major")
        semi_minor = self._cache.get("semi_minor")
        phi = self._cache.get("phi")

        # Ensure key is a numpy array for advanced indexing
        key_np = np.array(key) if isinstance(key, (list, set)) else key

        obj = XYData(
            pos=self.pos[key_np],
            covs=self.covs[key_np],
            idx=self.idx[key_np],
            proj=self.proj,
            timestamps=self.timestamps[key_np] if self.timestamps is not None else None,
            metadata=self.metadata.copy(),
            semi_major=semi_major[key_np] if semi_major is not None else None,
            semi_minor=semi_minor[key_np] if semi_minor is not None else None,
            phi=phi[key_np] if phi is not None else None,
            K=self.K,
            pi=self.pi,
            mu=self.mu,
            sigma=self.sigma,
            cell_size=self.g,
            labels=self.labels[key_np] if self.labels is not None else None,
            parent=self.parent or self,
        )
        # transfer other cached arrays to subset
        for name, arr in self._cache.items():
            if name not in ["semi_major", "semi_minor", "phi"]:
                # For G subgraph use subgraph
                if name == "G":
                    if key_np.dtype == bool:
                        key_np = np.where(key_np)[0]
                    obj._cache[name] = self._cache["G"].subgraph(key_np).copy()
                elif name == "_inv_cov_sums":
                    obj._cache[name] = arr[key_np, key_np]
                else:
                    obj._cache[name] = arr[
                        key_np
                    ]  # Slice other cached arrays like covs
        return obj

    def get_by_index(self, indices: np.ndarray) -> "XYData":
        """Get subset by original indices."""
        mask = np.isin(self.idx, indices)
        return self[mask]

    def reset_labels(self) -> None:
        """Reset cluster labels and parameters."""
        self.labels = None
        self.K = None
        self.mu = None
        self.sigma = None
        self.pi = None
        self.gamma = None
        self.clear_cache_item("G")

    def split(self, subsets: List[np.ndarray]) -> Iterator["XYData"]:
        """Split into multiple XYData objects"""
        for subset in subsets:
            yield self[np.array(list(subset))]

    def split_by_labels(self) -> Iterator["XYData"]:
        """Split into multiple XYData objects based on unique labels, with cluster parameters adapted for each subset."""
        if self.labels is None:
            raise ValueError("No labels set.")
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = self.labels == label
            subset = self[mask]
            # Adapt cluster parameters for this subset
            subset.K = 1
            if self.mu is not None:
                subset.mu = np.array([self.mu[label]])
            if self.sigma is not None:
                subset.sigma = np.array([self.sigma[label]])
            if self.pi is not None:
                subset.pi = np.array([self.pi[label]])
            if self.gamma is not None:
                subset.gamma = np.array([self.gamma[mask]])  # gamma is per-point
            subset.labels = np.zeros(len(subset), dtype=int)
            yield subset

    def clear_cache(self) -> None:
        """Clear all cached projections."""
        self._cache.clear()

    def clear_cache_item(self, name: str) -> None:
        """Clear a specific cached entry."""
        self._cache.pop(name, None)

    @property
    def posi(self) -> np.ndarray:
        """Return positions in integer grid coordinates (rounded)."""
        return np.stack((self.ix, self.iy), axis=1)

    @property
    def x(self) -> np.ndarray:
        """Return array of x coordinates."""
        return self.pos[:, 0]

    @property
    def ix(self) -> np.ndarray:
        return ((self.x - self.x_min) // self.g).astype(int)

    @property
    def x_min(self) -> float:
        return self.x.min()

    @property
    def x_max(self) -> float:
        return self.x.max()

    @property
    def y(self) -> np.ndarray:
        """Return array of y coordinates."""
        return self.pos[:, 1]

    @property
    def iy(self) -> np.ndarray:
        return ((self.y - self.y_min) // self.g).astype(int)

    @property
    def y_min(self) -> float:
        return self.y.min()

    @property
    def y_max(self) -> float:
        return self.y.max()

    @property
    def cell_size(self) -> float:
        return self.g or self.compute_cell_size()

    @property
    def grid_size(self) -> Tuple[int]:
        Ni = int((self.x_max - self.x_min) // self.cell_size) + 1
        Nj = int((self.y_max - self.y_min) // self.cell_size) + 1
        return (Ni, Nj)

    @property
    def semi_major(self) -> np.ndarray:
        """Semi-major axes at confidence level alpha."""
        if "semi_major" not in self._cache:
            self._compute_ellipse_params()
        return self._cache["semi_major"]

    @property
    def semi_minor(self) -> np.ndarray:
        """Semi-minor axes at confidence level alpha."""
        if "semi_minor" not in self._cache:
            self._compute_ellipse_params()
        return self._cache["semi_minor"]

    @property
    def phi(self) -> np.ndarray:
        """Orientation angles (radians) of semi-major axes."""
        if "phi" not in self._cache:
            self._compute_ellipse_params()
        return self._cache["phi"]

    @property
    def a(self) -> np.ndarray:
        """Alias for semi_major axes."""
        return self.semi_major

    @property
    def b(self) -> np.ndarray:
        """Alias for semi_minor axes."""
        return self.semi_minor

    @property
    def r(self) -> np.ndarray:
        return np.sqrt(self.a * self.b)

    @property
    def inv_covs(self) -> np.ndarray:
        """Inverses of covariance matrices, same shape as covs."""
        key = "inv_covs"
        if key not in self._cache:
            self._cache[key] = inverse_2x2_matrices(self.covs)
        return self._cache[key]

    @property
    def inv_sum_cov(self) -> np.ndarray:
        """Inverses of sums of every pair of cov matrices, shape NxNx2x2."""
        key = "inv_sum_cov"
        if key not in self._cache:
            # Use broadcasting: covs[:, None, :, :] + covs[None, :, :, :] gives NxNx2x2 sums
            sums = self.covs[:, None, :, :] + self.covs[None, :, :, :]
            self._cache[key] = inverse_2x2_matrices(sums)
        return self._cache[key]

    @property
    def cov_det(self) -> np.ndarray:
        """Determinant of each covariance matrix, shape (N,)."""
        key = "cov_det"
        if key not in self._cache:
            self._cache[key] = determinant_2x2_matrices(self.covs)
        return self._cache[key]

    @property
    def gauss_denom(self) -> np.ndarray:
        """2*pi*det(cov) for each covariance matrix, shape (N,). Not cached."""
        return 2 * np.pi * np.sqrt(self.cov_det)

    @property
    def N_clusters(self) -> Optional[int]:
        """Alias for K (number of clusters)."""
        return self.K

    @property
    def x0(self) -> np.ndarray:
        return self.pos[:, None, :] - self.mu[None, :, :]

    @property
    def dx(self) -> np.ndarray:
        return self.pos[:, None, :] - self.pos[None, :, :]

    def _compute_ellipse_params(self) -> None:
        """Compute and cache semi_major, semi_minor, and phi from covariances."""
        # chi-square factor for given confidence level
        c = chi2.ppf(self.ellipse_alpha, df=2)
        # eigen-decomposition: w[...,0] ≤ w[...,1]
        w, v = eigh_2x2_matrices(self.covs)
        major = np.sqrt(w[:, 1] * c)
        minor = np.sqrt(w[:, 0] * c)
        # eigenvector for largest eigenvalue
        vec = v[..., 1]  # shape (N,2)
        angle = np.arctan2(vec[:, 1], vec[:, 0])
        self._cache["semi_major"] = major
        self._cache["semi_minor"] = minor
        self._cache["phi"] = angle

    def get_grid_count(
        self,
        cell_size=None,
        k=None,
        max_memory=200e6,
        largest=False,
        g_min=10,
        g_max=500,
    ) -> np.ndarray:

        if largest:
            self.g = self.compute_largest_cell_size(max_memory)
        elif cell_size is not None:
            self.g = cell_size
        elif k is not None:
            self.g = self.compute_cell_size(k=k)
        else:
            self.g = self.compute_cell_size()

        grid_memory = np.prod(self.grid_size) * 8
        if grid_memory > max_memory:
            self.g *= np.sqrt(grid_memory / max_memory)

        self.g = np.clip(self.g, g_min, g_max)
        grid_memory = np.prod(self.grid_size) * 8
        if grid_memory > max_memory:
            raise MemoryError(
                f"Grid too large even with max cell size {g_max}m: "
                f"{self.grid_size} cells, {grid_memory/1e6:.1f}MB"
            )

        counts = np.zeros(self.grid_size, dtype=int)
        for i, j in zip(self.ix, self.iy):
            counts[i, j] += 1
        return counts

    def compute_largest_cell_size(self, max_memory):
        """Compute the largest cell size such that the grid fits within max_memory (in bytes)."""
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        # Calculate the maximum number of cells allowed by max_memory
        max_cells = max_memory // 8  # assuming float64 (8 bytes)
        # Calculate the aspect ratio of the bounding box
        aspect_ratio = x_range / y_range if y_range != 0 else 1.0
        # Solve for cell size g given max_cells and aspect_ratio
        g = np.sqrt((x_range * y_range) / (max_cells * aspect_ratio))
        return g

    def set_k(self, K: int, remove_existing: bool = True) -> None:
        """
        Set the number of clusters K.
        """
        if remove_existing:
            self.reset_labels()
        self.K = K

    def get_cluster(self, cluster_id: int) -> "XYData":
        """
        Return the subset of data belonging to the given cluster id.
        """
        if self.labels is None:
            raise ValueError("No cluster labels set for this XYData object.")
        if self.K is not None and (cluster_id < 0 or cluster_id >= self.K):
            raise ValueError(f"Cluster id {cluster_id} out of range for K={self.K}")
        mask = self.labels == cluster_id
        return self[mask]

    def iter_clusters(self):
        """
        Iterate over clusters, yielding XYData for each cluster.
        """
        if self.labels is None:
            raise ValueError("No cluster labels set for this XYData object.")
        unique_labels = np.unique(self.labels)
        for cid in unique_labels:
            yield cid, self.get_cluster(cid)

    def compute_adjacency_matrix(self, threshold=1e-3, alpha=0.95, k=10) -> np.ndarray:
        mahal = mahalanobis_2x2_matrices(self.inv_covs, self.dx)
        A = mahal / chi2.ppf(alpha, df=2)
        A = A.clip(max=5)
        A = 1 / (1 + np.exp(k * (A - 1)))
        A[A < threshold] = 0
        return A

    def compute_cluster_params(self):
        if self.labels is None:
            raise ValueError("No cluster labels set for this XYData object.")

        for label, cluster in self.iter_clusters():
            sum_inv_cov = np.sum(cluster.inv_covs, axis=0)  # shape (2, 2)
            # weighted sum: for each point, multiply inverse covariance with its position vector
            weighted_sum = np.sum(
                np.einsum("nij,nj->ni", cluster.inv_covs, cluster.pos), axis=0
            )  # shape (2,)
            self.mu[label] = solve_2x2_matrices(sum_inv_cov[None], weighted_sum[None])[
                0
            ]  # mu = inv(sum_inv_cov) @ weighted_sum
            self.sigma[label] = inverse_2x2_matrices(sum_inv_cov[None])[
                0
            ]  # covariance of the estimate
            self.pi[label] = len(cluster) / len(self)

    def compute_cell_size(self, k=0.5):
        return np.median(self.r) * k

    def extract_bounding_box(
        self, x_min=None, x_max=None, y_min=None, y_max=None, inverted=False
    ):
        mask = np.ones(len(self), dtype=bool)
        if x_min is not None:
            mask &= self.x >= x_min
        if x_max is not None:
            mask &= self.x <= x_max
        if y_min is not None:
            mask &= self.y >= y_min
        if y_max is not None:
            mask &= self.y <= y_max
        if inverted:
            mask = ~mask
        return self[mask]

    def extract_grid_bounding_box(
        self, i_min=None, i_max=None, j_min=None, j_max=None, inverted=False, margin=0
    ):

        x_min = i_min or self.x_min + (i_min + margin) * self.g
        x_max = i_max or self.x_min + (i_max + margin) * self.g
        y_min = j_min or self.y_min + (j_min + margin) * self.g
        y_max = j_max or self.y_min + (j_max + margin) * self.g
        return self.extract_bounding_box(x_min, x_max, y_min, y_max, inverted=inverted)

    def split_blobs(
        self, blobs: List[np.ndarray]
    ) -> Tuple[List["XYData"], Optional["XYData"]]:
        """
        Split into multiple XYData objects based on list of blobs using a fast numba-based approach.
        A blob is a list of (ix, iy) grid coordinate pairs.
        Returns a tuple: (list of blob subsets, remaining points subset).
        """
        if not blobs:
            if len(self) > 0:
                return [self], None
            return [], None

        # 1. Build a map from (ix, iy) coordinates to blob ID.
        # Numba requires typed dictionaries for JIT compilation.
        coord_to_blob_id = Dict.empty(
            key_type=types.UniTuple(types.int64, 2),
            value_type=types.int64,
        )
        for blob_id, blob in enumerate(blobs):
            for i in range(blob.shape[0]):
                coord_to_blob_id[(int(blob[i, 0]), int(blob[i, 1]))] = blob_id

        # 2. Assign each point to a blob ID in a single pass using numba.
        assignments = _assign_points_to_blobs_njit(
            self.ix.astype(np.int64), self.iy.astype(np.int64), coord_to_blob_id
        )

        # 3. Collect subsets for each blob.
        blob_subsets = []
        for blob_id in range(len(blobs)):
            mask = assignments == blob_id
            if np.any(mask):
                blob_subsets.append(self[mask])

        # 4. Collect remaining points not in any blob.
        remaining_mask = assignments == -1
        remaining_subset = self[remaining_mask] if np.any(remaining_mask) else None

        return blob_subsets, remaining_subset

    @staticmethod
    def merge(xydata_list: List["XYData"]) -> "XYData":
        """
        Merge a list of XYData objects into a single XYData.
        - All must have the same projection.
        - Metadata is shallow-merged (later entries override earlier).
        - Labels are offset so that same label in different datasets are not merged.
        - If a dataset has no labels, it is initialized with zeros (new label group).
        """
        if not xydata_list:
            raise ValueError("No XYData objects to merge.")
        proj = xydata_list[0].proj
        for d in xydata_list:
            if d.proj != proj:
                raise ValueError("All XYData objects must have the same projection.")

        # If all sets have same parent, preserve it
        parent = None
        if all(d.parent == xydata_list[0].parent for d in xydata_list):
            parent = xydata_list[0].parent

        pos = np.concatenate([d.pos for d in xydata_list], axis=0)
        covs = np.concatenate([d.covs for d in xydata_list], axis=0)
        idx = np.concatenate([d.idx for d in xydata_list], axis=0)
        timestamps = None
        if all(d.timestamps is not None for d in xydata_list):
            timestamps = np.concatenate([d.timestamps for d in xydata_list], axis=0)

        # Merge cached ellipse params if present
        semi_major = None
        semi_minor = None
        phi = None
        if all("semi_major" in d._cache for d in xydata_list):
            semi_major = np.concatenate([d._cache["semi_major"] for d in xydata_list])
        if all("semi_minor" in d._cache for d in xydata_list):
            semi_minor = np.concatenate([d._cache["semi_minor"] for d in xydata_list])
        if all("phi" in d._cache for d in xydata_list):
            phi = np.concatenate([d._cache["phi"] for d in xydata_list])

        # Merge labels: offset so that same label in different datasets are not merged
        labels_list = []
        label_offset = 0
        for d in xydata_list:
            if d.labels is None:
                # All zeros, new group
                labels = np.zeros(len(d), dtype=int) + label_offset
                label_offset += 1
            else:
                labels = d.labels + label_offset
                label_offset += d.labels.max() + 1
            labels_list.append(labels)
        labels = np.concatenate(labels_list)

        # Merge metadata (shallow merge)
        metadata = {}
        for d in xydata_list:
            metadata.update(d.metadata)

        return XYData(
            pos=pos,
            covs=covs,
            idx=idx,
            proj=proj,
            timestamps=timestamps,
            metadata=metadata,
            semi_major=semi_major,
            semi_minor=semi_minor,
            phi=phi,
            labels=labels,
            parent=parent,
        )
