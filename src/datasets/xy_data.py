import numpy as np
import networkx as nx
from typing import Optional, Iterator, List
from pyproj import Proj
from scipy.stats import chi2
from sklearn.cluster import KMeans

class XYData:
    def __init__(self, pos: np.ndarray, 
                 covs: np.ndarray,
                 idx: np.ndarray, proj: Proj,
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
                 gamma: Optional[np.ndarray] = None):
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
            self._cache['semi_major'] = semi_major
        if semi_minor is not None:
            self._cache['semi_minor'] = semi_minor
        if phi is not None:
            self._cache['phi'] = phi

        # if self.pos.ndim == 2:
        #     N = len(self.pos)
        #     self.pos = self.pos.reshape(N, -1, 1)

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, key) -> 'XYData':
        """Enable indexing/slicing: data[mask] or data[0:10]"""
        # Extract cached ellipse parameters for the subset
        semi_major = self._cache.get('semi_major')
        semi_minor = self._cache.get('semi_minor')
        phi = self._cache.get('phi')

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
        )
        # transfer other cached arrays to subset
        for name, arr in self._cache.items():
            if name not in ['semi_major', 'semi_minor', 'phi']:
                # For G subgraph use subgraph
                if name == 'G':
                    if key_np.dtype == bool:
                        key_np = np.where(key_np)[0]
                    obj._cache[name] = self._cache['G'].subgraph(key_np).copy()
                elif name == '_inv_cov_sums':  
                    obj._cache[name] = arr[key_np, key_np]
                else:
                    obj._cache[name] = arr[key_np]  # Slice other cached arrays like covs
        return obj

    def split(self, subsets: List[np.ndarray]) -> Iterator['XYData']:
        """Split into multiple XYData objects"""
        for subset in subsets:
            yield self[np.array(list(subset))]

    def split_by_labels(self) -> Iterator['XYData']:
        """Split into multiple XYData objects based on unique labels"""
        if self.labels is None:
            raise ValueError("No labels set.")
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            mask = (self.labels == label)
            yield self[mask]

    def clear_cache(self) -> None:
        """Clear all cached projections."""
        self._cache.clear()

    def clear_cache_item(self, name: str) -> None:
        """Clear a specific cached entry."""
        self._cache.pop(name, None)

    @property
    def x(self) -> np.ndarray:
        """Return array of x coordinates."""
        return self.pos[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return array of y coordinates."""
        return self.pos[:, 1]

    @property
    def semi_major(self) -> np.ndarray:
        """Semi-major axes at confidence level alpha."""
        if 'semi_major' not in self._cache:
            self._compute_ellipse_params()
        return self._cache['semi_major']

    @property
    def semi_minor(self) -> np.ndarray:
        """Semi-minor axes at confidence level alpha."""
        if 'semi_minor' not in self._cache:
            self._compute_ellipse_params()
        return self._cache['semi_minor']

    @property
    def phi(self) -> np.ndarray:
        """Orientation angles (radians) of semi-major axes."""
        if 'phi' not in self._cache:
            self._compute_ellipse_params()
        return self._cache['phi']

    @property
    def a(self) -> np.ndarray:
        """Alias for semi_major axes."""
        return self.semi_major

    @property
    def b(self) -> np.ndarray:
        """Alias for semi_minor axes."""
        return self.semi_minor

    @property
    def inv_covs(self) -> np.ndarray:
        """Inverses of covariance matrices, same shape as covs."""
        key = 'inv_covs'
        if key not in self._cache:
            self._cache[key] = np.linalg.inv(self.covs)
        return self._cache[key]

    @property
    def inv_sum_cov(self) -> np.ndarray:
        """Inverses of sums of every pair of cov matrices, shape NxNx2x2."""
        key = 'inv_sum_cov'
        if key not in self._cache:
            # Use broadcasting: covs[:, None, :, :] + covs[None, :, :, :] gives NxNx2x2 sums
            sums = self.covs[:, None, :, :] + self.covs[None, :, :, :]
            self._cache[key] = np.linalg.inv(sums)
        return self._cache[key]

    @property
    def cov_det(self) -> np.ndarray:
        """Determinant of each covariance matrix, shape (N,)."""
        key = 'cov_det'
        if key not in self._cache:
            self._cache[key] = np.linalg.det(self.covs)
        return self._cache[key]
    
    @property
    def G(self):
        if 'G' not in self._cache:
            self.compute_graph()
        return self._cache['G']

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
        w, v = np.linalg.eigh(self.covs)
        major = np.sqrt(w[:, 1] * c)
        minor = np.sqrt(w[:, 0] * c)
        # eigenvector for largest eigenvalue
        vec = v[..., 1]  # shape (N,2)
        angle = np.arctan2(vec[:, 1], vec[:, 0])
        self._cache['semi_major'] = major
        self._cache['semi_minor'] = minor
        self._cache['phi'] = angle

    def init_cluster_params(self, K: int) -> None:
        """
        Initialize cluster parameters: mu (Kx2), sigma (Kx2x2), pi (K,)
        """
        self.K = K
        self.mu = np.zeros((K, 2))
        self.sigma = np.zeros((K, 2, 2))
        self.pi = np.zeros(K)

    def get_cluster(self, cluster_id: int) -> 'XYData':
        """
        Return the subset of data belonging to the given cluster id.
        """
        if self.labels is None:
            raise ValueError("No cluster labels set for this XYData object.")
        if self.K is not None and (cluster_id < 0 or cluster_id >= self.K):
            raise ValueError(f"Cluster id {cluster_id} out of range for K={self.K}")
        mask = (self.labels == cluster_id)
        return self[mask]

    def iter_clusters(self):
        """
        Iterate over clusters, yielding XYData for each cluster.
        """
        unique_labels = np.unique(self.labels)
        for cid in unique_labels:
            yield cid, self.get_cluster(cid)

    def compute_graph(self, alpha=.95, k=10, threshold=1e-3):
	
        mahal = np.einsum('...i,...ij,...j->...', self.dx, self.inv_sum_cov, self.dx)
        mask = mahal / chi2.ppf(alpha, df=2)
        mask = mask.clip(max=2)
        mask = 1 / (1 + np.exp(k * (mask - 1)))
        mask[mask<threshold] = 0
        self._cache['G'] = nx.from_numpy_array(mask)

    def split_into_connected_components(self):
        components_idx = list(nx.connected_components(self.G))
        components_idx.sort(key=len, reverse=True)

        # If labels already assigned, raise an error
        if self.labels is not None:
            raise ValueError(
            "The input data already has assigned labels."
            )

        # Create labels based on components_idx
        self.labels = np.zeros(self.pos.shape[0], dtype=int)
        for cid, component in enumerate(components_idx):
            self.labels[list(component)] = cid

        return list(self.split(components_idx))

    def spectral_clustering(self, Kmax=50):
        # Ensure the graph is a single connected component
        if not nx.is_connected(self.G):
            raise ValueError("Graph must be a single connected component for spectral clustering.")
        
        L = nx.normalized_laplacian_matrix(self.G).toarray()
        vals, vecs = np.linalg.eigh(L)
        vecs = vecs[:,:Kmax]
        eig_gaps = np.diff(vals[:Kmax])
        sort_idx = np.argsort(eig_gaps)[::-1]
        vecs = vecs[:, :sort_idx[0]+1]
        K = sort_idx[0] + 1
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(vecs)
        self.labels = kmeans.labels_
        self.init_cluster_params(K)
        self.compute_cluster_params()


    def compute_cluster_params(self):
        if self.labels is None:
            raise ValueError("No cluster labels set for this XYData object.")
        
        for label, cluster in self.iter_clusters():
            sum_inv_cov = np.sum(cluster.inv_covs, axis=0)                     # shape (2, 2)
            # weighted sum: for each point, multiply inverse covariance with its position vector
            weighted_sum = np.sum(np.einsum('nij,nj->ni', cluster.inv_covs, cluster.pos), axis=0)  # shape (2,)
            self.mu[label] = np.linalg.solve(sum_inv_cov, weighted_sum)     # mu = inv(sum_inv_cov) @ weighted_sum
            self.sigma[label] = np.linalg.inv(sum_inv_cov)                    # covariance of the estimate
            self.pi[label] = len(cluster) / len(self)