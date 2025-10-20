import numpy as np
from numba import njit
from scipy.sparse import csgraph, find
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.stats import zscore
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms

from .utils.matrices import inverse_2x2_matrices, mahalanobis_2x2_matrices

EPS = 1e-12

def construct_graph(data, n_neighbors=100):
    n_neighbors = min(n_neighbors, len(data) - 1)
    G_dists = kneighbors_graph(data.pos, n_neighbors=n_neighbors, mode='distance', include_self=False)
    return G_dists

def make_symmetric(G):
    return G.maximum(G.T)

def compute_mahalanobis(G_dists, data):

    # Find the indices of the connected nodes (the edges)
    sources, targets, _ = find(G_dists)

    # Get the data for the start and end points of each edge
    pos_sources = data.pos[sources]
    pos_targets = data.pos[targets]
    covs_sources = data.covs[sources]
    covs_targets = data.covs[targets]

    deltas = pos_targets - pos_sources
    sum_covs = covs_sources + covs_targets
    inv_sum_covs = inverse_2x2_matrices(sum_covs)

    mahalanobis_sq = mahalanobis_2x2_matrices(inv_sum_covs, deltas)

    # We can take the square root if we need the actual distance
    mahalanobis_dists = np.sqrt(mahalanobis_sq)

    # Create a new sparse matrix with the Mahalanobis distances
    G_mahalanobis = G_dists.copy()
    G_mahalanobis.data = mahalanobis_dists
    
    return G_mahalanobis

def prune_graph(G, N):
    coo = G.tocoo()
    sorted_indices = np.lexsort((coo.data, coo.row))
    row, col, data = coo.row[sorted_indices], coo.col[sorted_indices], coo.data[sorted_indices]
    _, row_indices = np.unique(row, return_index=True)

    @njit
    def truncate_indices(row_indices, n):
        keep_indices = []
        for i in range(len(row_indices)):
            start_idx = row_indices[i]
            end_idx = row_indices[i + 1] if i + 1 < len(row_indices) else len(row_indices)
            row_length = end_idx - start_idx
            num_to_keep = min(n, row_length)
            keep_indices.extend(range(start_idx, start_idx + num_to_keep))
        return keep_indices

    kept_indices = truncate_indices(row_indices, N)
    pruned_row = row[kept_indices]
    pruned_col = col[kept_indices]
    pruned_data = data[kept_indices]

    return coo_matrix((pruned_data, (pruned_row, pruned_col)), shape=G.shape).tocsr()

def convert_to_similarities(G_dists, sigma='mean', d_squared=False):
    
    
    if sigma == 'mean':
        sigma = G_dists.data.mean()
    elif sigma == 'median':
        sigma = np.median(G_dists.data)
    elif isinstance(sigma, (int, float)):
        sigma = sigma
    else:
        raise ValueError("Invalid sigma. Use 'mean', 'median', or a numeric value.")

    # Copy G_dists to avoid modifying the original
    G_sim = G_dists.copy()
    sq = 1 if d_squared else 2
    G_sim.data = np.exp(-G_sim.data**sq / (2 * sigma**2))
    return G_sim

def remove_high_degree_nodes(G_sim, 
                             max_degree_ratio=20, 
                             z_score_start=3,
                             z_step=.1,
                             verbose=False): 

    if max_degree_ratio is None:
        return G_sim, np.ones(G_sim.shape[0], dtype=bool), np.array(G_sim.sum(axis=1)).flatten()

    degrees = np.array(G_sim.sum(axis=1)).flatten()
    degree_ratio = degrees.max() / degrees.min() if degrees.min() > 0 else float('inf')
    if verbose:
        print(f"Initial degree ratio: {degree_ratio:.1f}")
    z_threshold = z_score_start
    general_mask = np.ones(len(degrees), dtype=bool)
    while degree_ratio > max_degree_ratio:
        z_scores = -zscore(degrees)
        mask = z_scores < z_threshold
        general_mask[general_mask] = mask
        G_sim = G_sim[mask][:, mask]

        z_threshold -= z_step
        if z_threshold <= 1:
            # throw warning
            Warning('Could not reduce degree ratio below threshold, consider adjusting parameters!')
            break
        degrees = np.array(G_sim.sum(axis=1)).flatten()
        degree_ratio = degrees.max() / degrees.min() if degrees.min() > 0 else float('inf')
        if verbose:
            print(f"Degree ratio: {degree_ratio:.1f}, removed {np.sum(~mask)} nodes with z < {z_threshold:.2f}")

    return G_sim, general_mask, degrees

def compute_laplacian(G_sim, normalized=True):
    laplacian = csgraph.laplacian(G_sim, normed=normalized).tocsr()
    return laplacian

def _sort_eigs(vals, vecs):
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]

def compute_eigs(laplacian, degrees, k=10):
    if laplacian.shape[0] < 5000:
        vals, vecs = eigsh(laplacian, 
                     k=k, 
                     which='SM',)
        return _sort_eigs(vals, vecs)
    
    X = np.random.randn(laplacian.shape[0], k)
    X = X / np.sqrt(degrees[:, None] + EPS)
    X = X / np.linalg.norm(X, axis=0)[None, :]
    vals, vecs = lobpcg(
        laplacian, 
        X=X,
        largest=False,
        tol=1e-3,
        maxiter=500
        )
    
    return _sort_eigs(vals, vecs)

def get_optimal_k(eigenvalues, verbose=False):
    gaps = np.diff(eigenvalues)
    if verbose:
        print("Eigenvalue gaps:", gaps)
    # Punish higher K to avoid overclustering
    # penalties = np.log1p(np.arange(1, len(gaps) + 1))
    K = np.argmax(gaps) + 1  # +1 because gaps are between
    return K, gaps

def compute_labels(vecs, K):
    norms = np.sqrt(np.maximum(row_norms(vecs, squared=True), EPS))[:, None]
    vecs /= norms
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(vecs)
    return kmeans.labels_

def spectral_clustering(data,
                        max_k,
                        n_neighbors=100, 
                        sigma='mean', 
                        max_degree_ratio=20, 
                        z_score_start=3,
                        z_step=.1,
                        normalize_laplacian=False,
                        compute_labels=True,
                        return_gaps=False,
                        verbose=False,
                        use_mahalanobis=False,
                        n_mahal=None,
                        mahal_reduction=None):
    
    G_dists = construct_graph(data, n_neighbors=n_neighbors)
    G_dists = make_symmetric(G_dists)
    if use_mahalanobis or mahal_reduction is not None or n_mahal is not None:
        use_mahalanobis = True
        G_dists = compute_mahalanobis(G_dists, data)
        assert mahal_reduction is None or n_mahal is None, "Only one of mahal_reduction and n_mahal can be set."
        if mahal_reduction is not None:
            n_mahal = int(n_neighbors / mahal_reduction)
        if n_mahal is not None and n_mahal < n_neighbors:
            G_dists = prune_graph(G_dists, n_mahal)
        G_dists = make_symmetric(G_dists)

    G_sim = convert_to_similarities(G_dists, sigma=sigma, d_squared=use_mahalanobis)
    G_sim, data_mask, degrees = remove_high_degree_nodes(G_sim, 
                                          max_degree_ratio=max_degree_ratio, 
                                          z_score_start=z_score_start,
                                          verbose=verbose,
                                          z_step=z_step)
    laplacian = compute_laplacian(G_sim, normalized=normalize_laplacian)

    eigenvalues, eigenvectors = compute_eigs(laplacian, degrees, k=max_k)
    K, gaps = get_optimal_k(eigenvalues, verbose=verbose)
    data.set_k(K)
    if verbose:
        print(f"Optimal number of clusters: {K}")
    if not compute_labels:
        if return_gaps:
            return gaps
        else:
            return

    if K == 1:
        labels = np.zeros(data_mask.sum(), dtype=int)
    else:
        labels = compute_labels(eigenvectors, K)

    data.labels = -1 * np.ones(len(data), dtype=int)
    data.labels[data_mask] = labels

    if return_gaps:
        return gaps