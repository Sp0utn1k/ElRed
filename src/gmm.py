import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


from .utils.matrices import (
    inverse_2x2_matrices, 
    determinant_2x2_matrices, 
    eigh_2x2_matrices, 
    norm_2d, 
    solve_2x2_matrices,
	mahalanobis_2x2_matrices
)

_LOG_2PI = np.log(2.0 * np.pi)

def _ensure_positive_definite(cov, min_eig=1e-6, dtype=np.float32):
	eigvals, eigvecs = eigh_2x2_matrices(cov[None])
	eigvals, eigvecs = eigvals[0], eigvecs[0]
	eigvals = np.clip(eigvals, min_eig, None)
	return np.einsum('...ik,...k,...jk->...ij', eigvecs, eigvals, eigvecs).astype(dtype, copy=False)

def _effective_covariances(dataset):
	return dataset.covs[:, None, :, :] + dataset.sigma[None, :, :, :]

def bivariate_gaussian_log_pdf(dataset):
	if dataset.sigma is None:
		raise ValueError("dataset.sigma must be initialized before computing log-pdf.")
	eff_cov = _effective_covariances(dataset)
	inv_eff = inverse_2x2_matrices(eff_cov)
	delta = dataset.pos[:, None, :] - dataset.mu[None, :, :]
	# For 2x2 matrices, determinant_2x2_matrices is sufficient; slogdet sign is always positive for valid covariances
	log_det = np.log(determinant_2x2_matrices(eff_cov))
	mahal = mahalanobis_2x2_matrices(inv_eff, delta)
	log_pdf = -0.5 * (mahal + log_det + 2 * _LOG_2PI)
	return log_pdf, inv_eff

def bivariate_gaussian_pdf(dataset):
	log_pdf, _ = bivariate_gaussian_log_pdf(dataset)
	return np.exp(log_pdf)

def E_step(dataset, normalize=True):
	log_pdf, inv_eff = bivariate_gaussian_log_pdf(dataset)
	if dataset.pi is None:
		K = log_pdf.shape[1]
		dataset.pi = np.full(K, 1.0 / K, dtype=dataset.pos.dtype)
	log_pi = np.log(np.clip(dataset.pi, 1e-12, None))
	log_gamma = log_pdf + log_pi
	log_norm = logsumexp(log_gamma, axis=1, keepdims=True)
	gamma = np.exp(log_gamma - log_norm) if normalize else np.exp(log_gamma)
	dataset.gamma = gamma.astype(dataset.pos.dtype, copy=False)
	return inv_eff, float(np.sum(log_norm))

def M_step(dataset, inv_eff, previous_sigma=None, min_eig=1e-6):
	Nk = dataset.gamma.sum(axis=0)
	N = dataset.gamma.shape[0]
	eps = 1e-12
	dataset.pi = np.clip(Nk / N, eps, None)
	dataset.pi /= dataset.pi.sum()
	precision_sum = np.einsum('nk,nkij->kij', dataset.gamma, inv_eff)
	precision_sum = _ensure_positive_definite(precision_sum, min_eig=min_eig, dtype=dataset.pos.dtype)
	rhs = np.einsum('nk,nkij,nj->ki', dataset.gamma, inv_eff, dataset.pos).astype(dataset.pos.dtype)
	mu_candidate = solve_2x2_matrices(precision_sum, rhs)
	mask = (Nk > eps)[:, None]
	dataset.mu = np.where(mask, mu_candidate, dataset.mu)
	dataset.sigma = compute_sigma(dataset, inv_eff, previous_sigma=previous_sigma, min_eig=min_eig)

def compute_sigma(dataset, inv_eff, previous_sigma=None, min_eig=1e-6):
	if dataset.gamma is None or dataset.mu is None:
		raise ValueError("dataset.gamma and dataset.mu must be set before computing sigma.")
	Nk = dataset.gamma.sum(axis=0)
	K = dataset.gamma.shape[1]
	d = dataset.pos.shape[1]
	Nk_safe = np.maximum(Nk, min_eig)
	sigma_prior = previous_sigma if previous_sigma is not None else dataset.sigma
	if sigma_prior is None:
		identity = np.eye(d, dtype=dataset.pos.dtype)
		sigma_prior = np.broadcast_to(identity, (K, d, d)).copy()
	delta = dataset.pos[:, None, :] - dataset.mu[None, :, :]
	A = np.einsum('kij,nkjl->nkil', sigma_prior, inv_eff)
	m_minus_mu = np.einsum('nkil,nkl->nki', A, delta)
	V = sigma_prior[None, :, :, :] - np.einsum('nkil,klj->nkij', A, sigma_prior)
	outer_term = np.einsum('nki,nkj->nkij', m_minus_mu, m_minus_mu)
	T = V + outer_term
	weighted = np.einsum('nk,nkij->kij', dataset.gamma, T)
	sigma_candidate = weighted / Nk_safe[:, None, None]
	sigma_pd = _ensure_positive_definite(sigma_candidate, min_eig=min_eig, dtype=dataset.pos.dtype)
	fallback = sigma_prior
	mask = (Nk > min_eig)[:, None, None]
	return np.where(mask, sigma_pd, fallback)

def _initialize_parameters(dataset, min_eig=1e-6):
	K = dataset.K or (dataset.mu.shape[0] if dataset.mu is not None else None)
	if K is None:
		raise ValueError("dataset.K must be set before initialization.")
	if dataset.mu is None:
		kmeans = KMeans(n_clusters=K, n_init=10)
		sample_weight = 1/determinant_2x2_matrices(dataset.covs)
		labels = kmeans.fit_predict(dataset.pos, sample_weight=sample_weight)
		dataset.mu = kmeans.cluster_centers_.astype(dataset.pos.dtype, copy=False)
		dataset.labels = labels
	else:
		dist = norm_2d(dataset.pos[:, None, :] - dataset.mu[None, :, :], axis=2)
		labels = np.argmin(dist, axis=1)
	gamma = np.zeros((dataset.pos.shape[0], K), dtype=dataset.pos.dtype)
	gamma[np.arange(dataset.pos.shape[0]), labels] = 1.0
	dataset.gamma = gamma
	if dataset.pi is None:
		Nk = dataset.gamma.sum(axis=0)
		dataset.pi = np.clip(Nk / max(dataset.pos.shape[0], 1), 1e-12, None)
		dataset.pi /= dataset.pi.sum()
	if dataset.sigma is None:
		eye = np.eye(dataset.pos.shape[1], dtype=dataset.pos.dtype)
		dataset.sigma = np.broadcast_to(eye, (K, eye.shape[0], eye.shape[1])).copy()
	inv_eff, _ = E_step(dataset)
	dataset.sigma = compute_sigma(dataset, inv_eff, previous_sigma=dataset.sigma, min_eig=min_eig)
	dataset.labels = np.argmax(dataset.gamma, axis=1)

def gmm(dataset, max_iter=100, tol=1e-6, min_eig=1e-6, K=None):
	if K is not None:
		dataset.set_k(K)
	if dataset.mu is None or dataset.sigma is None or dataset.pi is None or dataset.gamma is None:
		_initialize_parameters(dataset, min_eig=min_eig)
	prev_ll = -np.inf
	for _ in range(max_iter):
		inv_eff, ll = E_step(dataset)
		previous_sigma = dataset.sigma.copy()
		M_step(dataset, inv_eff, previous_sigma=previous_sigma, min_eig=min_eig)
		if np.abs(ll - prev_ll) < tol * (np.abs(prev_ll) + 1e-12):
			break
		prev_ll = ll
	dataset.labels = np.argmax(dataset.gamma, axis=1)
	return -ll