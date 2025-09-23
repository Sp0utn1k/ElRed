import numpy as np


def mahalanobis(x0, inv_covs):
	return np.einsum('nki,nij,nkj->nk', 
					  x0, 
					  inv_covs, 
					  x0)    

def bivariate_gaussian_pdf(dataset):
	mahal = mahalanobis(dataset.x0, dataset.inv_covs)
	return np.exp(-0.5 * mahal) / dataset.gauss_denom[:, None]

def E_step(dataset, normalize=True):
	pdf = bivariate_gaussian_pdf(dataset)
	dataset.gamma = dataset.pi[None, :] * pdf
	if normalize:
		dataset.gamma /= np.sum(dataset.gamma, axis=1, keepdims=True) + 1e-10

def M_step(dataset):
	dataset.pi = np.sum(dataset.gamma, axis=0)
	dataset.pi /= dataset.gamma.shape[0]
	dataset.mu = compute_mu(dataset)

def compute_mu(dataset):
	temp = np.einsum('nk, nij->nkij', dataset.gamma, dataset.inv_covs)
	A = np.sum(temp, axis=0)
	b = np.einsum('nkij, nj->ki', temp, dataset.pos)[:,:,None]
	return np.linalg.solve(A,b)[:,:,0]

def compute_sigma(dataset):
	sigma_inv = np.einsum('nk, nij->kij', dataset.gamma, dataset.inv_covs)
	return np.linalg.inv(sigma_inv)

# def labels2gamma(labels):
# 	N = len(labels)
# 	K = np.max(labels)+1
# 	gamma = np.zeros((N,K))
# 	for label in set(labels):
# 		gamma[labels==label, label] = 1.
# 	return gamma/np.sum(gamma,axis=0,keepdims=True)

def gmm(dataset, max_iter=100):
	for iteration in range(max_iter):
		E_step(dataset)
		M_step(dataset)
		
	compute_sigma(dataset)
	dataset.labels = np.argmax(dataset.gamma, axis=1)