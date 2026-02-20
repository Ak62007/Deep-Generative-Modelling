import numpy as np

# Likelihood
def multivariate_normal_pdf(x: np.array, mean: np.array, cov: np.array):
    """Calculates the pdf for the multivariate normal distribution for given mean, cov and x

    Args:
        x (np.array): (N, d) array - N data points, d features
        mean (d, ) or (1, d): mean of the distribution(vector)
        cov (np.array): (d, d) cov matrix of the diatribution
    """
    N, d = x.shape
    det = np.linalg.det(cov)
    
    if det <= 1e-15:
        cov += np.eye(d) * 1e-6
        det = np.linalg.det(cov)
        
    inv_cov = np.linalg.inv(cov)
    
    diff = x - mean.reshape(1, -1)
    
    exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
    
    log_normalization = -0.5 * (d * np.log(2 * np.pi) + np.log(det))
    
    return np.exp(log_normalization + exponent)

def bayes_probs(X: np.array, means: np.array, cov: np.array, prior: float) -> np.array:
    """Gives the bayes_probs for all the sources in a array

    Args:
        X (np.array): data
        means (np.array): means of the sources
        cov (np.array): cov matrix if the sources
        prior (float, optional): uniform. Defaults to prior.

    Returns:
        np.array: bayes_probs for all the sources
    """
    
    likelihoods = np.zeros((len(X), means.shape[0]))
    
    for i in range(means.shape[0]):
        likelihoods[:, i] = multivariate_normal_pdf(
                                x=X,
                                mean=means[i],
                                cov=cov[i],
                            )
        
    weighted_likelihoods = likelihoods * prior
    probs = weighted_likelihoods/(weighted_likelihoods.sum(axis=1, keepdims=True) + 1e-15)
    
    return probs
        
# def compute_log_likelihood(X: np.array, means: np.array, cov: np.array, prior: float) -> float:
#     """Calcutates the log-likelihood of the GMMs how well they are fitting the data.

#     Args:
#         X (np.array): data
#         means (np.array): means of the sources 
#         cov (np.array): covariance matrixes of the sources
#         prior (float): prior

#     Returns:
#         float: total log-likehood
#     """
#     likelihoods = np.zeros((len(X), means.shape[0]))
    
#     for k in range(means.shape[0]):
#         likelihoods[:, k] = multivariate_normal_pdf(x=X, mean=means[k], cov=cov[k])
        
#     weighted_likelihoods = likelihoods * prior
#     total_likelihood = weighted_likelihoods.sum(axis=1)
    
#     log_likelihood = np.log(total_likelihood + 1e-15).sum()
    
#     return log_likelihood

# def higher_dim_expectation_maximization(X: np.array, K: int, iterations: int = 10):
#     """Gives you dataframe of how means and cov mat converge as we apply expectation maximization.

#         X (np.array): data
#         means (np.array): initial means of the assumed sources.
#         cov (np.array): initial cov matrixes of the assumed sources.
#         iterations (int): no of iteration to run EM.
#         prior (float): defaults to uniform
#     """
#     initial_means = X[np.random.randint(low=0, high=len(X)-1, size=K)]

#     data_cov = np.cov(X.T)
#     initial_cov = np.array([data_cov * np.random.uniform(low=0, high=2, size=1) + np.eye(X.shape[1]) * 0.1 for _ in range(K)])
#     initial_prior = np.ones(K) / K  
    
#     current_means = initial_means.copy()
#     current_cov = initial_cov.copy()
#     current_prior = initial_prior.copy()  
    
#     log_likelihood = []
    
#     for i in range(iterations):
        
#         ll = compute_log_likelihood(X, current_means, current_cov, current_prior)
#         log_likelihood.append(ll)
        
#         # E-step
#         bi_s = bayes_probs(X=X, means=current_means, cov=current_cov, prior=current_prior)
        
#         # M-step
#         new_means = np.zeros_like(current_means)
#         new_cov = np.zeros_like(current_cov)
#         new_prior = np.zeros(K)  
        
#         for source in range(bi_s.shape[1]):
#             weight = bi_s[:, source]
            
#             # Update prior (mixing coefficient)
#             new_prior[source] = weight.mean()  
            
#             # Update mean
#             new_means[source, :] = (weight.reshape(-1, 1) * X).sum(axis=0)/(weight.reshape(-1, 1)).sum(axis=0)
            
#             # Update covariance
#             diff = X - new_means[source, :]
#             num = (diff.T * weight) @ diff
#             den = weight.sum()
            
#             new_cov[source] = (num / den) + np.eye(X.shape[1]) * 1e-6
    
#         current_means = new_means
#         current_cov = new_cov
#         current_prior = new_prior
        
#     return current_means, current_cov, current_prior, log_likelihood


import numpy as np
from scipy.special import logsumexp
from scipy.linalg import solve_triangular
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def log_gaussian_pdf(X, mu, cov, reg=1e-5):
    """Computes log(N(X | mu, cov)) safely."""
    d = X.shape[1]
    
    sigma = cov + np.eye(d) * reg
    L = np.linalg.cholesky(sigma) 
    
    log_det = 2 * np.sum(np.log(np.diag(L)))
    
    diff = X - mu

    y = solve_triangular(L, diff.T, lower=True)
    mahalanobis = np.sum(np.square(y), axis=0)
    
    return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

def fit_gmm(X_raw, K, iterations=20):

    pca = PCA(n_components=0.95, svd_solver='full')
    X = pca.fit_transform(X_raw)
    N, D = X.shape
    
    # Step 2: K-Means Initialization
    kmeans = KMeans(n_clusters=K, n_init=K).fit(X)
    means = kmeans.cluster_centers_
    priors = np.ones(K) / K
    # Initialize covs as the global variance
    covs = np.array([np.cov(X.T) for _ in range(K)])
    
    for i in range(iterations):
        log_probs = np.zeros((N, K))
        for k in range(K):
            log_probs[:, k] = log_gaussian_pdf(X, means[k], covs[k]) + np.log(priors[k])
        
        # Log-Sum-Exp trick to get normalization factor
        log_totals = logsumexp(log_probs, axis=1)
        resp = np.exp(log_probs - log_totals[:, np.newaxis])
        
        # M-Step 
        weights_sum = resp.sum(axis=0)
        priors = weights_sum / N
        
        for k in range(K):
            # Update Mean
            means[k] = (resp[:, k] @ X) / weights_sum[k]
            
            # Update Covariance
            diff = X - means[k]
            # Weighted covariance
            covs[k] = (diff.T @ (diff * resp[:, k][:, np.newaxis])) / weights_sum[k]
            covs[k] += np.eye(D) * 1e-5 # Regularization
            
        print(f"Iteration {i+1} complete. Log-Likelihood: {np.sum(log_totals)}")
        
    return means, covs, priors, pca

# Sampling
def generate_digit(gmm_component_idx, pca_model, gmm_means, gmm_covs):
    # 1. Sample from the chosen Gaussian in the REDUCED space
    mean = gmm_means[gmm_component_idx]
    cov = gmm_covs[gmm_component_idx]
    sample_latent = np.random.multivariate_normal(mean, cov)
    
    # 2. Project back to 784D
    sample_pixel_flat = pca_model.inverse_transform(sample_latent.reshape(1, -1))
    
    # 3. Reshape for visualization
    return sample_pixel_flat.reshape(28, 28)

