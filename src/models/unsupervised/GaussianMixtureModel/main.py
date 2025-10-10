import numpy as np


class ManualGMM:
    """
    Gaussian Mixture Model thủ công (EM algorithm)
    Trả về:
        - labels
        - means
        - weights
        - log_likelihoods
        - cluster_matrix
    """

    @staticmethod
    def gaussian_pdf(x, mean, cov):
        d = len(x)
        cov_det = np.linalg.det(cov)
        if cov_det <= 0:
            cov_det = 1e-6
        cov_inv = np.linalg.inv(cov + np.eye(d) * 1e-6)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * cov_det)
        diff = x - mean
        return norm_const * np.exp(-0.5 * diff.T @ cov_inv @ diff)

    @staticmethod
    def fit(X, n_components=3, max_iters=100, tol=1e-6, random_state=42):
        np.random.seed(random_state)
        N, D = X.shape
        means = X[np.random.choice(N, n_components, replace=False)]
        covariances = [np.cov(X.T) for _ in range(n_components)]
        weights = np.ones(n_components) / n_components
        log_likelihoods = []

        for iteration in range(max_iters):
            responsibilities = np.zeros((N, n_components))
            for i in range(N):
                for k in range(n_components):
                    responsibilities[i, k] = (
                        weights[k] * ManualGMM.gaussian_pdf(X[i], means[k], covariances[k])
                    )
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            N_k = responsibilities.sum(axis=0)
            for k in range(n_components):
                means[k] = (responsibilities[:, k][:, None] * X).sum(axis=0) / N_k[k]
                diff = X - means[k]
                covariances[k] = ((responsibilities[:, k][:, None] * diff).T @ diff) / N_k[k]
                weights[k] = N_k[k] / N

            log_likelihood = np.sum([
                np.log(np.sum([
                    weights[k] * ManualGMM.gaussian_pdf(X[i], means[k], covariances[k])
                    for k in range(n_components)
                ]) + 1e-12) for i in range(N)
            ])
            log_likelihoods.append(log_likelihood)

            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                print(f"✅ Converged at iteration {iteration}")
                break

        labels = ManualGMM.predict(X, means, covariances, weights)
        return labels, means, covariances, weights, log_likelihoods

    @staticmethod
    def predict(X, means, covariances, weights):
        N, K = X.shape[0], len(means)
        probs = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                probs[i, k] = weights[k] * ManualGMM.gaussian_pdf(X[i], means[k], covariances[k])
        return np.argmax(probs, axis=1)

    @staticmethod
    def run(X, n_components=3, max_iters=100, tol=1e-6, random_state=42):
        labels, means, covs, weights, ll = ManualGMM.fit(X, n_components, max_iters, tol, random_state)
        cluster_matrix = np.column_stack(((labels + 1).astype(int), X))
        return {
            "labels": labels,
            "means": means,
            "weights": weights,
            "log_likelihoods": ll,
            "cluster_matrix": cluster_matrix
        }
