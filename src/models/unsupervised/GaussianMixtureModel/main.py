import numpy as np

class ManualGMM:
    @staticmethod
    def gaussian_pdf_matrix(X, mean, cov, inv_cov, det_cov):
        d = X.shape[1]
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * det_cov)
        return norm_const * np.exp(exponent)

    @staticmethod
    def fit_k(X, k, max_iters=100, tol=1e-6, random_state=42):
        """Huấn luyện GMM với K cụ thể"""
        np.random.seed(random_state)
        N, D = X.shape
        means = X[np.random.choice(N, k, replace=False)]
        covariances = [np.cov(X.T) + np.eye(D)*1e-6 for _ in range(k)]
        weights = np.ones(k) / k
        log_likelihoods = []

        for _ in range(max_iters):
            inv_covs = [np.linalg.inv(cov) for cov in covariances]
            det_covs = [np.linalg.det(cov) for cov in covariances]

            pdfs = np.zeros((N, k))
            for i in range(k):
                pdfs[:, i] = ManualGMM.gaussian_pdf_matrix(X, means[i], covariances[i], inv_covs[i], det_covs[i])

            responsibilities = pdfs * weights
            responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

            N_k = responsibilities.sum(axis=0)
            for i in range(k):
                means[i] = (responsibilities[:, i][:, None] * X).sum(axis=0) / N_k[i]
                diff = X - means[i]
                covariances[i] = ((responsibilities[:, i][:, None] * diff).T @ diff) / N_k[i]
                covariances[i] += np.eye(D) * 1e-6
                weights[i] = N_k[i] / N

            total_pdf = np.sum(pdfs * weights, axis=1)
            log_likelihood = np.sum(np.log(total_pdf + 1e-12))
            log_likelihoods.append(log_likelihood)

            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                break

        labels = np.argmax(responsibilities, axis=1)
        metrics = {
            "log_likelihood": log_likelihoods[-1],
            "curve": log_likelihoods,
            "curve_label": "Log-Likelihood"
        }
        return {
            "labels": labels,
            "centers": means,
            "metrics": metrics
        }
