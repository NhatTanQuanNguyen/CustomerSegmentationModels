# src/models/unsupervised/GaussianMixtureModel/main.py
import numpy as np
__all__ = ["ManualGMM"]
class ManualGMM:
    def __init__(self, n_components=3, max_iters=100, tol=1e-6, random_state=42):
        self.K = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihoods_ = []
        self.log_likelihood_ = None
        self.aic_ = None
        self.bic_ = None
        self.n_params_ = None
    @staticmethod
    def _gaussian_pdf_matrix(X, mean, cov, inv_cov=None, det_cov=None):
        d = X.shape[1]
        if inv_cov is None:
            inv_cov = np.linalg.inv(cov)
        if det_cov is None:
            det_cov = np.linalg.det(cov)
        diff = X - mean
        expo = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        norm = 1.0 / np.sqrt((2.0 * np.pi) ** d * det_cov + 1e-12)
        return norm * np.exp(expo)
    def _initialize(self, X):
        np.random.seed(self.random_state)
        N, D = X.shape
        self.means_ = X[np.random.choice(N, self.K, replace=False)]
        base_cov = np.cov(X.T) + np.eye(D) * 1e-6
        self.covariances_ = np.array([base_cov.copy() for _ in range(self.K)])
        self.weights_ = np.ones(self.K) / self.K
    def _e_step(self, X):
        N, D = X.shape
        pdfs = np.zeros((N, self.K))
        invs, dets = [], []
        for k in range(self.K):
            cov = self.covariances_[k]
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            invs.append(inv); dets.append(det)
            pdfs[:, k] = self._gaussian_pdf_matrix(X, self.means_[k], cov, inv, det)
        weighted = pdfs * self.weights_
        denom = np.sum(weighted, axis=1, keepdims=True) + 1e-300
        gamma = weighted / denom
        ll = np.sum(np.log(denom.squeeze()))
        return gamma, ll
    def _m_step(self, X, gamma):
        N, D = X.shape
        Nk = np.sum(gamma, axis=0) + 1e-12
        self.weights_ = Nk / N
        self.means_ = (gamma.T @ X) / Nk[:, None]
        covs = []
        for k in range(self.K):
            diff = X - self.means_[k]
            cov_k = (diff.T @ (diff * gamma[:, [k]])) / Nk[k]
            cov_k += np.eye(D) * 1e-6
            covs.append(cov_k)
        self.covariances_ = np.array(covs)
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        self._initialize(X)
        prev_ll = -np.inf
        self.log_likelihoods_.clear()
        for _ in range(self.max_iters):
            gamma, ll = self._e_step(X)
            self._m_step(X, gamma)
            self.log_likelihoods_.append(ll)
            if np.abs(ll - prev_ll) <= self.tol:
                break
            prev_ll = ll
        self.log_likelihood_ = self.log_likelihoods_[-1]
        self.n_params_ = (self.K - 1) + self.K * D + self.K * (D * (D + 1) // 2)
        self.aic_ = 2 * self.n_params_ - 2 * self.log_likelihood_
        self.bic_ = self.n_params_ * np.log(N) - 2 * self.log_likelihood_
        return self
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        pdfs = np.zeros((N, self.K))
        for k in range(self.K):
            cov = self.covariances_[k]
            pdfs[:, k] = self._gaussian_pdf_matrix(X, self.means_[k], cov)
        weighted = pdfs * self.weights_
        denom = np.sum(weighted, axis=1, keepdims=True) + 1e-300
        return weighted / denom
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    @staticmethod
    def fit_k(X, k, max_iters=200, tol=1e-6, random_state=42):
        model = ManualGMM(n_components=k, max_iters=max_iters, tol=tol, random_state=random_state)
        model.fit(X)
        labels = model.predict(X)
        metrics = {
            "bic": float(model.bic_),
            "aic": float(model.aic_),
            "log_likelihood": float(model.log_likelihood_),
            "n_params": int(model.n_params_),
        }
        return {"labels": labels, "metrics": metrics, "model": model}
