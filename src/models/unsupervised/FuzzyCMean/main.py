import numpy as np

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=100, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.centers = None
        self.U = None
        self.labels_ = None

    def _initialize_U(self, n_samples):
        np.random.seed(self.random_state)
        U = np.random.rand(n_samples, self.n_clusters)
        return U / np.sum(U, axis=1, keepdims=True)

    def _compute_centers(self, X, U):
        um = U ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def _update_U(self, X, centers):
        dist = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        dist = np.fmax(dist, 1e-10)
        exponent = 2 / (self.m - 1)
        denominator = np.sum((dist[:, :, None] / dist[:, None, :]) ** exponent, axis=2)
        return 1 / denominator

    def fit(self, X):
        n_samples = X.shape[0]
        self.U = self._initialize_U(n_samples)
        for _ in range(self.max_iter):
            self.centers = self._compute_centers(X, self.U)
            U_new = self._update_U(X, self.centers)
            if np.linalg.norm(U_new - self.U) < self.error:
                break
            self.U = U_new
        self.labels_ = np.argmax(self.U, axis=1)
        return self

    def fuzzy_partition_coefficient(self):
        return np.sum(self.U ** 2) / self.U.shape[0]

    @staticmethod
    def fit_k(X, k, random_state=42):
        """Huấn luyện FuzzyCMeans với K cụ thể"""
        model = FuzzyCMeans(n_clusters=k, random_state=random_state)
        model.fit(X)
        fpc = model.fuzzy_partition_coefficient()
        metrics = {"fpc": fpc, "curve": [fpc], "curve_label": "FPC"}
        return {
            "labels": model.labels_,
            "centers": model.centers,
            "metrics": metrics
        }
