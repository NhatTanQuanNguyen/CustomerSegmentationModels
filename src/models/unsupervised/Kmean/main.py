import numpy as np

class KMeansNumpy:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self

    @staticmethod
    def fit_k(X, k, random_state=42):
        """Huấn luyện KMeans với K cụ thể"""
        model = KMeansNumpy(n_clusters=k, random_state=random_state)
        model.fit(X)
        distances = np.linalg.norm(X - model.centroids[model.labels_], axis=1)
        sse = np.sum(distances ** 2)

        metrics = {"curve": [sse], "curve_label": "SSE"}
        return {
            "labels": model.labels_,
            "centers": model.centroids,
            "metrics": metrics
        }
