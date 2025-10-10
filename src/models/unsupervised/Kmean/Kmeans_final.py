import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


# Thuật toán K-Means thủ công 
class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Khởi tạo ngẫu nhiên tâm cụm
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # Gán cụm gần nhất cho từng điểm
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Tính lại tâm cụm
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            # Nếu tâm cụm không thay đổi => dừng
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    @staticmethod
    def cluster(file_path, k_clusters=4):
        """
        Return: pandas.DataFrame [Recency, Frequency, Monetary]
        """
        pre = RFMPreprocessor(file_path)
        X = pre.process()

        kmeans = KMeansNumpy(n_clusters=k_clusters, random_state=0)
        kmeans.fit(X)

        # Gắn nhãn cụm vào RFM gốc
        pre.rfm["Cluster"] = kmeans.labels_

        # Tính trung bình mỗi cụm
        cluster_summary = (
            pre.rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .reset_index()
        )

        np.set_printoptions(precision=6, suppress=True)

        return cluster_summary.values









