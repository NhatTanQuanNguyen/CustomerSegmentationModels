import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

class UnsupervisedEvaluator:

    def __init__(self, X):
        self.X = X

    def evaluate(self, labels):
        return self._internal_metrics(labels)

    def _internal_metrics(self, labels):
        X = self.X
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return {
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "dunn": np.nan,
            }

        silhouette = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        chi = calinski_harabasz_score(X, labels)
        dunn = self._dunn_index(X, labels)

        return {
            "silhouette": silhouette,
            "davies_bouldin": dbi,
            "calinski_harabasz": chi,
            "dunn": dunn
        }

    @staticmethod
    def _dunn_index(X, labels):
        unique_clusters = np.unique(labels)
        intra_dists = []
        inter_dists = []

        for cluster in unique_clusters:
            cluster_points = X[labels == cluster]
            if len(cluster_points) > 1:
                intra_dists.append(np.max(cdist(cluster_points, cluster_points)))
            else:
                intra_dists.append(0)

        for i in range(len(unique_clusters)):
            for j in range(i + 1, len(unique_clusters)):
                cluster_i = X[labels == unique_clusters[i]]
                cluster_j = X[labels == unique_clusters[j]]
                inter_dists.append(np.min(cdist(cluster_i, cluster_j)))

        if len(inter_dists) == 0 or np.max(intra_dists) == 0:
            return np.nan

        return np.min(inter_dists) / np.max(intra_dists)

    def compare_models(self, labels_a, labels_b, name_a="Model A", name_b="Model B"):
        """So sánh hai mô hình không giám sát qua độ tương đồng nhãn + metrics"""
        match_ratio = (labels_a == labels_b).mean()
        metrics_a = self._internal_metrics(labels_a)
        metrics_b = self._internal_metrics(labels_b)

        return {
            "match_ratio": match_ratio,
            name_a: metrics_a,
            name_b: metrics_b
        }
