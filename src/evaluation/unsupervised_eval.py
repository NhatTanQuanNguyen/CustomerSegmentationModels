import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist


class UnsupervisedEvaluator:
    """
    Bộ công cụ đánh giá mô hình không giám sát (KMeans, GMM, FCM,...)
    """

    @staticmethod
    def internal_metrics(X, labels):
        """
        Tính các chỉ số nội tại của mô hình:
        - Silhouette Score
        - Davies-Bouldin Index
        - Calinski-Harabasz Index
        - Dunn Index
        """
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

        # === Dunn Index ===
        cluster_means = [X[labels == i].mean(axis=0) for i in unique_labels]
        dist_matrix = cdist(cluster_means, cluster_means)
        inter_cluster = np.min(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        intra_cluster = np.max([np.max(cdist(X[labels == i], X[labels == i]))
                                for i in unique_labels])
        dunn = inter_cluster / (intra_cluster + 1e-12)

        return {
            "silhouette": silhouette,
            "davies_bouldin": dbi,
            "calinski_harabasz": chi,
            "dunn": dunn
        }

    @staticmethod
    def compare_models(X, labels_a, labels_b, name_a="KMeans", name_b="GMM"):
        """
        So sánh hai mô hình không giám sát qua:
        - Độ tương đồng nhãn
        - Các chỉ số nội tại
        """
        print(f"\n===== 🔎 COMPARISON: {name_a} vs {name_b} =====")

        # 1️⃣ Tính độ tương đồng giữa hai nhãn
        match_ratio = (labels_a == labels_b).mean()
        print(f"🔸 Label Match Ratio: {match_ratio:.4f}")

        # 2️⃣ Tính các chỉ số nội tại
        print("\n📊 Internal Evaluation Metrics:")
        metrics_a = UnsupervisedEvaluator.internal_metrics(X, labels_a)
        metrics_b = UnsupervisedEvaluator.internal_metrics(X, labels_b)

        def fmt(m):
            return {k: round(v, 4) for k, v in m.items()}

        print(f"{name_a}:", fmt(metrics_a))
        print(f"{name_b}:", fmt(metrics_b))

        return {
            "match_ratio": match_ratio,
            name_a: metrics_a,
            name_b: metrics_b
        }
