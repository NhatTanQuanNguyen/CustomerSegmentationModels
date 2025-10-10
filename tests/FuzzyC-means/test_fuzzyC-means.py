import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import skfuzzy as fuzz
from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from scipy.spatial.distance import cdist, pdist, squareform

def dunn_index(X, labels):
        distances = squareform(pdist(X))
        unique_clusters = np.unique(labels)
        intra_dists = []
        inter_dists = []

        for i in unique_clusters:
            cluster_i = np.where(labels == i)[0]
            if len(cluster_i) > 1:
                intra_dists.append(np.max(distances[np.ix_(cluster_i, cluster_i)]))
            else:
                intra_dists.append(0)
            for j in unique_clusters:
                if i < j:
                    cluster_j = np.where(labels == j)[0]
                    inter_dists.append(np.min(distances[np.ix_(cluster_i, cluster_j)]))
        
        if len(inter_dists) == 0:
            return 0
        return np.min(inter_dists) / np.max(intra_dists)

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=100, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

        self.centers = None
        self.u = None  # ma trận độ thuộc
        self.hard_labels = None

    def fit(self, X):
        """
        Thực hiện Fuzzy C-Means clustering
        """
        X = np.array(X)
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, c=self.n_clusters, m=self.m, error=self.error,
            maxiter=self.max_iter, init=None, seed=self.random_state
        )

        self.centers = cntr
        self.u = u
        self.hard_labels = np.argmax(u, axis=0)

    def predict(self, X):
        """
        Dự đoán nhãn cụm cho dữ liệu mới
        """
        if self.centers is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit(X) trước.")
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T, self.centers, self.m, error=self.error, maxiter=self.max_iter
        )
        return np.argmax(u, axis=0)

    def evaluate(self, X):
        """
        Đánh giá kết quả phân cụm bằng 5 tiêu chí:
        Silhouette, Davies–Bouldin, Calinski–Harabasz, Dunn, FPC
        """
        X = np.array(X)
        labels = self.hard_labels

        results = {}

        # --- Silhouette Score ---
        if len(set(labels)) > 1:
            results["Silhouette"] = silhouette_score(X, labels)
        else:
            results["Silhouette"] = np.nan

        # --- Davies–Bouldin Index ---
        results["Davies–Bouldin"] = davies_bouldin_score(X, labels)

        # --- Calinski–Harabasz Index ---
        results["Calinski–Harabasz"] = calinski_harabasz_score(X, labels)

        # --- Dunn Index ---
        results["Dunn"] = dunn_index(X, labels)

        # # --- FPC ---
        # results["FPC"] = fuzz.cluster.fpc(self.u)

        return results

    def cluster_summary(self, rfm_df):
        """
        Tính thống kê trung bình mỗi cụm theo Recency, Frequency, Monetary
        """
        if self.hard_labels is None:
            raise ValueError("Bạn cần chạy fit() trước khi xem thống kê cụm.")
        rfm_df = rfm_df.copy()
        rfm_df["Cluster"] = self.hard_labels
        summary = rfm_df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
        return summary


# ================== DEMO SỬ DỤNG ==================

if __name__ == "__main__":

    # --- 1. Tiền xử lý dữ liệu ---
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X = pre.process()

    # --- 2. Chạy Fuzzy C-Means ---
    fcm = FuzzyCMeans(n_clusters=4, random_state=42)
    fcm.fit(X)

    # --- 3. Đánh giá ---
    results = fcm.evaluate(X)
    print("\n📊 ĐÁNH GIÁ CỤM:")
    for k, v in results.items():
        print(f"{k:20}: {v:.4f}")

    # --- 4. Thống kê trung bình mỗi cụm ---
    print("\n📈 THỐNG KÊ TRUNG BÌNH MỖI CỤM:")
    print(fcm.cluster_summary(pre.rfm))
