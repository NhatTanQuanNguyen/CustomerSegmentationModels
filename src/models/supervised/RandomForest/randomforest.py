import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

class RandomForestCluster:
    def __init__(self, n_estimators=200, max_depth=None, test_size=0.3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels_ = None
        self.metrics_ = []
        self.feature_importance_ = None

    # Tính chỉ số Dunn Index
    @staticmethod
    def dunn_index(X, labels):
        unique_clusters = np.unique(labels)
        if len(unique_clusters) < 2:
            return 0.0
        intra_dists = [np.max(pdist(X[labels == c])) if np.sum(labels == c) > 1 else 0 for c in unique_clusters]
        inter_dists = [
            np.min(cdist(X[labels == unique_clusters[i]], X[labels == unique_clusters[j]]))
            for i in range(len(unique_clusters)) for j in range(i + 1, len(unique_clusters))
        ]
        max_intra = np.max(intra_dists)
        min_inter = np.min(inter_dists)
        return float(min_inter / max_intra) if max_intra > 0 else 0.0

    # Tách dữ liệu train/test
    def load_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    # Huấn luyện mô hình RF
    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    # Dự đoán nhãn test
    def predict(self):
        return self.model.predict(self.X_test)

    def evaluate(self):
        labels = np.concatenate([
            self.model.predict(self.X_train),
            self.model.predict(self.X_test)
        ])
        X_all = np.concatenate([self.X_train, self.X_test])

        if len(np.unique(labels)) < 2:
            self.metrics_ = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            self.metrics_ = np.array([
                silhouette_score(X_all, labels),
                davies_bouldin_score(X_all, labels),
                calinski_harabasz_score(X_all, labels),
                self.dunn_index(X_all, labels)
            ])
        self.labels_ = np.array(labels)
        return self.metrics_, self.labels_

    # Trọng số quan trọng của đặc trưng
    def feature_importance(self, feature_cols):
        self.feature_importance_ = np.array([
            (name, float(imp)) for name, imp in zip(feature_cols, self.model.feature_importances_)
        ], dtype=object)
        return self.feature_importance_

    # Pipeline tổng
    def run(self, X, y, feature_cols):
        print(">>> BƯỚC 3: Tách train/test và huấn luyện Random Forest...")
        self.load_data(X, y)
        self.fit()
        print("   -> Huấn luyện hoàn tất.")
        print(">>> BƯỚC 4: Đánh giá mô hình...")
        self.evaluate()
        self.feature_importance(feature_cols)
        print("   -> Đánh giá hoàn tất.")
        return {
            "labels": self.labels_,
            "metrics": self.metrics_,
            "result": self.feature_importance_
        }

class CustomerClusteringPipeline:
    def __init__(self, file_path, kmeans_k=3,
                 rf_n_estimators=300, rf_test_size=0.3,
                 rf_max_depth=None, random_state=42):
        self.file_path = file_path
        self.kmeans_k = kmeans_k
        self.rf_n_estimators = rf_n_estimators
        self.rf_test_size = rf_test_size
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state

        self.X = None
        self.y_kmeans = None
        self.feature_cols = ["Recency", "Frequency", "Monetary"]
        self.rf_output = None

    # Bước 1: Tiền xử lý RFM
    def load_rfm_data(self):
        print(">>> BƯỚC 1: Đang xử lý dữ liệu RFM...")
        preprocessor = RFMPreprocessor(self.file_path)
        X_scaled, rfm_original, _ = preprocessor.process()
        self.X = X_scaled
        print("   -> Hoàn tất xử lý RFM.")
        return self.X

    # Bước 2: Phân cụm bằng K-Means
    def run_kmeans(self):
        print(">>> BƯỚC 2: Đang phân cụm K-Means...")
        kmeans_out = KMeansNumpy.fit_k(self.X, self.kmeans_k, random_state=self.random_state)
        self.y_kmeans = kmeans_out["labels"]
        print(f"   -> K-Means hoàn tất. ({self.kmeans_k} cụm)")
        return kmeans_out

    # Bước 3: Random Forest phân lớp cụm
    def run_random_forest(self):
        rf_cluster = RandomForestCluster(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            test_size=self.rf_test_size,
            random_state=self.random_state
        )
        self.rf_output = rf_cluster.run(self.X, self.y_kmeans, self.feature_cols)
        return self.rf_output

    def run_pipeline(self):
        self.load_rfm_data()
        kmeans_res = self.run_kmeans()
        rf_res = self.run_random_forest()
        return {
            "kmeans": kmeans_res,
            "random_forest": rf_res
        }

if __name__ == "__main__":
    file_path = "data/raw/noLabel/Online Retail.xlsx"

    pipeline = CustomerClusteringPipeline(
        file_path=file_path,
        kmeans_k=4,
        rf_n_estimators=300,
        rf_test_size=0.3
    )

    output = pipeline.run_pipeline()

    print("\n>>> KẾT QUẢ RANDOM FOREST <<<")
    metric_names = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Dunn"]
    for name, val in zip(metric_names, output["random_forest"]["metrics"]):
        print(f"{name:20s}: {val:.4f}")

    print("\n>>> Trọng số đặc trưng (Feature Importance):")
    for name, val in output["random_forest"]["result"]:
        print(f"{name:20s}: {val:.4f}")
