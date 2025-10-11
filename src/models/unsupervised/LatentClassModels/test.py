import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, cdist
import pickle
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Hàm tính Dunn Index
def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    k = len(unique_clusters)
    intra_dists = []
    inter_dists = []

    for c in unique_clusters:
        cluster_points = X[labels == c]
        if len(cluster_points) > 1:
            intra_dists.append(np.max(pdist(cluster_points)))
        else:
            intra_dists.append(0)

    for i in range(k):
        for j in range(i+1, k):
            ci = X[labels == unique_clusters[i]]
            cj = X[labels == unique_clusters[j]]
            inter_dists.append(np.min(cdist(ci, cj)))

    max_intra = np.max(intra_dists)
    min_inter = np.min(inter_dists)
    return (min_inter / max_intra) if max_intra > 0 else 0

# LCM
class LatentClassModel:
    def __init__(self, n_classes, max_iter=100, tol=1e-6, random_state=None, eps=1e-8):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.random_state = np.random.RandomState(random_state)
        self.log_likelihood_history_ = []

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]

        self.pi_ = np.ones(self.n_classes) / self.n_classes
        self.theta_ = [
            self.random_state.dirichlet(np.ones(len(cats)), size=self.n_classes)
            for cats in self.categories_
        ]

        prev_log_likelihood = -np.inf
        self.log_likelihood_history_ = []

        for iteration in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            log_likelihood = self._log_likelihood(X)
            self.log_likelihood_history_.append(log_likelihood)

            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
            prev_log_likelihood = log_likelihood

        self.log_likelihood_ = log_likelihood
        return self

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_classes))
        for i in range(n_samples):
            for k in range(self.n_classes):
                prob = self.pi_[k]
                for j in range(self.n_features_):
                    cats = self.categories_[j]
                    idx = np.where(cats == X[i, j])[0]
                    prob *= self.theta_[j][k, idx[0]] if len(idx) > 0 else self.eps
                resp[i, k] = max(prob, self.eps)
        resp_sum = resp.sum(axis=1, keepdims=True) + self.eps
        resp /= resp_sum
        return resp

    def _m_step(self, X, resp):
        n_samples = X.shape[0]
        Nk = resp.sum(axis=0) + self.eps
        self.pi_ = Nk / n_samples
        for j in range(self.n_features_):
            cats = self.categories_[j]
            n_cat = len(cats)
            probs = np.zeros((self.n_classes, n_cat))
            for k in range(self.n_classes):
                for c, cat in enumerate(cats):
                    mask = (X[:, j] == cat)
                    probs[k, c] = np.sum(resp[mask, k])
                probs[k] = np.maximum(probs[k], self.eps)
                probs[k] /= probs[k].sum()
            self.theta_[j] = probs

    def _log_likelihood(self, X):
        n_samples = X.shape[0]
        log_likelihood = 0
        for i in range(n_samples):
            total = 0
            for k in range(self.n_classes):
                prob = self.pi_[k]
                for j in range(self.n_features_):
                    cats = self.categories_[j]
                    idx = np.where(cats == X[i, j])[0]
                    prob *= self.theta_[j][k, idx[0]] if len(idx) > 0 else self.eps
                total += prob
            log_likelihood += np.log(total + self.eps)
        return log_likelihood

# Run model, lưu, load
class LatentClassPipeline:
    def __init__(self, file_path, min_k=2, max_k=8, random_state=42, model_dir="LatentClassModels"):
        self.file_path = file_path
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, f"latent_class_model_{min_k}-{max_k}.pkl")
        self.best_model = None
        self.result_df = None
        self.rfm_df = None

        os.makedirs(model_dir, exist_ok=True)

    def load_or_train(self):
        pre = RFMPreprocessor(self.file_path)
        result = pre.process()
        X_df = result[0] if isinstance(result, tuple) else result
        print(f"Ma trận đặc trưng sau tiền xử lý: {X_df.shape}")

        # Nếu X_df không có tên cột, gán tên mặc định Recency/Frequency/Monetary
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=["Recency", "Frequency", "Monetary"])
        elif all(isinstance(c, int) for c in X_df.columns):
            X_df.columns = ["Recency", "Frequency", "Monetary"]

        X = np.array(X_df, dtype=str)

        # ==== Load model nếu có ====
        if os.path.exists(self.model_file):
            print(f"Loading model đã train từ {self.model_file} ...")
            with open(self.model_file, "rb") as f:
                data = pickle.load(f)

            if isinstance(data, dict) and "best_model" in data:
                self.best_model = data["best_model"]
                self.result_df = data["results"]
            else:
                self.best_model = data
                self.result_df = pd.DataFrame([[self.best_model.n_classes, self.best_model.log_likelihood_, 0]],
                                              columns=["n_classes", "log_likelihood", "BIC"])

            print(f"Loaded model với K={self.best_model.n_classes}")

        # ==== Train mới nếu chưa có ====
        else:
            print("Chưa có model, bắt đầu train...")
            results = []
            best_bic = np.inf

            for k in range(self.min_k, self.max_k + 1):
                print(f"\nĐang chạy mô hình {k} lớp (class)...")
                model = LatentClassModel(n_classes=k, random_state=self.random_state)
                model.fit(X)
                log_likelihood = model.log_likelihood_
                n_params = (k - 1) + sum((len(np.unique(X[:, j])) - 1) * k for j in range(X.shape[1]))
                bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
                print(f"   → Log-likelihood = {log_likelihood:.4f}, BIC = {bic:.4f}")
                results.append((k, log_likelihood, bic))

                if bic < best_bic:
                    best_bic = bic
                    self.best_model = model

            self.result_df = pd.DataFrame(results, columns=["n_classes", "log_likelihood", "BIC"])
            with open(self.model_file, "wb") as f:
                pickle.dump({"best_model": self.best_model, "results": self.result_df}, f)

            print(f"Saved trained model và kết quả BIC vào {self.model_file}")

        return X_df, X

    # ========================
    # EVALUATE
    # ========================
    def evaluate(self, X_df, X):
        resp = self.best_model._e_step(X)
        hard_labels = resp.argmax(axis=1)
        unique, counts = np.unique(hard_labels, return_counts=True)

        X_num = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            le = LabelEncoder()
            X_num[:, j] = le.fit_transform(X[:, j])

        if len(np.unique(hard_labels)) < 2:
            sil_score, dbi_score, chi_score, dunn_score = np.nan, np.nan, np.nan, np.nan
        else:
            sil_score = silhouette_score(X_num, hard_labels)
            dbi_score = davies_bouldin_score(X_num, hard_labels)
            chi_score = calinski_harabasz_score(X_num, hard_labels)
            dunn_score = dunn_index(X_num, hard_labels)

        scores = np.array([sil_score, dbi_score, chi_score, dunn_score])

        log_likelihood = np.array(self.best_model.log_likelihood_)
        log_curve = np.array(self.best_model.log_likelihood_history_)

        labels = np.array(hard_labels)
        
        metrics = {
            "log_likelihood": log_likelihood,
            "curve": log_curve,
            "curve_label": "Log-Likelihood"
        }

        centers_df = self.compute_rfm_means(X_df, hard_labels, unique)

        print(scores)
        print(metrics)
        print(labels)

        return {
            "labels": hard_labels,
            "centers": centers_df,
            "metrics": metrics
        }, scores

    # Biểu đồ
    def plot_results(self, unique, counts):
        plt.figure(figsize=(6, 4))
        plt.plot(self.result_df['n_classes'], self.result_df['BIC'], marker='o')
        plt.xlabel("K")
        plt.ylabel("BIC")
        plt.title("BIC theo số lớp K")
        plt.xticks(self.result_df['n_classes'])
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.bar([f"Cluster {i+1}" for i in unique], counts, color='skyblue')
        plt.xlabel("Cluster")
        plt.ylabel("Số mẫu")
        plt.title(f"Số mẫu phân vào từng cluster (K={self.best_model.n_classes})")
        plt.show()
    
    # Tính trung bình RFM
    def compute_rfm_means(self, X_df, hard_labels, unique):
        cluster_stats = []
        for cluster_id in unique:
            X_df = pd.DataFrame(X_df, columns=["Recency", "Frequency", "Monetary"]) if isinstance(X_df, np.ndarray) else X_df
            cluster_points = X_df[hard_labels == cluster_id]

            cluster_stats.append({
                "Cluster": cluster_id + 1,
                "Recency": cluster_points["Recency"].mean(),
                "Frequency": cluster_points["Frequency"].mean(),
                "Monetary": cluster_points["Monetary"].mean()
            })
        centers_df = pd.DataFrame(cluster_stats)
        return centers_df

if __name__ == "__main__":
    pipeline = LatentClassPipeline("data/raw/noLabel/Online Retail.xlsx", min_k=2, max_k=8)
    X_df, X = pipeline.load_or_train()
    pipeline.evaluate(X_df, X)

