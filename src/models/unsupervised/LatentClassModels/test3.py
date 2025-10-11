import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, cdist
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

warnings.filterwarnings("ignore", category=RuntimeWarning)


def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return np.array(0.0)

    intra_dists = [np.max(pdist(X[labels == c])) if np.sum(labels == c) > 1 else 0 for c in unique_clusters]
    inter_dists = [
        np.min(cdist(X[labels == unique_clusters[i]], X[labels == unique_clusters[j]]))
        for i in range(len(unique_clusters)) for j in range(i + 1, len(unique_clusters))
    ]

    return np.array(np.min(inter_dists) / np.max(intra_dists) if np.max(intra_dists) > 0 else 0.0)


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
        self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]
        self.idx_maps_ = [{cat: i for i, cat in enumerate(cats)} for cats in self.categories_]

        self.pi_ = np.full(self.n_classes, 1 / self.n_classes)
        self.theta_ = [self.random_state.dirichlet(np.ones(len(cats)), size=self.n_classes)
                       for cats in self.categories_]

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._log_likelihood(X)
            self.log_likelihood_history_.append(ll)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = ll
        return self

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.tile(self.pi_, (n_samples, 1))
        for j, idx_map in enumerate(self.idx_maps_):
            X_col_idx = np.vectorize(idx_map.get)(X[:, j])
            for k in range(self.n_classes):
                resp[:, k] *= np.maximum(self.theta_[j][k, X_col_idx], self.eps)

        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X, resp):
        Nk = resp.sum(axis=0) + self.eps
        self.pi_ = Nk / Nk.sum()
        for j, cats in enumerate(self.categories_):
            probs = np.zeros((self.n_classes, len(cats)))
            for k in range(self.n_classes):
                for c, cat in enumerate(cats):
                    probs[k, c] = np.sum(resp[X[:, j] == cat, k])
                probs[k] = np.maximum(probs[k], self.eps)
                probs[k] /= probs[k].sum()
            self.theta_[j] = probs

    def _log_likelihood(self, X):
        total_probs = np.zeros(X.shape[0])
        for k in range(self.n_classes):
            prob = np.full(X.shape[0], self.pi_[k])
            for j, idx_map in enumerate(self.idx_maps_):
                X_col_idx = np.vectorize(idx_map.get)(X[:, j])
                prob *= self.theta_[j][k, X_col_idx]
            total_probs += prob
        return np.sum(np.log(np.nan_to_num(total_probs, nan=self.eps) + self.eps))


class LatentClassPipeline:
    def __init__(self, file_path, min_k=2, max_k=2, random_state=42):
        self.file_path = file_path
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state
        self.best_model = None
        self.result_df = None

    def load_or_train(self):
        X_df = RFMPreprocessor(self.file_path).process()
        if isinstance(X_df, tuple):
            X_df = X_df[0]
        X_df = pd.DataFrame(X_df, columns=["Recency", "Frequency", "Monetary"])
        X = np.array(X_df, dtype=str)
        print(f"→ Ma trận đặc trưng: {X_df.shape}")

        results, best_bic, best_model = [], np.inf, None
        for k in range(self.min_k, self.max_k + 1):
            print(f"Đang chạy mô hình {k} lớp...")
            model = LatentClassModel(n_classes=k, random_state=self.random_state).fit(X)
            ll = model.log_likelihood_
            n_params = (k - 1) + sum((len(np.unique(X[:, j])) - 1) * k for j in range(X.shape[1]))
            bic = -2 * ll + n_params * np.log(X.shape[0])
            results.append((k, ll, bic))
            if bic < best_bic:
                best_bic, best_model = bic, model

        self.best_model = best_model
        self.result_df = pd.DataFrame(results, columns=["n_classes", "log_likelihood", "BIC"])
        print(f"→ Mô hình tốt nhất: K={best_model.n_classes}, BIC={best_bic:.4f}")
        return X_df, X

    def evaluate(self, X_df, X):
        resp = self.best_model._e_step(X)
        hard_labels = resp.argmax(axis=1)
        unique = np.unique(hard_labels)

        X_num = np.stack([LabelEncoder().fit_transform(X[:, j]) for j in range(X.shape[1])], axis=1)
        scores = np.full(4, np.nan) if len(unique) < 2 else np.array([
            silhouette_score(X_num, hard_labels),
            davies_bouldin_score(X_num, hard_labels),
            calinski_harabasz_score(X_num, hard_labels),
            dunn_index(X_num, hard_labels)
        ])

        metrics = {
            "log_likelihood": np.array(self.best_model.log_likelihood_),
            "curve": np.array(self.best_model.log_likelihood_history_),
            "curve_label": "Log-Likelihood"
        }

        centers_df = self.compute_rfm_means(X_df, hard_labels)
        return {"labels": hard_labels, "centers": centers_df, "metrics": metrics}, scores

    def compute_rfm_means(self, X_df, hard_labels):
        means = X_df.groupby(hard_labels).mean(numeric_only=True).reset_index(drop=True)
        means.index = np.arange(1, len(means) + 1)
        means.insert(0, "Cluster", means.index)
        return means


if __name__ == "__main__":
    pipeline = LatentClassPipeline("data/raw/noLabel/Online Retail.xlsx", min_k=2, max_k=2)
    X_df, X = pipeline.load_or_train()
    result, scores = pipeline.evaluate(X_df, X)
