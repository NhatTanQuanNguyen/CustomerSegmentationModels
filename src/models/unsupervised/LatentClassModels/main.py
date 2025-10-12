# src/models/unsupervised/LatentClass/main.py
import numpy as np
import pandas as pd

__all__ = ["LatentClass"]

def _discretize_rfm_from_X(X_numeric, q=5):
    df = pd.DataFrame(X_numeric, columns=["Recency", "Frequency", "Monetary"])
    out = df.copy()
    for col in df.columns:
        labels = [f"{col}_{i}" for i in range(1, q + 1)]
        try:
            # Tránh lỗi trùng biên (duplicates) của qcut
            out[col] = pd.qcut(
                df[col].rank(method="first"),
                q=q,
                labels=labels,
                duplicates="drop"
            )
        except Exception:
            # Fallback: chia đều theo khoảng (cut) với số bins hợp lệ
            uniq = int(df[col].nunique())
            bins = max(2, min(q, uniq))
            labels2 = [f"{col}_{i}" for i in range(1, bins + 1)]
            out[col] = pd.cut(df[col], bins=bins, labels=labels2, include_lowest=True)
    return out.astype(str).values

# ---------- Core LCA ----------
class _LatentClassModel:
    def __init__(self, n_classes, max_iter=100, tol=1e-6, random_state=None, eps=1e-9):
        self.K = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.rng = np.random.default_rng(random_state)
        self.log_likelihood_history_ = []

    def _prepare_categories(self, X_cat):
        self.categories_ = [np.unique(X_cat[:, j]) for j in range(X_cat.shape[1])]
        self.pi_ = np.full(self.K, 1.0 / self.K)
        self.theta_ = [self.rng.dirichlet(np.ones(len(cats)), size=self.K) for cats in self.categories_]

    def _e_step(self, X_cat):
        N = X_cat.shape[0]
        log_resp = np.tile(np.log(self.pi_ + self.eps), (N, 1))  # [N,K]
        for j, cats in enumerate(self.categories_):
            idx = {c: i for i, c in enumerate(cats)}
            for k in range(self.K):
                p = np.array([self.theta_[j][k, idx.get(x, -1)] if x in idx else self.eps for x in X_cat[:, j]])
                log_resp[:, k] += np.log(p + self.eps)
        m = np.max(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - m)
        resp /= resp.sum(axis=1, keepdims=True) + self.eps
        return resp

    def _m_step(self, X_cat, resp):
        Nk = resp.sum(axis=0) + self.eps
        self.pi_ = Nk / (Nk.sum() + self.eps)
        for j, cats in enumerate(self.categories_):
            Cj = len(cats)
            probs = np.zeros((self.K, Cj))
            for k in range(self.K):
                for c, cat in enumerate(cats):
                    probs[k, c] = np.sum(resp[X_cat[:, j] == cat, k])
                probs[k] = np.maximum(probs[k], self.eps)
                probs[k] /= probs[k].sum()
            self.theta_[j] = probs

    def _log_likelihood(self, X_cat):
        N = X_cat.shape[0]
        logp = np.tile(np.log(self.pi_ + self.eps), (N, 1))
        for j, cats in enumerate(self.categories_):
            idx = {c: i for i, c in enumerate(cats)}
            for k in range(self.K):
                p = np.array([self.theta_[j][k, idx.get(x, -1)] if x in idx else self.eps for x in X_cat[:, j]])
                logp[:, k] += np.log(p + self.eps)
        m = np.max(logp, axis=1, keepdims=True)
        return float(np.sum(m + np.log(np.sum(np.exp(logp - m), axis=1, keepdims=True) + self.eps)))

    def fit(self, X_cat):
        self._prepare_categories(X_cat)
        prev = -np.inf
        for _ in range(self.max_iter):
            resp = self._e_step(X_cat)
            self._m_step(X_cat, resp)
            ll = self._log_likelihood(X_cat)
            self.log_likelihood_history_.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.log_likelihood_ = ll
        return self

    def predict(self, X_cat):
        return self._e_step(X_cat).argmax(axis=1)

# ---------- Public API: y hệt format KMeans/Fuzzy/GMM ----------
class LatentClass:
    @staticmethod
    def fit_k(X, k, q=5, max_iter=200, tol=1e-6, random_state=42):
        X = np.asarray(X, dtype=float)
        X_cat = _discretize_rfm_from_X(X, q=q)
        model = _LatentClassModel(n_classes=k, max_iter=max_iter, tol=tol, random_state=random_state).fit(X_cat)
        labels = model.predict(X_cat)

        df = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"])
        centers = df.groupby(labels).mean(numeric_only=True).values

        metrics = {
            "log_likelihood": model.log_likelihood_,
            "curve": model.log_likelihood_history_,
            "curve_label": "Log-Likelihood",
        }
        return {"labels": labels, "centers": centers, "metrics": metrics}

    @staticmethod
    def bic_for_k(X, k, q=5, max_iter=200, tol=1e-6, random_state=42):
        X = np.asarray(X, dtype=float)
        X_cat = _discretize_rfm_from_X(X, q=q)
        model = _LatentClassModel(n_classes=k, max_iter=max_iter, tol=tol, random_state=random_state).fit(X_cat)
        ll = model.log_likelihood_
        # p = (K-1) + Σ_j K*(C_j-1) với C_j là số hạng mục THỰC TẾ
        Cjs = [len(cats) for cats in model.categories_]
        n_params = (k - 1) + sum(k * (Cj - 1) for Cj in Cjs)
        bic = -2 * ll + n_params * np.log(max(X_cat.shape[0], 1))
        return float(ll), float(bic)
