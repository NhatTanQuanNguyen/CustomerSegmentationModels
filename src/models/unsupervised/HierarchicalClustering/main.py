# src/models/unsupervised/HierarchicalClustering/main.py
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

__all__ = ["HierarchicalWard"]

def _zscore(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd

def _sse_from_labels(X, labels):
    sse = 0.0
    for g in np.unique(labels):
        Xi = X[labels == g]
        if Xi.size == 0:
            continue
        mu = Xi.mean(axis=0, keepdims=True)
        diff = Xi - mu
        sse += float(np.sum(diff * diff))
    return sse

class HierarchicalWard:
    @staticmethod
    def fit_k(X, k, standardize=True):
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=float)
        Z = linkage(X_, method="ward", metric="euclidean")
        labels = fcluster(Z, t=k, criterion="maxclust")
        sse = _sse_from_labels(X_, labels)
        return {"labels": labels, "metrics": {"sse": sse, "k_used": int(k)}, "linkage": Z}

    @staticmethod
    def fit_distance(X, height, standardize=True):
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=float)
        Z = linkage(X_, method="ward", metric="euclidean")
        labels = fcluster(Z, t=height, criterion="distance")
        sse = _sse_from_labels(X_, labels)
        return {"labels": labels, "metrics": {"sse": sse, "cut_height": float(height)}, "linkage": Z}

    @staticmethod
    def sse_vs_k(X, k_min=2, k_max=11, standardize=True):
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=float)
        Z = linkage(X_, method="ward", metric="euclidean")
        ks, sses = [], []
        for k in range(k_min, k_max):
            labels = fcluster(Z, t=k, criterion="maxclust")
            sses.append(_sse_from_labels(X_, labels))
            ks.append(k)
        return ks, sses

    @staticmethod
    def dendrogram_coords(X, standardize=True, truncate_mode=None, p=30):
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=float)
        Z = linkage(X_, method="ward", metric="euclidean")
        d = dendrogram(Z, no_plot=True, color_threshold=None,
                       truncate_mode=truncate_mode, p=p)
        d["linkage"] = Z
        return d

    @staticmethod
    def suggest_k_by_jump(X, standardize=True, top=1):
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=float)
        Z = linkage(X_, method="ward", metric="euclidean")  # (N-1, 4)
        heights = Z[:, 2]                                   # merge distances (asc)
        N = Z.shape[0] + 1
        if len(heights) < 2:
            cut_h = float(heights[-1]) if len(heights) else 0.0
            return max(1, N), cut_h, Z, np.array([])
        diffs = np.diff(heights)                            # jumps
        idx = np.argsort(diffs)[::-1][:top]                 # largest jumps
        j = int(idx[0])                                     # jump giữa merge j và j+1
        k_suggest = N - (j + 1)                             # clusters trước merge j+1
        cut_height = heights[j] + 1e-9                      # cắt ngay dưới merge tiếp theo
        return int(k_suggest), float(cut_height), Z, np.sort(diffs)[::-1]
