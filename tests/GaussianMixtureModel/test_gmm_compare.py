import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM


if __name__ == "__main__":
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, _, _ = pre.process()

    k = 3
    print(f"🚀 Test Gaussian Mixture (K={k}) trên {X.shape[0]} khách hàng")

    # === Code tay ===
    manual = ManualGMM.fit_k(X, k=k, random_state=42)
    labels_manual = manual["labels"]
    log_ll_manual = manual["metrics"]["log_likelihood"]

    # === Thư viện ===
    lib = GaussianMixture(n_components=k, random_state=42, max_iter=100)
    lib.fit(X)
    labels_lib = lib.predict(X)
    log_ll_lib = lib.score(X) * len(X)

    # === Log-likelihood ===
    print("\n📊 So sánh Log-Likelihood:")
    print(f"Manual GMM: {log_ll_manual:.4f}")
    print(f"Library GMM: {log_ll_lib:.4f}")

    # === So nhãn (gióng cụm) ===
    cm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cm[i, j] = np.sum((labels_lib == i) & (labels_manual == j))
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = sum(cm[i, j] for i, j in zip(row_ind, col_ind))
    match_ratio = matched / len(X)
    print(f"\n🎯 Tỷ lệ trùng nhãn (đã gióng cụm): {match_ratio:.4f}")

    # === So trung tâm ===
    print("\nMeans (Manual):")
    print(np.round(manual["centers"], 4))
    print("\nMeans (Library):")
    print(np.round(lib.means_, 4))
