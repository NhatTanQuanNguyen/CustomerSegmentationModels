import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.utils.export_compare_results import export_compare_results


# ====== Hàm tính Dunn Index ======
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)
    intra_dists, inter_dists = [], []
    for i in unique_clusters:
        ci = np.where(labels == i)[0]
        intra_dists.append(np.max(distances[np.ix_(ci, ci)]) if len(ci) > 1 else 0)
        for j in unique_clusters:
            if i < j:
                cj = np.where(labels == j)[0]
                inter_dists.append(np.min(distances[np.ix_(ci, cj)]))
    if len(inter_dists) == 0:
        return 0
    return np.min(inter_dists) / np.max(intra_dists)


# ====== MAIN ======
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

    # === Tính các chỉ số ===
    results = []
    for lbls, name, ll in [
        (labels_manual, "Manual GMM", log_ll_manual),
        (labels_lib, "Library GMM", log_ll_lib),
    ]:
        sil = silhouette_score(X, lbls)
        dbi = davies_bouldin_score(X, lbls)
        chi = calinski_harabasz_score(X, lbls)
        dunn = dunn_index(X, lbls)
        results.append({
            "Phiên bản": name,
            "Log-Likelihood": ll,
            "Silhouette": sil,
            "DBI": dbi,
            "CHI": chi,
            "Dunn": dunn
        })

    # === So sánh nhãn (gióng cụm) ===
    cm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cm[i, j] = np.sum((labels_lib == i) & (labels_manual == j))
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = sum(cm[i, j] for i, j in zip(row_ind, col_ind))
    match_ratio = matched / len(X)
    results.append({"Phiên bản": "So sánh nhãn", "MatchRate": match_ratio})

    # === Xuất ảnh ===
    export_compare_results(
        compare_data=results,
        title=f"So sánh Gaussian Mixture (K={k}) - Code tay vs Thư viện",
        save_path=f"results/GMM_compare_K{k}.png"
    )

    print(f"✅ Ảnh kết quả đã lưu tại: results/GMM_compare_K{k}.png")
