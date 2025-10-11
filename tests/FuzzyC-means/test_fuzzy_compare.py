import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.FuzzyCMean.main import FuzzyCMeans
from src.utils.export_compare_results import export_compare_results


# ====== Hàm tính chỉ số Dunn ======
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


# ====== Main: chạy và lưu kết quả ======
if __name__ == "__main__":
    # === Tiền xử lý dữ liệu ===
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, _, _ = pre.process()

    k = 4
    print(f"🚀 Test Fuzzy C-Means (K={k}) trên {X.shape[0]} khách hàng")

    # === Code tay ===
    manual = FuzzyCMeans(n_clusters=k, random_state=42)
    manual.fit(X)
    labels_manual = manual.labels_

    # === Thư viện ===
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, c=k, m=2, error=1e-5, maxiter=100, seed=42
    )
    labels_lib = np.argmax(u, axis=0)

    # === Đánh giá ===
    results = []
    for lbls, name in [(labels_manual, "Manual"), (labels_lib, "Library")]:
        sil = silhouette_score(X, lbls)
        dbi = davies_bouldin_score(X, lbls)
        chi = calinski_harabasz_score(X, lbls)
        dunn = dunn_index(X, lbls)
        results.append({
            "Phiên bản": name,
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

    # === Thêm dòng thống kê match rate ===
    results.append({"Phiên bản": "So sánh nhãn", "MatchRate": match_ratio})

    # === Xuất ảnh bảng so sánh ===
    export_compare_results(
        compare_data=results,
        title=f"So sánh Fuzzy C-Means (K={k}) - Code tay vs Thư viện",
        save_path=f"results/FuzzyCMeans_compare_K{k}.png"
    )

    print(f"✅ Ảnh kết quả đã lưu tại: results/FuzzyCMeans_compare_K{k}.png")
