# src/experiments/unsupervised/compare_hierarchical.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.HierarchicalClustering.main import HierarchicalWard
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
        # Xử lý trường hợp chỉ có 1 cụm hoặc cụm không có điểm dữ liệu
        return 0 
    # Dunn Index = (Khoảng cách giữa các cụm nhỏ nhất) / (Khoảng cách nội cụm lớn nhất)
    # Nếu np.max(intra_dists) = 0 (chỉ có cụm 1 điểm) thì cần xử lý tránh chia cho 0
    max_intra_dist = np.max(intra_dists)
    if max_intra_dist == 0:
        # Nếu cụm chỉ có 1 điểm hoặc khoảng cách nội cụm max là 0, trả về giá trị lớn để phản ánh cụm rất tốt
        return np.inf 
    return np.min(inter_dists) / max_intra_dist


# ====== MAIN ======
if __name__ == "__main__":
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, _, _ = pre.process()

    k = 4
    print(f"🚀 Test Hierarchical Clustering (Ward, K={k}) trên {X.shape[0]} khách hàng")

    # === Code tay (class HierarchicalWard) ===
    # Đã sửa: Chạy trên dữ liệu đã chuẩn hóa (nội bộ class)
    manual = HierarchicalWard.fit_k(X, k=k, standardize=True)
    labels_manual = manual["labels"]
    sse_manual = manual["metrics"]["sse"]

    # === Thư viện (scipy) ===
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    Z = linkage(X_std, method="ward", metric="euclidean")
    labels_lib = fcluster(Z, t=k, criterion="maxclust")
    
    # === THÊM: Tính SSE cho phiên bản thư viện (trên dữ liệu ĐÃ CHUẨN HÓA X_std) ===
    sse_lib = 0
    for label in np.unique(labels_lib):
        cluster_points = X_std[labels_lib == label]
        if len(cluster_points) > 0:
            # Tính tâm cụm
            centroid = np.mean(cluster_points, axis=0)
            # Tính tổng bình phương khoảng cách từ các điểm đến tâm cụm
            sse_lib += np.sum(np.sum((cluster_points - centroid)**2, axis=1))
    
    
    # ==============================================================================

    # === Tính các chỉ số ===
    results = []
    # Cập nhật giá trị SSE cho phiên bản thư viện
    for lbls, name, sse in [
        (labels_manual, "Manual Ward", sse_manual),
        (labels_lib, "Library Ward", sse_lib), 
    ]:
        # Các chỉ số đánh giá khác vẫn được tính trên dữ liệu GỐC (X)
        sil = silhouette_score(X, lbls)
        dbi = davies_bouldin_score(X, lbls)
        chi = calinski_harabasz_score(X, lbls)
        dunn = dunn_index(X, lbls)
        results.append({
            "Phiên bản": name,
            "SSE": sse,
            "Silhouette": sil,
            "DBI": dbi,
            "CHI": chi,
            "Dunn": dunn
        })

    # === So sánh nhãn giữa 2 phiên bản ===
    cm = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # Nhãn của scipy là 1, 2, 3, 4. Nhãn của manual có thể là 0, 1, 2, 3 
            # Giả định manual cũng trả về nhãn từ 1 trở đi hoặc bạn điều chỉnh theo logic:
            cm[i, j] = np.sum((labels_lib == i+1) & (labels_manual == j+1))
            
    # NOTE: Nếu labels_manual bắt đầu từ 0, dòng trên sẽ bị lỗi.
    # Giả sử cả hai đều trả về nhãn bắt đầu từ 1.
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = sum(cm[i, j] for i, j in zip(row_ind, col_ind))
    match_ratio = matched / len(X)
    results.append({"Phiên bản": "So sánh nhãn", "MatchRate": match_ratio})

    # === Xuất ảnh so sánh ===
    export_compare_results(
        compare_data=results,
        title=f"So sánh Hierarchical Clustering (Ward, K={k}) - Code tay vs Thư viện",
        save_path=f"results/Hierarchical_compare_K{k}.png"
    )

    print(f"✅ Ảnh kết quả đã lưu tại: results/Hierarchical_compare_K{k}.png")