# hierarchical_ward.py (Phiên bản O(N^2) Matrix Update - Đã Sửa Lỗi và Bổ sung Methods)

import numpy as np
import math
# Bổ sung dendrogram và linkage
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from scipy.spatial.distance import pdist, squareform 
from typing import Dict, Tuple, List

# Không cần heapq trong phiên bản O(N^2) này

__all__ = ["HierarchicalWardManual", "_zscore", "_sse_from_labels"]

# =============================================
# 🛠️ Hàm Tiện ích và Tính toán Khoảng cách
# =============================================

def _zscore(X: np.ndarray) -> np.ndarray:
    """Thực hiện Chuẩn hóa Z-score, đảm bảo np.float64."""
    X = np.asarray(X, dtype=np.float64) 
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    # Thêm epsilon nhỏ để tránh chia cho 0.
    return (X - mu) / (sd + 1e-12) 

def _sse_from_labels(X: np.ndarray, labels: np.ndarray) -> float:
    """Tính Tổng Bình phương Sai số (SSE) từ nhãn cụm."""
    sse = 0.0
    for g in np.unique(labels):
        Xi = X[labels == g]
        if Xi.size == 0: continue
        mu = Xi.mean(axis=0, keepdims=True)
        diff = Xi - mu
        sse += float(np.sum(diff * diff)) 
    return sse

def _ward_distance_sq_optimized(centroid1, centroid2, n1, n2):
    """
    Tính khoảng cách Ward bình phương (Dist^2) giữa 2 centroid.
    """
    dist_sq_euclidean = np.sum((centroid1 - centroid2) ** 2)
    ward_dist_sq = dist_sq_euclidean * (n1 * n2) / (n1 + n2)
    return np.float64(ward_dist_sq) 

# =============================================
# 🚀 Thuật toán Ward's Linkage O(N^2) - Matrix Update (FINAL)
# =============================================

def _build_linkage_matrix_manual(data: np.ndarray) -> np.ndarray:
    """
    Xây dựng Linkage Matrix (Z) bằng Ward's Linkage, 
    sử dụng Distance Matrix Update (O(N^2) thời gian và bộ nhớ).
    """
    N = len(data)
    linkage_matrix = np.zeros((N - 1, 4), dtype=np.float64)
    
    # 1. Khởi tạo trạng thái
    counts = np.ones(N, dtype=np.float64)
    cluster_labels = np.arange(N, dtype=np.float64)
    
    # 2. Khởi tạo Ward Distance Squared Matrix
    # Công thức Ward Dist^2 ban đầu: D^2 = 0.5 * ||c1-c2||^2
    distances_sq = squareform(pdist(data, metric='sqeuclidean'))
    distances_sq = distances_sq * 0.5 
    np.fill_diagonal(distances_sq, np.inf)

    current_cluster_id = N 
    
    for k in range(N - 1):
        
        # 3. Tìm cặp cụm gần nhất
        min_idx = np.unravel_index(np.argmin(distances_sq), distances_sq.shape)
        i, j = int(min_idx[0]), int(min_idx[1])
        
        # Lấy thông tin cụm
        best_dist_sq = distances_sq[i, j]
        size_i, size_j = counts[i], counts[j]
        new_size = size_i + size_j

        # 4. Ghi lại kết quả gộp (Z matrix)
        dist_ward = np.sqrt(best_dist_sq) 
        label1, label2 = cluster_labels[i], cluster_labels[j]
        linkage_matrix[k] = [min(label1, label2), max(label1, label2), dist_ward, new_size]
        
        # 5. Cập nhật Distances (Lance-Williams formula cho Ward)
        
        # SỬA LỖI: Chỉ duyệt qua các chỉ mục ban đầu (0 đến N-1) của ma trận
        for m in range(N): 
            # Bỏ qua cụm i, cụm j, và các cụm đã bị gộp (vì distances_sq[i,m] đã là inf)
            # Điều kiện distances_sq[i, m] == np.inf kiểm tra cụm m đã bị gộp hay chưa
            if m == i or m == j or distances_sq[i, m] == np.inf:
                continue

            # Các khoảng cách Ward Dist^2 cũ
            d_im_sq = distances_sq[i, m]
            d_jm_sq = distances_sq[j, m]
            d_ij_sq = best_dist_sq
            
            n_m = counts[m]
            
            # Công thức Lance-Williams (Ward)
            new_d_sq = (
                (size_i + n_m) * d_im_sq +
                (size_j + n_m) * d_jm_sq -
                n_m * d_ij_sq
            ) / (size_i + size_j + n_m)
            
            # Ghi vào vị trí i (cụm mới)
            distances_sq[i, m] = distances_sq[m, i] = np.float64(new_d_sq)

        # 6. Cập nhật trạng thái cụm
        counts[i] = new_size
        cluster_labels[i] = current_cluster_id
        
        # Loại bỏ cụm j bằng cách đặt khoảng cách đến nó thành INF
        distances_sq[j, :] = np.inf
        distances_sq[:, j] = np.inf
        
        # Đặt lại khoảng cách từ cụm i đến chính nó (dù đã được đảm bảo ở trên)
        distances_sq[i, i] = np.inf
        
        current_cluster_id += 1

    return linkage_matrix

# =============================================
# 🧩 Class HierarchicalWardManual
# =============================================

class HierarchicalWardManual:
    """
    Thực hiện Hierarchical Ward Clustering thủ công (Manual O(N^2)).
    """
    
    def __init__(self, threshold=None):
        self.threshold = threshold
        self.Z = None 
        self.X_scaled = None
        
    @staticmethod
    def _zscore(X):
        return _zscore(X)
        
    def fit(self, X):
        """
        Chuẩn hóa dữ liệu X và tính toán Linkage Matrix Z bằng O(N^2).
        """
        X_ = _zscore(X)
        self.X_scaled = X_
        # Sử dụng hàm build O(N^2)
        self.Z = _build_linkage_matrix_manual(X_) 
        return self
    
    def predict(self, k=None, height=None):
        """
        Trả về nhãn phân cụm dựa trên số cụm (k) hoặc ngưỡng khoảng cách (height).
        """
        if self.Z is None:
            raise AttributeError("Phải gọi .fit(X) trước khi gọi .predict().")
        
        Z_ = np.asarray(self.Z, dtype=np.float64)
        
        if k is not None:
            return fcluster(Z_, t=k, criterion="maxclust")
        elif height is not None:
            return fcluster(Z_, t=height, criterion="distance")
        else:
            raise ValueError("Phải cung cấp k (maxclust) hoặc height (distance).")
            
    # --- PHƯƠNG THỨC BỔ SUNG: DÙNG CHO APP.PY (Curves & Dendrogram) ---
    
    @staticmethod
    def dendrogram_coords(X, standardize=True, truncate_mode=None, p=30) -> Dict:
        """
        Xây dựng ma trận liên kết (Z) và trích xuất tọa độ Dendrogram 
        (dùng cho Plotly trong Streamlit). (FIX: BỔ SUNG PHƯƠNG THỨC BỊ THIẾU)
        """
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=np.float64)
        Z = _build_linkage_matrix_manual(X_)
        
        # Hàm dendrogram của scipy cần Z matrix
        d_result = dendrogram(
            Z, 
            truncate_mode=truncate_mode, 
            p=p, 
            no_plot=True, 
            show_leaf_counts=True
        )
        
        return {
            'icoord': d_result['icoord'],
            'dcoord': d_result['dcoord'],
            'ivl': d_result['ivl'],
            'leaves': d_result['leaves'],
            'Z': Z
        }

    @staticmethod
    def sse_vs_k(X, k_min=2, k_max=11, standardize=True) -> Tuple[List[int], List[float]]:
        """
        Tính SSE cho một loạt các giá trị K. (FIX: BỔ SUNG PHƯƠNG THỨC BỊ THIẾU)
        """
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=np.float64)
        Z = _build_linkage_matrix_manual(X_)
        
        ks, sses = [], []
        for k in range(k_min, k_max):
            labels = fcluster(Z, t=k, criterion="maxclust")
            sse = _sse_from_labels(X_, labels)
            ks.append(k)
            sses.append(sse)
        
        return ks, sses
        
    @staticmethod
    def fit_k(X, k, standardize=True):
        """Phương thức tiện ích để chạy fit và predict nhanh chóng bằng số cụm K."""
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=np.float64)
        Z = _build_linkage_matrix_manual(X_)
        labels = fcluster(Z, t=k, criterion="maxclust")
        sse = _sse_from_labels(X_, labels)
        return {"labels": labels, "metrics": {"sse": sse, "k_used": int(k)}, "linkage": Z}
        
    @staticmethod
    def fit_distance(X, height, standardize=True):
        """Phương thức tiện ích để chạy fit và predict nhanh chóng bằng ngưỡng khoảng cách."""
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=np.float64)
        Z = _build_linkage_matrix_manual(X_)
        labels = fcluster(Z, t=height, criterion="distance")
        sse = _sse_from_labels(X_, labels)
        # cut_height được thêm vào metrics để app.py có thể dùng để vẽ đường cắt trên dendrogram
        return {"labels": labels, "metrics": {"sse": sse, "cut_height": float(height)}, "linkage": Z}

    @staticmethod
    def suggest_k_by_jump(X, standardize=True, top=1) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """Phương thức tiện ích gợi ý k bằng phương pháp khoảng cách lớn nhất (Jump/Elbow)."""
        X_ = _zscore(X) if standardize else np.asarray(X, dtype=np.float64)
        Z = _build_linkage_matrix_manual(X_) 
        
        heights = Z[:, 2] 
        N = Z.shape[0] + 1
        
        if len(heights) < 2:
            cut_h = float(heights[-1]) if len(heights) else 0.0
            # Hoàn thiện giá trị trả về
            return max(1, N), cut_h, Z, np.array([])
            
        diffs = np.diff(heights)
        
        # Tìm chỉ mục của bước nhảy lớn nhất (thường là N-k)
        idx = np.argsort(diffs)[::-1][:top] 
        j = int(idx[0])
        
        # Số cụm tối ưu: N - (số lần gộp + 1)
        k_suggest = N - (j + 1)
        # Ngưỡng cắt: Khoảng cách tại lần gộp trước bước nhảy lớn nhất
        cut_height = heights[j]
        
        # Hoàn thiện giá trị trả về
        return int(k_suggest), cut_height, Z, diffs
