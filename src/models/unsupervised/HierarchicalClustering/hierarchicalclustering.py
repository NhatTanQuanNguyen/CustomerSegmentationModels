import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


# =============================================
# 🧩 Hàm tính khoảng cách Ward giữa 2 cụm
# =============================================
def ward_distance(points1, points2, data):
    mean1 = np.mean(data[list(points1)], axis=0)
    mean2 = np.mean(data[list(points2)], axis=0)
    n1, n2 = len(points1), len(points2)
    dist_sq = np.sum((mean1 - mean2) ** 2) * (n1 * n2) / (n1 + n2)
    return math.sqrt(dist_sq)


# =============================================
# ⚡ Hierarchical Clustering (Ward - thủ công)
# =============================================
def hierarchical_clustering_fast(data):
    n = len(data)
    clusters = [set([i]) for i in range(n)]
    linkage_matrix = []
    dist_matrix = np.full((n, n), np.inf)

    # Tính khoảng cách ban đầu
    for i in range(n):
        for j in range(i + 1, n):
            dist_matrix[i, j] = dist_matrix[j, i] = ward_distance(clusters[i], clusters[j], data)

    active = list(range(n))

    # Gộp cụm
    while len(active) > 1:
        min_dist = np.inf
        pair = None
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                i, j = active[a], active[b]
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    pair = (i, j)

        if pair is None:
            break

        i, j = pair
        new_cluster = clusters[i] | clusters[j]
        new_size = len(new_cluster)
        linkage_matrix.append([float(i), float(j), float(min_dist), float(new_size)])

        clusters.append(new_cluster)
        new_idx = len(clusters) - 1

        active.remove(i)
        active.remove(j)
        active.append(new_idx)

        dist_matrix = np.pad(dist_matrix, ((0, 1), (0, 1)), constant_values=np.inf)
        for k in active[:-1]:
            dist_matrix[k, new_idx] = dist_matrix[new_idx, k] = ward_distance(clusters[k], new_cluster, data)

    return np.array(linkage_matrix)


# =============================================
# 🚀 Main clustering function
# =============================================
def run_hierarchical_clustering(file_path, n_samples=100):
    # 1️⃣ Đọc dữ liệu
    pre = RFMPreprocessor(file_path)
    X = pre.process()
    df = pre.rfm[["Recency", "Frequency", "Monetary"]].dropna().head(n_samples)
    print(f"✅ Dữ liệu đọc xong: {len(df)} dòng")

    # 2️⃣ Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # 3️⃣ Chạy clustering thủ công
    print("🚀 Đang chạy Hierarchical Clustering (Ward - thủ công)...")
    linkage_fast = hierarchical_clustering_fast(data)
    print(f"✅ Hoàn tất tính toán! Tổng số lần gộp: {len(linkage_fast)}")

    # 4️⃣ Vẽ dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_fast)
    plt.title("Dendrogram (Ward - thủ công)")
    plt.xlabel("Samples / Cluster IDs")
    plt.ylabel("Khoảng cách (Ward distance)")
    plt.grid(True)
    plt.show()

    # 5️⃣ Nhập threshold để cắt cây
    threshold = float(input("👉 Nhập khoảng cách threshold để cắt cây (VD: 3.5): "))
    labels = fcluster(linkage_fast, t=threshold, criterion='distance')

    df["Cluster"] = labels
    print("\n📊 Số cụm thu được:", df["Cluster"].nunique())
    print(df.groupby("Cluster").mean().round(2))

    # 6️⃣ Vẽ scatter plot (2 biến đầu)
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=50)
    plt.title(f"Hierarchical Clustering - Scatter plot (threshold = {threshold})")
    plt.xlabel("Recency (scaled)")
    plt.ylabel("Frequency (scaled)")
    plt.grid(True)
    plt.show()

    # 7️⃣ Trả về kết quả dạng numpy matrix (giống KMeans)
    summary = (
        df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
        .values
    )
    np.set_printoptions(precision=6, suppress=True)
    return summary


# =============================================
# 🔧 Test chạy
# =============================================
if __name__ == "__main__":
    file_path = "data/raw/noLabel/Online Retail.xlsx"
    result = run_hierarchical_clustering(file_path, n_samples=100)
    print("\n📈 Kết quả phân cụm :")
    print(result)
