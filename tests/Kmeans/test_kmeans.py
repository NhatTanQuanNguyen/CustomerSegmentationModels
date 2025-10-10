import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist, pdist, squareform
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


# ------------------------------
# Hàm tính Dunn Index
# ------------------------------
def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_clusters = np.unique(labels)
    intra_dists = []
    inter_dists = []

    for i in unique_clusters:
        cluster_i = np.where(labels == i)[0]
        if len(cluster_i) > 1:
            intra_dists.append(np.max(distances[np.ix_(cluster_i, cluster_i)]))
        else:
            intra_dists.append(0)
        for j in unique_clusters:
            if i < j:
                cluster_j = np.where(labels == j)[0]
                inter_dists.append(np.min(distances[np.ix_(cluster_i, cluster_j)]))
    
    if len(inter_dists) == 0:
        return 0
    return np.min(inter_dists) / np.max(intra_dists)


# ------------------------------
# Xử lý RFM
# ------------------------------
file_path = "data/raw/noLabel/Online Retail.xlsx"  # đổi thành đường dẫn của bạn
preprocessor = RFMPreprocessor(file_path)
X = preprocessor.process()
rfm = preprocessor.rfm.copy()  # lấy bảng RFM gốc (chưa chuẩn hóa)

# ------------------------------
# Chạy K-Means và đánh giá
# ------------------------------
results = []
models = {}

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    chi = calinski_harabasz_score(X, labels)
    dunn = dunn_index(X, labels)

    results.append({
        "k": k,
        "Silhouette": sil,
        "DBI": dbi,
        "CHI": chi,
        "Dunn": dunn
    })

    models[k] = (kmeans, labels)

results_df = pd.DataFrame(results)
print("\n=== Kết quả đánh giá K-Means ===")
print(results_df)

# ------------------------------
# Chọn số cụm tối ưu
# ------------------------------
best_k = results_df.loc[results_df["Silhouette"].idxmax(), "k"]
print(f"\n✅ Số cụm tối ưu theo Silhouette: {best_k}")

best_model, best_labels = models[best_k]
rfm["Cluster"] = best_labels

# ------------------------------
# 🧮 Thống kê trung bình mỗi cụm
# ------------------------------
cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
cluster_summary["Count"] = rfm["Cluster"].value_counts().sort_index().values
cluster_summary = cluster_summary.sort_values("Monetary", ascending=False)

print("\n=== Thống kê trung bình mỗi cụm ===")
print(cluster_summary)
