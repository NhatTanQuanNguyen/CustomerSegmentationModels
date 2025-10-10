import numpy as np
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


# Thuật toán K-Means thủ công 
class KMeansNumpy:
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # Khởi tạo ngẫu nhiên tâm cụm
        random_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # Gán cụm gần nhất cho từng điểm
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Tính lại tâm cụm
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if len(X[labels == k]) > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            # Nếu tâm cụm không thay đổi => dừng
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


# Phương pháp Elbow chọn K tối ưu
def elbow_method(X, k_range=(1, 10), random_state=42):
    sse = []

    for k in range(k_range[0], k_range[1] + 1):
        model = KMeansNumpy(n_clusters=k, random_state=random_state)
        model.fit(X)
        distances = np.linalg.norm(X - model.centroids[model.labels_], axis=1)
        sse.append(np.sum(distances ** 2))

    plt.plot(range(k_range[0], k_range[1] + 1), sse, 'bo-')
    plt.xlabel('Số cụm (K)')
    plt.ylabel('Tổng sai số bình phương (SSE)')
    plt.title('Phương pháp Elbow – Chọn K tối ưu')
    plt.grid(True)
    plt.show()

    print("→ Quan sát điểm khuỷu tay trên biểu đồ để chọn K hợp lý.")


# Chạy toàn bộ pipeline
if __name__ == "__main__":
    file_path = "data/raw/noLabel/Online Retail.xlsx"  
    pre = RFMPreprocessor(file_path)
    X = pre.process()

    # Chọn số cụm K bằng Elbow
    elbow_method(X, k_range=(2, 10))

    # Sau khi quan sát biểu đồ, chọn K 
    k = int(input("Nhập số cụm K đã chọn: "))
    kmeans = KMeansNumpy(n_clusters=k, random_state=0)
    kmeans.fit(X)

    # Gắn nhãn cụm vào bảng RFM
    pre.rfm["Cluster"] = kmeans.labels_
    print(pre.rfm.head())

    cluster_summary = pre.rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    print(cluster_summary)

    
