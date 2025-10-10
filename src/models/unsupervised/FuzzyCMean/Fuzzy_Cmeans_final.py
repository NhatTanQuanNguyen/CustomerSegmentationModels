import numpy as np
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, max_iter=100, error=1e-5, random_state=None):
        self.n_clusters = n_clusters  # số cụm
        self.m = m                    # hệ số mờ (thường = 2)
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

        # Các thuộc tính sẽ được gán sau khi fit
        self.centers = None
        self.U = None  # ma trận mức độ thành viên
        self.labels_ = None  # ✅ thêm dòng này để tránh lỗi AttributeError
        self.centroids = None  # ✅ thêm dòng này để tránh lỗi khi gọi fcm.centroids

    def _initialize_U(self, n_samples):
        np.random.seed(self.random_state)
        U = np.random.rand(n_samples, self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)  # chuẩn hóa để tổng mỗi hàng = 1
        return U

    def _compute_centers(self, X, U):
        um = U ** self.m
        centers = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return centers

    def _update_U(self, X, centers):
        n_samples = X.shape[0]
        n_clusters = centers.shape[0]
        dist = np.zeros((n_samples, n_clusters))

        # Tính khoảng cách Euclidean giữa mỗi điểm và mỗi tâm cụm
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)

        # Tránh chia cho 0
        dist = np.fmax(dist, 1e-10)

        # Cập nhật ma trận U theo công thức FCM
        exponent = 2 / (self.m - 1)
        denominator = np.sum((dist[:, :, None] / dist[:, None, :]) ** exponent, axis=2)
        U_new = 1 / denominator
        return U_new

    def fit(self, X):
        n_samples = X.shape[0]
        self.U = self._initialize_U(n_samples)

        for iteration in range(self.max_iter):
            centers_old = self.centers
            self.centers = self._compute_centers(X, self.U)
            U_new = self._update_U(X, self.centers)

            # Tính sai số hội tụ
            diff = np.linalg.norm(U_new - self.U)
            self.U = U_new

            if diff < self.error:
                print(f"✅ Converged at iteration {iteration+1}")
                break

        # ✅ chỉ thêm 2 dòng này, không đổi biến
        self.labels_ = np.argmax(self.U, axis=1)
        self.centroids = self.centers

        return self

    def predict(self, X):
        """
        Trả về cụm mà mỗi điểm có độ thuộc lớn nhất
        """
        if self.U is None:
            raise Exception("Model chưa được fit!")
        return np.argmax(self.U, axis=1)

    def get_membership(self):
        """
        Trả về ma trận U (độ thành viên của mỗi điểm trong mỗi cụm)
        """
        return self.U
    
    def fuzzy_partition_coefficient(self):
        """
        Tính hệ số phân chia mờ (Fuzzy Partition Coefficient - FPC)
        """
        if self.U is None:
            raise Exception("Model chưa được fit!")
        n_samples = self.U.shape[0]
        return np.sum(self.U ** 2) / n_samples

    @staticmethod
    def cluster(file_path):
        file_path = "data/raw/noLabel/Online Retail.xlsx"  
        pre = RFMPreprocessor(file_path)    
        X = pre.process()

        # Tìm số cụm tối ưu bằng FPC
        cs = range(2, 10)
        fpcs = []
        for c in cs:
            fcm = FuzzyCMeans(n_clusters=c, m=2, max_iter=200, error=1e-5, random_state=42)
            fcm.fit(X)
            fpc = fcm.fuzzy_partition_coefficient()
            fpcs.append(fpc)

        # Chọn số cụm với giá trị FPC lớn nhất
        best_c = cs[np.argmax(fpcs)]
        # Huấn luyện Fuzzy C-Means
        fcm = FuzzyCMeans(n_clusters=best_c, m=2, max_iter=200, error=1e-5, random_state=42)
        fcm.fit(X)

        # Lấy kết quả
        labels = fcm.predict(X)
        memberships = fcm.get_membership()
        centers = fcm.centers

        # Gắn kết quả vào bảng RFM
        pre.rfm["Cluster"] = labels
        pre.rfm["MembershipMax"] = memberships.max(axis=1)

        cluster_summary = (
            pre.rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .reset_index()
        )
        np.set_printoptions(precision=6, suppress=True)

        return cluster_summary.values
