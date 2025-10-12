import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


# CODE SVM THỦ CÔNG (NHỊ PHÂN) GỐC
# Đây là viên gạch nền tảng cho mô hình đa lớp
class BinarySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.alpha, self.lambda_param, self.n_iters = learning_rate, lambda_param, n_iters
        self.w, self.b = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = y_[idx]
                self.w -= self.alpha * dw
                self.b -= self.alpha * db

    def decision_function(self, X):
        # Trả về "điểm số" quyết định, không phải nhãn cuối cùng
        return np.dot(X, self.w) - self.b
    


# CODE SVM THỦ CÔNG (ĐA LỚP - ONE-VS-REST)
class MultiClassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.classifiers = [] # Danh sách để chứa các mô hình SVM nhị phân
        self.classes = []     # Danh sách các lớp (ví dụ: [0, 1, 2, 3])

    def fit(self, X, y):
        self.classes = np.unique(y)
        
        # Lặp qua từng lớp để huấn luyện một mô hình SVM nhị phân
        for cls in self.classes:
            # Tạo nhãn mới: +1 cho lớp hiện tại, -1 cho các lớp còn lại
            y_binary = np.where(y == cls, 1, -1)
            
            # Khởi tạo và huấn luyện một mô hình SVM nhị phân
            svm = BinarySVM(learning_rate=self.learning_rate,
                            lambda_param=self.lambda_param,
                            n_iters=self.n_iters)
            svm.fit(X, y_binary)
            
            # Lưu mô hình đã huấn luyện vào danh sách
            self.classifiers.append(svm)

    def predict(self, X):
        # Lấy điểm số quyết định từ mỗi mô hình SVM nhị phân
        decision_scores = np.array([clf.decision_function(X) for clf in self.classifiers])
        
        # Chuyển vị để có shape (n_samples, n_classifiers)
        decision_scores = decision_scores.T
        
        # Với mỗi mẫu dữ liệu, chọn lớp có điểm số cao nhất
        # np.argmax sẽ trả về chỉ số của classifier có điểm cao nhất
        # Chỉ số này tương ứng với lớp trong self.classes
        return self.classes[np.argmax(decision_scores, axis=1)]


# MAIN SCRIPT
if __name__ == "__main__":
    file_path = 'data/raw/noLabel/Online Retail.xlsx' 
    preprocessor = RFMPreprocessor(file_path)
    X_scaled, rfm_original, _ = preprocessor.process()
    print(f"   -> Xử lý hoàn tất.")

    kmeans_result = KMeansNumpy.fit_k(X_scaled, k=4, random_state=42)
    cluster_labels = kmeans_result["labels"]
    print(f"   -> Phân cụm hoàn tất.")

    print("\n>>> BƯỚC 3: Đang chuẩn bị dữ liệu cho SVM...")
    # Input (X) là ma trận RFM đã chuẩn hóa
    X_svm_features = X_scaled
    # Output (y) là nhãn cụm từ K-Means
    y_svm_target = cluster_labels
    
    # Tách dữ liệu thành 70% huấn luyện và 30% kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(
        X_svm_features, 
        y_svm_target, 
        test_size=0.3, 
        random_state=42,
        stratify=y_svm_target # Đảm bảo tỷ lệ các cụm trong train/test là như nhau
    )
    print("   -> Đã tách dữ liệu thành tập train và test.")
    print(f"      - Kích thước tập train: {X_train.shape}")
    print(f"      - Kích thước tập test:  {X_test.shape}")

    # --- BƯỚC 4: HUẤN LUYỆN SVM THỦ CÔNG ĐA LỚP ---
    print("\n>>> BƯỚC 4: Bắt đầu huấn luyện mô hình MultiClassSVM thủ công...")
    # Giảm n_iters để chạy nhanh hơn trong ví dụ này
    # Bạn có thể tăng lên để có kết quả tốt hơn
    manual_svm = MultiClassSVM(learning_rate=0.01, lambda_param=0.01, n_iters=300)
    manual_svm.fit(X_train, y_train)
    print("   -> Huấn luyện hoàn tất!")

    # --- BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ---
    print("\n>>> BƯỚC 5: Đánh giá hiệu năng mô hình trên tập test...")
    y_pred = manual_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   -> Độ chính xác của SVM thủ công: {accuracy:.4f} ({accuracy:.2%})")