import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

class SVM_RBF:
    def __init__(self, C=1.0, gamma=0.1, tol=1e-4, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0
        self.support_vectors_ = None
        self.support_labels_ = None

    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y):
        n_samples, _ = X.shape
        y = np.where(y <= 0, -1, 1).astype(float)

        K = self.rbf_kernel(X, X)

        self.alpha = np.zeros(n_samples)
        self.b = 0

        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            for i in range(n_samples):
                f_i = np.sum(self.alpha * y * K[:, i]) + self.b
                E_i = f_i - y[i]

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    f_j = np.sum(self.alpha * y * K[:, j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = (self.b - E_i
                          - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i]
                          - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j])
                    b2 = (self.b - E_j
                          - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j]
                          - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j])

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < 1e-5:
                break

        idx = self.alpha > 1e-5
        self.support_vectors_ = X[idx]
        self.support_labels_ = y[idx]
        self.alpha = self.alpha[idx]

    def project(self, X):
        K = self.rbf_kernel(X, self.support_vectors_)
        return np.dot(K, self.alpha * self.support_labels_) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


class MultiClassSVM:
    def __init__(self, C=1.0, gamma=0.1, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.models = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)
            model = SVM_RBF(C=self.C, gamma=self.gamma, max_iter=self.max_iter)
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        scores = np.array([model.project(X) for model in self.models.values()])
        preds = np.argmax(scores, axis=0)
        return self.classes[preds]


if __name__ == "__main__":
    file_path = "data/raw/noLabel/Online Retail.xlsx"
    preprocessor = RFMPreprocessor(file_path)
    X_scaled, rfm_original, _ = preprocessor.process()

    kmeans_result = KMeansNumpy.fit_k(X_scaled, k=4, random_state=42)
    cluster_labels = kmeans_result["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
    )

    svm_rbf = MultiClassSVM(C=2.0, gamma=0.5, max_iter=500)
    svm_rbf.fit(X_train, y_train)
    y_pred = svm_rbf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Độ chính xác SVM RBF thủ công: {acc:.4f} ({acc:.2%})")