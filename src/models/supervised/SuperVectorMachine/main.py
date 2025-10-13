import numpy as np

def _sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))
def _fit_platt(f_decision: np.ndarray, y_binary_pm1: np.ndarray, max_iter: int = 100):
    y01 = (y_binary_pm1 == 1).astype(float)
    f = f_decision.astype(float)
    A, B = 0.0, np.log((y01.mean() + 1e-6) / (1 - y01.mean() + 1e-6))
    for _ in range(max_iter):
        z = A * f + B
        p = _sigmoid(z)
        g1 = np.sum((p - y01) * f)
        g2 = np.sum(p - y01)

        w = p * (1 - p)
        h11 = np.sum(w * f * f)
        h22 = np.sum(w)
        h12 = np.sum(w * f)

        det = h11 * h22 - h12 * h12
        if det <= 1e-12:
            break
        dA = (-g1 * h22 + g2 * h12) / det
        dB = (-g2 * h11 + g1 * h12) / det

        step = 1.0
        old_loss = -np.sum(y01 * np.log(p + 1e-12) + (1 - y01) * np.log(1 - p + 1e-12))
        for _ls in range(20):
            A_try = A + step * dA
            B_try = B + step * dB
            p_try = _sigmoid(A_try * f + B_try)
            new_loss = -np.sum(y01 * np.log(p_try + 1e-12) + (1 - y01) * np.log(1 - p_try + 1e-12))
            if new_loss <= old_loss - 1e-9:
                A, B = A_try, B_try
                break
            step *= 0.5
        if step < 1e-6:
            break

    return float(A), float(B)

class SVM_RBF:
    def __init__(self, C=1.0, gamma=0.1, tol=1e-4, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = None
        self.b = 0.0
        self.support_vectors_ = None
        self.support_labels_ = None  # labels in {-1, +1}
        # Platt params
        self.platt_A_ = None
        self.platt_B_ = None

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
        self.b = 0.0

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
                        self.b = 0.5 * (b1 + b2)

            if np.linalg.norm(self.alpha - alpha_prev) < 1e-5:
                break

        idx = self.alpha > 1e-5
        self.support_vectors_ = X[idx]
        self.support_labels_ = y[idx]
        self.alpha = self.alpha[idx]

        return self

    def decision_function(self, X):
        return self.project(X)

    def project(self, X):
        if self.support_vectors_ is None or len(self.support_vectors_) == 0:
            return np.zeros(X.shape[0])
        K = self.rbf_kernel(X, self.support_vectors_)
        return np.dot(K, self.alpha * self.support_labels_) + self.b

    def predict(self, X):
        return np.sign(self.project(X))
    def calibrate(self, X_calib, y_calib_pm1):
        f = self.decision_function(X_calib)
        A, B = _fit_platt(f, y_calib_pm1)
        self.platt_A_ = A
        self.platt_B_ = B
        return self

    def predict_proba_binary(self, X):
        f = self.decision_function(X)
        A = 1.0 if self.platt_A_ is None else self.platt_A_
        B = 0.0 if self.platt_B_ is None else self.platt_B_
        p_pos = _sigmoid(A * f + B)
        p_neg = 1.0 - p_pos
        return np.vstack([p_neg, p_pos]).T  # shape (n, 2)
class MultiClassSVM:
    def __init__(self, C=1.0, gamma=0.1, max_iter=1000, calib_split=0.0, random_state=42):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.calib_split = calib_split
        self.random_state = random_state

        self.models = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        rng = np.random.RandomState(self.random_state)

        for cls in self.classes:
            # y_binary in {-1,+1}
            y_binary_pm1 = np.where(y == cls, 1, -1).astype(float)
            model = SVM_RBF(C=self.C, gamma=self.gamma, max_iter=self.max_iter)
            # Nếu cần tách calibration
            if self.calib_split > 0.0:
                idx = np.arange(len(y))
                rng.shuffle(idx)
                n_calib = int(len(y) * self.calib_split)
                calib_idx = idx[:n_calib]
                train_idx = idx[n_calib:]
                model.fit(X[train_idx], y_binary_pm1[train_idx])
                # học A,B trên phần calib
                model.calibrate(X[calib_idx], y_binary_pm1[calib_idx])
            else:
                model.fit(X, y_binary_pm1)
                # calibrate trên chính train (đơn giản)
                model.calibrate(X, y_binary_pm1)

            self.models[cls] = model
        return self

    def decision_function(self, X):
        scores = np.array([mdl.decision_function(X) for mdl in self.models.values()])
        return scores

    def predict(self, X):
        # Dùng xác suất (ổn định hơn score thô)
        proba = self.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        return self.classes[preds]

    def predict_proba(self, X):
        class_probs = []
        for cls in self.classes:
            mdl = self.models[cls]
            p = mdl.predict_proba_binary(X)[:, 1]  # lấy cột +1
            class_probs.append(p)

        proba = np.vstack(class_probs).T  # (n_samples, n_classes)
        row_sum = proba.sum(axis=1, keepdims=True)
        zero_row = (row_sum[:, 0] <= 1e-12)
        if np.any(zero_row):
            proba[zero_row, :] = 1.0 / proba.shape[1]
            row_sum[zero_row, 0] = 1.0
        proba /= row_sum
        return proba


