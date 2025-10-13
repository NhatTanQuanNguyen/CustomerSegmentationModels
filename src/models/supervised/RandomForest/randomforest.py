import numpy as np
from sklearn.model_selection import train_test_split

from collections import Counter
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.preprocesses.noLabel.cleanData import RFMPreprocessor

class SimpleDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None  

    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def split(self, X, y, feature, threshold):
        left = X[:, feature] <= threshold
        right = X[:, feature] > threshold
        return (X[left], y[left]), (X[right], y[right])

    def best_split(self, X, y):
        best_gini, best_feat, best_thresh, best_gain = 1e9, None, None, 0.0
        n_samples, n_features = X.shape
        gini_parent = self.gini(y)

        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_sorted, y_sorted = X[sorted_idx], y[sorted_idx]
            unique_vals = np.unique(X_sorted[:, feature])

            thresholds = np.random.choice(unique_vals, size=min(10, len(unique_vals)), replace=False)

            for thr in thresholds:
                left_mask = X_sorted[:, feature] <= thr
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                g_left = self.gini(y_sorted[left_mask])
                g_right = self.gini(y_sorted[right_mask])
                weighted = (left_mask.sum() * g_left + right_mask.sum() * g_right) / n_samples
                gain = gini_parent - weighted  

                if weighted < best_gini:
                    best_gini = weighted
                    best_feat = feature
                    best_thresh = thr
                    best_gain = gain

        return best_feat, best_thresh, best_gain

    def build_tree(self, X, y, depth=0, importances=None):
        if importances is None:
            importances = np.zeros(X.shape[1])

        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0], importances

        feat, thr, gain = self.best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0], importances

        importances[feat] += gain  

        (X_left, y_left), (X_right, y_right) = self.split(X, y, feat, thr)
        left_tree, importances = self.build_tree(X_left, y_left, depth + 1, importances)
        right_tree, importances = self.build_tree(X_right, y_right, depth + 1, importances)

        node = {
            "feature": feat,
            "threshold": thr,
            "left": left_tree,
            "right": right_tree
        }
        return node, importances

    def fit(self, X, y):
        self.tree, importances = self.build_tree(X, y)
        total = np.sum(importances)
        if total > 0:
            self.feature_importances_ = importances / total
        else:
            self.feature_importances_ = np.zeros(X.shape[1])

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        else:
            return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class RandomForestManual:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = np.random.RandomState(random_state)
        self.trees = []
        self._feature_importances = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        feature_importances = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            idx = self.random_state.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idx], y[idx]
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            feature_importances += tree.feature_importances_
        total = np.sum(feature_importances)
        self._feature_importances = feature_importances / total if total > 0 else feature_importances

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = [Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(y_pred)

    @property
    def feature_importances_(self):
        return self._feature_importances

class RandomForestCluster:
    def __init__(self, n_estimators=200, max_depth=None, test_size=0.3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_importance_ = None

    def initialize_model(self):
        self.model = RandomForestManual(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def load_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)
    
    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.model.trees])
        n_samples = X.shape[0]
        n_classes = len(np.unique(preds))
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            counts = np.bincount(preds[:, i].astype(int), minlength=n_classes)
            proba[i] = counts / np.sum(counts)
        return proba

    def feature_importance(self, feature_cols):
        self.feature_importance_ = np.array([
            (name, float(imp)) for name, imp in zip(feature_cols, self.model.feature_importances_)
        ], dtype=object)
        return self.feature_importance_

    def run(self, X, y, feature_cols):
        self.load_data(X, y)
        self.initialize_model()
        self.fit()
        self.feature_importance(feature_cols)

        y_pred = self.model.predict(self.X_test)
        y_proba = self.predict_proba(self.X_test) 

        feature_importance = np.array([
            [name, float(imp)] for name, imp in zip(feature_cols, self.model.feature_importances_)
        ], dtype=object)

        return {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "feature_importance": feature_importance
        }

class CustomerClusteringPipeline:
    def __init__(self, file_path, kmeans_k=3,
                 rf_n_estimators=300, rf_test_size=0.3,
                 rf_max_depth=None, random_state=42):
        self.file_path = file_path
        self.kmeans_k = kmeans_k
        self.rf_n_estimators = rf_n_estimators
        self.rf_test_size = rf_test_size
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state

        self.X = None
        self.y_kmeans = None
        self.feature_cols = ["Recency", "Frequency", "Monetary"]
        self.rf_output = None

    def load_rfm_data(self):
        preprocessor = RFMPreprocessor(self.file_path)
        X_scaled, rfm_original, _ = preprocessor.process()
        self.X = X_scaled
        return self.X

    def run_kmeans(self):
        kmeans_out = KMeansNumpy.fit_k(self.X, self.kmeans_k, random_state=self.random_state)
        self.y_kmeans = kmeans_out["labels"]
        return kmeans_out

    def run_random_forest(self):
        rf_cluster = RandomForestCluster(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            test_size=self.rf_test_size,
            random_state=self.random_state
        )
        self.rf_output = rf_cluster.run(self.X, self.y_kmeans, self.feature_cols)
        return self.rf_output

    def run_pipeline(self):
        self.load_rfm_data()
        kmeans_res = self.run_kmeans()
        rf_res = self.run_random_forest()
        return {
            "kmeans": kmeans_res,
            "random_forest": rf_res
        }