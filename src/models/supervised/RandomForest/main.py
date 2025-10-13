# src/models/supervised/RandomForestManual/main.py

import numpy as np
from collections import Counter

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
                    best_gini, best_feat, best_thresh, best_gain = weighted, feature, thr, gain

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
        node = {"feature": feat, "threshold": thr, "left": left_tree, "right": right_tree}
        return node, importances

    def fit(self, X, y):
        self.tree, importances = self.build_tree(X, y)
        total = np.sum(importances)
        self.feature_importances_ = importances / total if total > 0 else np.zeros(X.shape[1])

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self.predict_one(x, node["left"])
        return self.predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class RandomForestManual:
    def __init__(self, n_estimators=50, max_depth=None, random_state=None):
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

    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        n_samples = X.shape[0]
        n_classes = len(np.unique(preds))
        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            counts = np.bincount(preds[:, i].astype(int), minlength=n_classes)
            proba[i] = counts / np.sum(counts)
        return proba

    @property
    def feature_importances_(self):
        return self._feature_importances
