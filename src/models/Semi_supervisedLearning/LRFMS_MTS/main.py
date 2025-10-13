import time, numpy as np, pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from src.preprocesses.noLabel.cleanDataRMFLS import LRFMSPreprocessor
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator


# ===============================================================
# Helper
# ===============================================================
def _scale_block(X: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "none":
        return X
    if method == "minmax":
        return MinMaxScaler().fit_transform(X)
    return StandardScaler().fit_transform(X)


def _normalize_mts(F: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return StandardScaler().fit_transform(F), StandardScaler().fit_transform(M)


def build_mts(df: pd.DataFrame, cust_col="CustomerID", date_col="InvoiceDate",
              money_col="TotalAmount", freq="M", periods=12,
              ref_date=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if ref_date is None:
        ref_date = df[date_col].max()

    cols = pd.period_range(end=ref_date.to_period(freq), periods=periods, freq=freq).astype(str)
    df["Period"] = df[date_col].dt.to_period(freq).astype(str)

    f_tab = df.groupby([cust_col, "Period"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    m_tab = df.groupby([cust_col, "Period"])[money_col].sum().unstack(fill_value=0).reindex(columns=cols, fill_value=0.0)

    F_scaled, M_scaled = _normalize_mts(f_tab.to_numpy(float), m_tab.to_numpy(float))
    mts_flat = np.hstack([F_scaled, M_scaled])
    return mts_flat, f_tab.index.to_numpy(), cols


# ===============================================================
# Main model: LRFMS + MTS Early Fusion
# ===============================================================
class LRFMS_MTS:
    def __init__(self, k=4, periods=12, freq="M", alpha=0.5,
                 lrfms_scaler="zscore", mts_scaler="zscore",
                 random_state=42, pca_dim=2):
        self.k = k
        self.periods = periods
        self.freq = freq
        self.alpha = alpha
        self.lrfms_scaler = lrfms_scaler
        self.mts_scaler = mts_scaler
        self.random_state = random_state
        self.pca_dim = pca_dim

        self.labels_ = None
        self.customer_ids_ = None
        self.cols_periods_ = None
        self.features_pca_ = None
        self.fused_features_ = None
        self.metrics_ = None

    # ===========================================================
    # Fit model
    # ===========================================================
    def fit(self, excel_path: str) -> Dict:
        t0 = time.perf_counter()
        print("=== BẮT ĐẦU HUẤN LUYỆN LRFMS + MTS ===")

        # --- LRFMS ---
        pre = LRFMSPreprocessor(excel_path)
        df = pre.handle_outliers(pre.add_total_amount(pre.filter_invalid(pre.remove_duplicates(pre.read_data()))), "TotalAmount")

        lrfms = pre.calculate_lrfms(df)
        lrfms_scaled = pre.standardize_with_library(
            lrfms, ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"],
            method=self.lrfms_scaler
        )

        X_l = lrfms_scaled[["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]].to_numpy()
        cid_l = lrfms_scaled["CustomerID"].to_numpy()

        # --- MTS ---
        mts_flat, cid_m, cols = build_mts(
            df[df["CustomerID"].isin(cid_l)],
            periods=self.periods, freq=self.freq
        )
        order = pd.Index(cid_l).get_indexer(cid_m)
        X_l = X_l[order]

        # --- Fusion ---
        Xl = _scale_block(X_l, self.lrfms_scaler)
        Xm = _scale_block(mts_flat, self.mts_scaler)
        Xl /= np.sqrt(np.var(Xl, axis=0).mean())
        Xm /= np.sqrt(np.var(Xm, axis=0).mean())
        fused_features = np.hstack([self.alpha * Xl, (1 - self.alpha) * Xm])

        # --- KMeans ---
        km = KMeansNumpy(n_clusters=self.k, random_state=self.random_state).fit(fused_features)
        labels = km.labels_

        # --- Evaluation (dùng evaluator chuẩn) ---
        evaluator = UnsupervisedEvaluator(fused_features)
        metrics = evaluator.evaluate(labels)

        # --- PCA ---
        features_pca = None
        if self.pca_dim > 0:
            pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
            features_pca = pca.fit_transform(fused_features)

        # --- Lưu instance state ---
        self.labels_ = labels
        self.customer_ids_ = cid_m
        self.cols_periods_ = cols
        self.fused_features_ = fused_features
        self.features_pca_ = features_pca
        self.metrics_ = metrics

        print(f"Thời gian chạy: {time.perf_counter() - t0:.2f}s")

        return {
            "labels": labels,
            "metrics": metrics,
            "customer_ids": cid_m,
            "period_cols": cols,
            "fused_features": fused_features,
            "features_pca": features_pca
        }

    # ===========================================================
    # GUI helper functions
    # ===========================================================
    def lrfms_profile(self, df_clean: pd.DataFrame):
        if self.labels_ is None:
            raise ValueError("Chưa chạy fit().")
        pre = LRFMSPreprocessor("unused")
        pre.read_data = lambda: df_clean
        lrfms = pre.calculate_lrfms(df_clean)
        lrfms["label"] = self.labels_
        cols = ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]
        return lrfms.groupby("label")[cols].mean().reset_index(), cols

    def mts_means(self, df_clean: pd.DataFrame):
        if self.labels_ is None or self.customer_ids_ is None:
            raise ValueError("Chưa chạy fit().")
        df = df_clean.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Period"] = df["InvoiceDate"].dt.to_period(self.freq).astype(str)
        cols = self.cols_periods_

        labmap = pd.DataFrame({"CustomerID": self.customer_ids_, "label": self.labels_})
        f_tab = df.groupby(["CustomerID", "Period"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        m_tab = df.groupby(["CustomerID", "Period"])["TotalAmount"].sum().unstack(fill_value=0).reindex(columns=cols, fill_value=0.0)

        f_tab = labmap.set_index("CustomerID").join(f_tab, how="inner").groupby("label")[cols].mean()
        m_tab = labmap.set_index("CustomerID").join(m_tab, how="inner").groupby("label")[cols].mean()
        return f_tab, m_tab, cols
