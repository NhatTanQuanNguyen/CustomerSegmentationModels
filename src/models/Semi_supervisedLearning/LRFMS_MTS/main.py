import time, numpy as np, pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.preprocesses.noLabel.cleanDataRMFLS import LRFMSPreprocessor
from src.models.unsupervised.Kmean.main import KMeansNumpy


# ==========================================================
# Helper functions
# ==========================================================
def _scale_block(X: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Chuẩn hóa từng khối (block) LRFMS hoặc MTS."""
    if method == "none":
        return X
    if method == "minmax":
        return MinMaxScaler().fit_transform(X)
    return StandardScaler().fit_transform(X)


def _normalize_mts(F: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Chuẩn hóa frequency (F) và monetary (M) dùng thư viện sklearn."""
    scaler_f = StandardScaler()
    scaler_m = StandardScaler()
    F_scaled = scaler_f.fit_transform(F)
    M_scaled = scaler_m.fit_transform(M)
    return F_scaled, M_scaled


def build_mts(
    df: pd.DataFrame,
    cust_col: str = "CustomerID",
    date_col: str = "InvoiceDate",
    money_col: str = "TotalAmount",
    freq: str = "M",
    periods: int = 12,
    ref_date: Optional[pd.Timestamp] = None,
    per_period_zscore: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Xây dựng ma trận chuỗi thời gian MTS từ dữ liệu hóa đơn."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if ref_date is None:
        ref_date = df[date_col].max()

    cols = pd.period_range(end=ref_date.to_period(freq), periods=periods, freq=freq).astype(str)
    df["Period"] = df[date_col].dt.to_period(freq).astype(str)

    f_tab = (
        df.groupby([cust_col, "Period"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cols, fill_value=0)
    )
    m_tab = (
        df.groupby([cust_col, "Period"])[money_col]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=cols, fill_value=0.0)
    )

    # Chuẩn hóa theo thư viện sklearn
    F_scaled, M_scaled = _normalize_mts(f_tab.to_numpy(dtype=float), m_tab.to_numpy(dtype=float))
    mts_flat = np.hstack([F_scaled, M_scaled])
    return mts_flat, f_tab.index.to_numpy(), cols


# ==========================================================
# Main class: LRFMS + MTS Fusion
# ==========================================================
class LRFMS_MTS:
    """
    Early-fusion LRFMS (5 đặc trưng) + MTS (2*T đặc trưng)
    Chuẩn hóa bằng sklearn → ghép → KMeans thủ công.
    """

    def __init__(
        self,
        k: int = 4,
        periods: int = 12,
        freq: str = "M",
        alpha: float = 0.5,
        lrfms_scaler: str = "zscore",
        mts_scaler: str = "zscore",
        per_period_zscore: bool = True,
        with_mean: bool = True,
        random_state: int = 42,
        pca_dim: int = 2,
    ):
        self.k = int(k)
        self.periods = int(periods)
        self.freq = freq
        self.alpha = float(alpha)
        self.lrfms_scaler = lrfms_scaler
        self.mts_scaler = mts_scaler
        self.per_period_zscore = per_period_zscore
        self.with_mean = with_mean
        self.random_state = int(random_state)
        self.pca_dim = int(pca_dim)

        # output containers
        self.labels_ = None
        self.features_ = None
        self.features_pca_ = None
        self.customer_ids_ = None
        self.cols_periods_ = None
        self.metrics_ = None

    # ==========================================================
    # Fit toàn bộ pipeline
    # ==========================================================
    def fit(self, excel_path: str) -> Dict:
        t0 = time.perf_counter()
        print(f"[1/6] Load & preprocess LRFMS ...", flush=True)

        # --- LRFMS ---
        pre = LRFMSPreprocessor(excel_path)
        df = pre.read_data()
        df = pre.remove_duplicates(df)
        df = pre.filter_invalid(df)
        df = pre.add_total_amount(df)
        df = pre.handle_outliers(df, "TotalAmount")

        lrfms = pre.calculate_lrfms(df)
        lrfms_scaled = pre.standardize_with_library(
            lrfms,
            ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"],
            method=self.lrfms_scaler,
        )

        X_l = lrfms_scaled[
            ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]
        ].to_numpy()
        cid_l = lrfms_scaled["CustomerID"].to_numpy()
        print(f" -> LRFMS: N={X_l.shape[0]}, d=5, t={time.perf_counter()-t0:.2f}s")

        # --- MTS ---
        print("[2/6] Build MTS ...")
        mts_flat, cid_m, cols = build_mts(
            df[df["CustomerID"].isin(cid_l)],
            cust_col="CustomerID",
            date_col="InvoiceDate",
            money_col="TotalAmount",
            freq=self.freq,
            periods=self.periods,
            ref_date=df["InvoiceDate"].max(),
            per_period_zscore=self.per_period_zscore,
        )

        order = pd.Index(cid_l).get_indexer(cid_m)
        X_l = X_l[order]
        self.customer_ids_ = cid_m
        self.cols_periods_ = cols
        print(f" -> MTS: N={mts_flat.shape[0]}, d={mts_flat.shape[1]} (=2*T)")

        # --- Fusion ---
        print("[3/6] Scale & fuse ...")
        Xl = _scale_block(X_l, method=self.lrfms_scaler)
        Xm = _scale_block(mts_flat, method=self.mts_scaler)

        # Cân bằng năng lượng giữa 2 block để tránh lệch
        Xl /= np.sqrt(np.var(Xl, axis=0).mean())
        Xm /= np.sqrt(np.var(Xm, axis=0).mean())

        X_fused = np.hstack([self.alpha * Xl, (1 - self.alpha) * Xm])
        self.features_ = X_fused
        print(f" -> Fused shape: {X_fused.shape}")

        # --- KMeans ---
        print(f"[4/6] Run KMeans (k={self.k}) ...")
        km = KMeansNumpy(n_clusters=self.k, random_state=self.random_state).fit(X_fused)
        self.labels_ = km.labels_

        # --- Metrics ---
        print("[5/6] Compute metrics ...")
        sil = silhouette_score(X_fused, self.labels_) if self.k > 1 else np.nan
        ch = calinski_harabasz_score(X_fused, self.labels_) if self.k > 1 else np.nan
        db = davies_bouldin_score(X_fused, self.labels_) if self.k > 1 else np.nan
        self.metrics_ = {
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(db),
        }
        print(f" -> Sil={sil:.4f}, CH={ch:.2f}, DB={db:.4f}")

        # --- PCA for visualization ---
        if self.pca_dim > 0:
            pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
            self.features_pca_ = pca.fit_transform(X_fused)
            print(f"[6/6] PCA done: shape={self.features_pca_.shape}")
        else:
            self.features_pca_ = None

        print(f"TOTAL TIME: {time.perf_counter()-t0:.2f}s")

        return {
            "labels": self.labels_,
            "metrics": self.metrics_,
            "customer_ids": self.customer_ids_,
            "period_cols": self.cols_periods_,
            "features": self.features_,
            "features_pca": self.features_pca_,
        }

    # ==========================================================
    # GUI helper functions
    # ==========================================================
    def lrfms_profile(self, df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """Trả về trung bình LRFMS theo từng cụm để hiển thị GUI."""
        pre = LRFMSPreprocessor("unused")
        pre.read_data = lambda: df_clean
        lrfms = pre.calculate_lrfms(df_clean)
        prof = lrfms.copy()
        prof["label"] = self.labels_
        cols = ["Loyalty", "Recency", "Frequency", "Monetary", "Satisfaction"]
        prof_mean = prof.groupby("label")[cols].mean().reset_index()
        return prof_mean, cols

    def mts_means(self, df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Trả về trung bình Frequency & Monetary theo thời gian (GUI dùng vẽ đồ thị)."""
        df = df_clean.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Period"] = df["InvoiceDate"].dt.to_period(self.freq).astype(str)
        cols = self.cols_periods_
        labmap = pd.DataFrame({"CustomerID": self.customer_ids_, "label": self.labels_})
        f_tab = df.groupby(["CustomerID", "Period"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
        m_tab = df.groupby(["CustomerID", "Period"])["TotalAmount"].sum().unstack(fill_value=0).reindex(columns=cols, fill_value=0.0)
        f_tab = labmap.set_index("CustomerID").join(f_tab, how="inner")
        m_tab = labmap.set_index("CustomerID").join(m_tab, how="inner")
        return f_tab.groupby("label")[cols].mean(), m_tab.groupby("label")[cols].mean(), cols
