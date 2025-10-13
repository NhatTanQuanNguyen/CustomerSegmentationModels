# src/gui/app_4.py
import os, sys, numpy as np, pandas as pd, streamlit as st
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.Semi_supervisedLearning.LRFMS_MTS.main import LRFMS_MTS
from src.preprocesses.noLabel.cleanDataRMFLS import LRFMSPreprocessor

st.set_page_config(page_title="LRFMS + MTS — KMeans (Early Fusion)", layout="wide")
st.title("LRFMS + MTS — Early Fusion KMeans (tùy chỉnh cao)")

with st.sidebar:
    src_mode = st.radio("Nguồn dữ liệu", ["Path nội bộ", "Upload Excel"], index=0)
    default_path = "data/raw/noLabel/Online Retail.xlsx"
    file_path = st.text_input("Đường dẫn Excel", value=default_path) if src_mode == "Path nội bộ" else None
    up_file = st.file_uploader("Chọn file Excel", type=["xlsx", "xls"]) if src_mode != "Path nội bộ" else None

    st.divider()
    st.subheader("Tham số MTS")
    periods = st.slider("Số kỳ T", 3, 36, 12)
    freq = st.selectbox("Granularity", ["M", "W", "Q"], index=0)
    per_period_z = st.checkbox("Z-score theo từng kỳ (per-column)", value=True)

    st.divider()
    st.subheader("Tham số LRFMS + Fusion")
    k = st.slider("Số cụm k", 2, 15, 4)
    alpha = st.slider("Trọng số LRFMS (0..1)", 0.0, 1.0, 0.5, 0.05)
    lrfms_scaler = st.selectbox("Scaler LRFMS", ["zscore", "minmax", "none"], index=0)
    mts_scaler = st.selectbox("Scaler MTS", ["zscore", "minmax", "none"], index=0)
    with_mean = st.checkbox("Z-score with_mean", value=True)
    pca_dim = st.selectbox("PCA để trực quan", [0, 2], index=1)

    st.divider()
    seed = st.number_input("Random state", value=42, step=1)
    run_btn = st.button("Chạy phân cụm", use_container_width=True)

@st.cache_data(show_spinner=True)
def load_clean_df(src):
    pre = LRFMSPreprocessor(src)
    df = pre.read_data()
    df = pre.remove_duplicates(df)
    df = pre.filter_invalid(df)
    df = pre.add_total_amount(df)
    df = pre.handle_outliers(df, "TotalAmount")
    return df

if run_btn:
    if src_mode == "Path nội bộ":
        if not os.path.exists(file_path):
            st.error(f"Không tìm thấy file: {file_path}")
            st.stop()
        df_clean = load_clean_df(file_path)
        src_used = file_path
    else:
        if up_file is None:
            st.error("Hãy upload file Excel.")
            st.stop()
        df_clean = load_clean_df(up_file)
        src_used = "uploaded_file"

    st.write(f"Số dòng sau làm sạch: {len(df_clean):,}")
    st.dataframe(df_clean.head(8), use_container_width=True)

    st.subheader("Huấn luyện")
    with st.spinner("Đang chạy..."):
        model = LRFMS_MTS(
            k=k, periods=periods, freq=freq, alpha=alpha,
            lrfms_scaler=lrfms_scaler, mts_scaler=mts_scaler,
            per_period_zscore=per_period_z, with_mean=with_mean,
            random_state=seed, pca_dim=pca_dim
        )
        res = model.fit(src_used)

    # ===== Metrics & counts =====
    metrics = res["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", f"{metrics['silhouette']:.4f}")
    c2.metric("Calinski–Harabasz", f"{metrics['calinski_harabasz']:.2f}")
    c3.metric("Davies–Bouldin", f"{metrics['davies_bouldin']:.4f}")

    counts = pd.Series(res["labels"]).value_counts().sort_index()
    st.markdown("**Phân bố cụm**")
    st.bar_chart(counts)

    st.download_button(
        "Tải CSV nhãn cụm",
        data=pd.DataFrame({"CustomerID": res["customer_ids"], "label": res["labels"]}).to_csv(index=False),
        file_name="clusters_lrfms_mts.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ===== Radar LRFMS per cluster =====
    st.subheader("Profile LRFMS theo cụm")
    prof_df, lrfms_cols = model.lrfms_profile(df_clean)
    st.dataframe(prof_df, use_container_width=True)

    radar = go.Figure()
    for _, row in prof_df.iterrows():
        radar.add_trace(go.Scatterpolar(
            r=row[lrfms_cols].values, theta=lrfms_cols, fill="toself", name=f"Cluster {int(row['label'])}"
        ))
    radar.update_layout(title="Radar LRFMS (mean theo cụm)")
    st.plotly_chart(radar, use_container_width=True)

    # ===== MTS means per cluster (lines) =====
    st.subheader("MTS trung bình theo cụm")
    freq_mean, money_mean, cols = model.mts_means(df_clean)
    tabs = st.tabs([f"Cluster {i}" for i in freq_mean.index.tolist()])
    for i, tab in zip(freq_mean.index.tolist(), tabs):
        with tab:
            f_fig = go.Figure()
            f_fig.add_trace(go.Scatter(x=cols, y=freq_mean.loc[i].values, mode="lines+markers", name="Frequency"))
            f_fig.update_layout(title=f"Frequency theo thời gian — Cluster {i}", xaxis_title="Period", yaxis_title="Count")
            st.plotly_chart(f_fig, use_container_width=True)

            m_fig = go.Figure()
            m_fig.add_trace(go.Scatter(x=cols, y=money_mean.loc[i].values, mode="lines+markers", name="Monetary"))
            m_fig.update_layout(title=f"Monetary theo thời gian — Cluster {i}", xaxis_title="Period", yaxis_title="Sum")
            st.plotly_chart(m_fig, use_container_width=True)

    # ===== PCA scatter of fused features =====
    if res["features_pca"] is not None:
        st.subheader("PCA scatter (fused features)")
        df_p = pd.DataFrame(res["features_pca"], columns=["PC1","PC2"])
        df_p["label"] = res["labels"]
        fig = px.scatter(df_p, x="PC1", y="PC2", color="label", title="PCA 2D of fused features", opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Ghi chú: Early fusion (5 LRFMS + 2×T MTS), scaler block riêng, α cân bằng hai khối, KMeans (Euclid).")
