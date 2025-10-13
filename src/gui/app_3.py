import os, sys, numpy as np, pandas as pd, streamlit as st
import plotly.express as px
import plotly.graph_objects as go

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.models.Semi_supervisedLearning.LRFMS_MTS.main import LRFMS_MTS
from src.preprocesses.noLabel.cleanDataRMFLS import LRFMSPreprocessor

st.set_page_config(page_title="LRFMS + MTS — Early Fusion KMeans", layout="wide")
st.title("📊 Dashboard phân cụm khách hàng — LRFMS + MTS (Early Fusion)")

with st.sidebar:
    src_mode = st.radio("Nguồn dữ liệu", ["Path nội bộ", "Upload Excel"], 0)
    default_path = "data/raw/noLabel/Online Retail.xlsx"
    file_path = st.text_input("Đường dẫn Excel", default_path) if src_mode == "Path nội bộ" else None
    up_file = st.file_uploader("Upload file Excel", type=["xlsx", "xls"]) if src_mode != "Path nội bộ" else None
    k = st.slider("Số cụm (k)", 2, 15, 4)
    periods = st.slider("Số kỳ (T)", 3, 36, 12)
    freq = st.selectbox("Tần suất", ["M", "W", "Q"], 0)
    alpha = st.slider("Trọng số LRFMS (α)", 0.0, 1.0, 0.5, 0.05)
    seed = st.number_input("Random state", value=42, step=1)
    run_btn = st.button("🚀 Chạy mô hình", use_container_width=True)

@st.cache_data(show_spinner=False)
def load_clean_df(src):
    pre = LRFMSPreprocessor(src)
    df = pre.read_data()
    df = pre.remove_duplicates(df)
    df = pre.filter_invalid(df)
    df = pre.add_total_amount(df)
    df = pre.handle_outliers(df, "TotalAmount")
    return df

if run_btn:
    src_used = file_path if src_mode == "Path nội bộ" else up_file
    if not src_used:
        st.error("❌ Chưa chọn file dữ liệu.")
        st.stop()
    df_clean = load_clean_df(src_used)
    model = LRFMS_MTS(k=k, periods=periods, freq=freq, alpha=alpha, random_state=seed)
    res = model.fit(src_used)
    st.success("✅ Phân cụm hoàn tất!")

    m = res["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", f"{m['silhouette']:.4f}")
    c2.metric("Calinski–Harabasz", f"{m['calinski_harabasz']:.2f}")
    c3.metric("Davies–Bouldin", f"{m['davies_bouldin']:.4f}")

    fused_df = pd.DataFrame(res["fused_features"])
    fused_df["label"] = res["labels"]
    st.subheader("🧩 Ma trận đặc trưng hợp nhất (LRFMS + MTS)")
    st.dataframe(fused_df.head(10), use_container_width=True)
    st.download_button("💾 Tải CSV đặc trưng hợp nhất", fused_df.to_csv(index=False),
                       "fused_features.csv", mime="text/csv")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Song song", "📊 Trung bình cụm", "🕸️ Radar LRFMS", "🔥 Tương quan"])

    with tab1:
        if len(fused_df) > 0:
            fig = px.parallel_coordinates(fused_df, dimensions=[c for c in fused_df.columns if c != "label"],
                                          color="label", color_continuous_scale=px.colors.diverging.Tealrose)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu để hiển thị.")

    with tab2:
        if len(fused_df) > 0:
            mean_df = fused_df.groupby("label").mean().reset_index()
            fig = go.Figure()
            for _, r in mean_df.iterrows():
                fig.add_trace(go.Scatter(y=r.drop("label").values, mode="lines+markers",
                                         fill="tonexty", name=f"Cluster {int(r['label'])}"))
            fig.update_layout(template="plotly_dark", title="Giá trị trung bình theo cụm")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu hợp lệ.")

    with tab3:
        prof_df, lrfms_cols = model.lrfms_profile(df_clean)
        if not prof_df.empty:
            try:
                max_val = float(np.nanmax(prof_df[lrfms_cols].values))
                if not np.isfinite(max_val) or max_val == 0:
                    max_val = 1.0
            except Exception:
                max_val = 1.0
            fig = go.Figure()
            for _, r in prof_df.iterrows():
                fig.add_trace(go.Scatterpolar(r=r[lrfms_cols].values, theta=lrfms_cols,
                                              fill="toself", name=f"Cluster {int(r['label'])}"))
            fig.update_layout(template="plotly_dark",
                              polar=dict(radialaxis=dict(visible=True, range=[0, max_val])),
                              title="Radar LRFMS trung bình theo cụm")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không thể vẽ biểu đồ Radar.")

    with tab4:
        if len(fused_df) > 1:
            corr = fused_df.drop(columns="label").corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                            title="Ma trận tương quan giữa các đặc trưng")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Không đủ dữ liệu để vẽ Heatmap.")
