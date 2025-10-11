import streamlit as st
import sys, os, time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.models.unsupervised.FuzzyCMean.main import FuzzyCMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator

st.set_page_config(page_title="So sánh mô hình không giám sát", layout="wide")
st.title("🧠 So sánh & Đánh giá mô hình không giám sát")
st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_rfm_data():
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, rfm, _ = pre.process()
    return X, pd.DataFrame(rfm, columns=["CustomerID", "Recency", "Frequency", "Monetary"])

with st.spinner("🔄 Đang xử lý dữ liệu..."):
    X, rfm = load_rfm_data()

st.success(f"✅ Dữ liệu RFM sẵn sàng: {X.shape[0]} khách hàng, {X.shape[1]} đặc trưng")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Cấu hình mô hình")

models = st.sidebar.multiselect(
    "Chọn mô hình cần chạy",
    ["KMeans", "FuzzyCMeans", "ManualGMM"],
    default=["KMeans", "FuzzyCMeans", "ManualGMM"]
)

k_values = {m: st.sidebar.slider(f"{m} — Chọn K", 2, 15, 3) for m in models}
viz_type = st.sidebar.radio("🎨 Kiểu hiển thị cụm", ["3D", "2D"], index=0)
chart_type = st.sidebar.selectbox("📊 Kiểu biểu đồ chỉ số", ["Heatmap", "Bar", "Line"], index=0)
run_btn = st.sidebar.button("🚀 Chạy mô hình")

# ==================== BIỂU ĐỒ CHỌN K ====================
def show_model_curves(X):
    st.markdown("## 🔍 Biểu đồ tham khảo tìm K tối ưu")
    tabs = st.tabs(["KMeans", "FuzzyCMeans", "ManualGMM"])

    with tabs[0]:
        sse = []
        for k in range(2, 11):
            sse.append(KMeansNumpy.fit_k(X, k)["metrics"]["curve"][0])
        fig = px.line(x=list(range(2, 11)), y=sse, markers=True, title="KMeans — SSE vs K")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fpcs = []
        for k in range(2, 11):
            fpcs.append(FuzzyCMeans.fit_k(X, k)["metrics"]["fpc"])
        fig = px.line(x=list(range(2, 11)), y=fpcs, markers=True, title="FuzzyCMeans — FPC vs K")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        lls = []
        for k in range(2, 11):
            lls.append(ManualGMM.fit_k(X, k)["metrics"]["log_likelihood"])
        fig = px.line(x=list(range(2, 11)), y=lls, markers=True, title="ManualGMM — Log-Likelihood vs K")
        st.plotly_chart(fig, use_container_width=True)

# ==================== CHẠY MÔ HÌNH ====================
def run_models(X, rfm, models, k_values):
    evaluator = UnsupervisedEvaluator(X)
    results = {}
    progress = st.progress(0)

    for i, name in enumerate(models):
        k = k_values[name]
        start = time.time()

        if name == "KMeans":
            result = KMeansNumpy.fit_k(X, k)
        elif name == "FuzzyCMeans":
            result = FuzzyCMeans.fit_k(X, k)
        else:
            result = ManualGMM.fit_k(X, k)

        elapsed = round(time.time() - start, 3)
        labels = result["labels"]
        metrics = evaluator.evaluate(labels)
        metrics.update(result["metrics"])
        metrics["best_k"] = k
        metrics["time"] = elapsed
        results[name] = {"labels": labels, "metrics": metrics}
        progress.progress((i+1)/len(models))

    return results

# ==================== HIỂN THỊ ====================
def show_metrics(results):
    st.markdown("## 📈 Bảng tổng hợp chỉ số")
    df = pd.DataFrame([{"Model": m, **r["metrics"]} for m, r in results.items()]).set_index("Model")
    st.dataframe(df.style.format(precision=4))
    return df

def show_metrics_chart(df, chart_type):
    st.markdown("## 📊 Biểu đồ so sánh chỉ số")
    if chart_type == "Heatmap":
        fig = px.imshow(df.T, text_auto=".2f", aspect="auto")
        fig.update_layout(width=950, height=600)
    elif chart_type == "Bar":
        fig = px.bar(df.T, barmode="group")
    else:
        fig = px.line(df.T, markers=True)
    st.plotly_chart(fig, use_container_width=True)

def show_clusters(results, X, viz_type):
    st.markdown("## 🎨 Phân cụm trực quan")
    df = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"])
    tabs = st.tabs(list(results.keys()))
    for tab, model in zip(tabs, results.keys()):
        with tab:
            df["Cluster"] = results[model]["labels"].astype(str)
            fig = px.scatter_3d(df, x="Recency", y="Frequency", z="Monetary",
                                color="Cluster", title=f"{model} ({viz_type})") if viz_type=="3D" else \
                  px.scatter(df, x="Recency", y="Monetary", color="Cluster", title=f"{model} (2D)")
            st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN ====================
show_model_curves(X)
if run_btn:
    results = run_models(X, rfm, models, k_values)
    df = show_metrics(results)
    show_metrics_chart(df, chart_type)
    show_clusters(results, X, viz_type)
else:
    st.info("👈 Chọn mô hình và nhấn **Chạy mô hình** để bắt đầu.")
