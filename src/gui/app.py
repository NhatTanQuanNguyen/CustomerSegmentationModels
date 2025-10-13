# src/gui/app.py
import sys, os, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.models.unsupervised.FuzzyCMean.main import FuzzyCMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.models.unsupervised.LatentClassModels.main import LatentClass
from src.models.unsupervised.HierarchicalClustering.main import HierarchicalWardManual

try:
    from src.evaluation.unsupervised_eval import UnsupervisedEvaluator  # nếu tách thư mục con
except ModuleNotFoundError:
    from src.evaluation.unsupervised_eval import UnsupervisedEvaluator               # nếu để phẳng

st.set_page_config(page_title="So sánh mô hình không giám sát", layout="wide")
st.title("So sánh & đánh giá mô hình không giám sát")
st.markdown("---")

@st.cache_resource(show_spinner=False)
def load_rfm_data():
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, rfm, _ = pre.process()
    return X, pd.DataFrame(rfm, columns=["CustomerID", "Recency", "Frequency", "Monetary"])

with st.spinner("Đang xử lý dữ liệu..."):
    X, rfm = load_rfm_data()
st.success(f"Dữ liệu RFM sẵn sàng: {X.shape[0]} KH, {X.shape[1]} đặc trưng")

st.sidebar.header("Cấu hình mô hình")

ALL_MODELS = ["KMeans", "FuzzyCMeans", "ManualGMM", "LatentClass", "HierarchicalWardManual"]
models = st.sidebar.multiselect("Chọn mô hình", ALL_MODELS, default=ALL_MODELS)

k_values = {m: st.sidebar.slider(f"{m} — K", 2, 15, 3) for m in models if m != "HierarchicalWardManual"}

st.sidebar.subheader("Hierarchical (Ward)")
auto_k_hier = st.sidebar.checkbox("Tự đề xuất K (largest jump)", value=True)
hier_cut_mode = st.sidebar.radio("Cách cắt", ["Theo K", "Theo ngưỡng khoảng cách"], index=0)
cut_height_manual = st.sidebar.number_input("Ngưỡng khoảng cách (height)", min_value=0.0, value=0.0, step=0.5, format="%.3f")

viz_type = st.sidebar.radio("Hiển thị cụm", ["3D", "2D"], index=0)
chart_type = st.sidebar.selectbox("Biểu đồ chỉ số", ["Heatmap", "Bar", "Line"], index=0)
run_btn = st.sidebar.button("Chạy mô hình")

@st.cache_data(show_spinner=False)
def kmeans_curve_cached(X):
    ks = list(range(2, 11))
    sse = [KMeansNumpy.fit_k(X, k)["metrics"]["curve"][0] for k in ks]
    return ks, sse

@st.cache_data(show_spinner=False)
def fcm_curve_cached(X):
    ks = list(range(2, 11))
    fpc = [FuzzyCMeans.fit_k(X, k)["metrics"]["fpc"] for k in ks]
    return ks, fpc

@st.cache_data(show_spinner=False)
def gmm_curve_cached(X):
    ks = list(range(2, 11))
    bic = [ManualGMM.fit_k(X, k)["metrics"]["bic"] for k in ks]
    return ks, bic

@st.cache_data(show_spinner=False)
def lca_curve_cached(X):
    ks, bics = [], []
    for k in range(2, 11):
        _, bic = LatentClass.bic_for_k(X, k, q=5, max_iter=200, tol=1e-6, random_state=42)
        ks.append(k); bics.append(bic)
    return ks, bics

@st.cache_data(show_spinner=False)
def hier_curves_cached(X):
    d = HierarchicalWardManual.dendrogram_coords(X, standardize=True, truncate_mode=None, p=30)
    ks, sses = HierarchicalWardManual.sse_vs_k(X, k_min=2, k_max=11, standardize=True)
    k_auto, cut_h, _, _ = HierarchicalWardManual.suggest_k_by_jump(X, standardize=True)
    return d, (ks, sses), (k_auto, cut_h)

def plotly_dendrogram_from_scipy(d):
    icoord, dcoord = d["icoord"], d["dcoord"]
    traces = [go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1)) for xs, ys in zip(icoord, dcoord)]
    fig = go.Figure(traces)
    fig.update_layout(
        title="Hierarchical (Ward) — Dendrogram",
        xaxis_title="Samples (trục cây)",
        yaxis_title="Khoảng cách hợp nhất",
        showlegend=False, height=500,
    )
    return fig

def show_model_curves(X):
    st.markdown("### Biểu đồ tham khảo tìm K tối ưu (tùy mô hình)")
    tabs = st.tabs(["KMeans", "FuzzyCMeans", "ManualGMM", "LatentClass", "HierarchicalWardManual"])

    with tabs[0]:
        ks, sse = kmeans_curve_cached(X)
        fig = px.line(x=ks, y=sse, markers=True, title="KMeans — SSE vs K (Elbow)")
        fig.update_xaxes(title="K"); fig.update_yaxes(title="SSE (↓ tốt hơn)")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        ks, fpc = fcm_curve_cached(X)
        fig = px.line(x=ks, y=fpc, markers=True, title="FuzzyCMeans — FPC vs K (↑ tốt hơn)")
        fig.update_xaxes(title="K"); fig.update_yaxes(title="FPC")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        ks, bic = gmm_curve_cached(X)
        fig = px.line(x=ks, y=bic, markers=True, title="ManualGMM — BIC vs K (↓ tốt hơn)")
        fig.update_xaxes(title="K"); fig.update_yaxes(title="BIC")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        ks, bics = lca_curve_cached(X)
        fig = px.line(x=ks, y=bics, markers=True, title="LatentClass — BIC vs K (↓ tốt hơn)")
        fig.update_xaxes(title="K"); fig.update_yaxes(title="BIC")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        d, (ks_ref, sses), (k_auto, cut_h) = hier_curves_cached(X)
        fig = plotly_dendrogram_from_scipy(d)
        fig.add_hline(y=cut_h, line_dash="dash", line_width=2)
        fig.update_layout(title=f"Hierarchical (Ward) — Dendrogram (gợi ý K ≈ {k_auto}, cut@{cut_h:.3f})")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(x=ks_ref, y=sses, markers=True, title="Hierarchical Ward — SSE vs K (tham khảo)")
        fig2.update_xaxes(title="K"); fig2.update_yaxes(title="SSE")
        st.plotly_chart(fig2, use_container_width=True)

def _run_one_model(name, X, k, hier_cfg):
    start = time.time()
    if name == "KMeans":
        out = KMeansNumpy.fit_k(X, k)
    elif name == "FuzzyCMeans":
        out = FuzzyCMeans.fit_k(X, k)
    elif name == "ManualGMM":
        out = ManualGMM.fit_k(X, k)  # AIC/BIC ở trong model
    elif name == "LatentClass":
        out = LatentClass.fit_k(X, k, q=5, max_iter=200, tol=1e-6, random_state=42)
    elif name == "HierarchicalWardManual":
        cut_mode, auto_k, cut_height = hier_cfg
        if cut_mode == "Theo ngưỡng khoảng cách":
            if auto_k:
                k_auto, cut_h, _, _ = HierarchicalWardManual.suggest_k_by_jump(X, standardize=True)
                out = HierarchicalWardManual.fit_distance(X, cut_h, standardize=True)
                out["metrics"]["k_auto"] = int(k_auto)
            else:
                out = HierarchicalWardManual.fit_distance(X, cut_height, standardize=True)
        else:
            if auto_k:
                k_auto, _, _, _ = HierarchicalWardManual.suggest_k_by_jump(X, standardize=True)
                k = int(max(2, k_auto))
            out = HierarchicalWardManual.fit_k(X, k, standardize=True)
    else:
        out = None
    elapsed = round(time.time() - start, 3)
    return name, out, elapsed

def run_models(X, rfm, models, k_values):
    evaluator = UnsupervisedEvaluator(X)
    results = {}
    progress = st.progress(0)

    tasks = []
    for m in models:
        if m == "HierarchicalWardManual":
            tasks.append((m, X, k_values.get(m, 3), (hier_cut_mode, auto_k_hier, cut_height_manual)))
        else:
            tasks.append((m, X, k_values.get(m, 3), None))

    # ThreadPool: tránh copy lớn của X giữa processes, ổn trên Windows/Streamlit.
    done = 0
    with ThreadPoolExecutor(max_workers=min(5, len(tasks))) as ex:
        futures = [ex.submit(_run_one_model, *t) for t in tasks]
        for fut in as_completed(futures):
            name, result, elapsed = fut.result()
            labels = result["labels"]
            metrics = evaluator.evaluate(labels)
            metrics.update(result.get("metrics", {}))
            metrics["time"] = elapsed
            results[name] = {"labels": labels, "metrics": metrics, "extra": result}
            done += 1
            progress.progress(done / len(tasks))

    return results

def show_metrics(results):
    st.markdown("### Bảng tổng hợp chỉ số")
    rows = []
    for m, r in results.items():
        row = {"Model": m}
        for k, v in r["metrics"].items():
            if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                row[k] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df.style.format(precision=4))
    return df

def show_metrics_chart(df, chart_type):
    st.markdown("## Biểu đồ so sánh chỉ số")
    if df.empty:
        st.info("Không có chỉ số để vẽ.")
        return
    if chart_type == "Heatmap":
        fig = px.imshow(df.T, text_auto=".2f", aspect="auto")
        fig.update_layout(height=600)
    elif chart_type == "Bar":
        fig = px.bar(df.T, barmode="group")
    else:
        fig = px.line(df.T, markers=True)
    st.plotly_chart(fig, use_container_width=True)

def show_clusters(results, X, viz_type):
    st.markdown("## Phân cụm trực quan")
    df = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"])
    tabs = st.tabs(list(results.keys()))
    for tab, model in zip(tabs, results.keys()):
        with tab:
            labels = results[model]["labels"].astype(str)
            df_plot = df.copy(); df_plot["Cluster"] = labels

            if model == "HierarchicalWardManual":
                st.markdown("**Dendrogram (Ward)**")
                d, _, (k_auto, cut_h) = hier_curves_cached(X)
                fig_d = plotly_dendrogram_from_scipy(d)
                # nếu có cut_height trong metrics -> kẻ đường cắt
                cut_in_metrics = results[model]["metrics"].get("cut_height", None)
                if cut_in_metrics is not None:
                    fig_d.add_hline(y=cut_in_metrics, line_dash="dash", line_width=2)
                else:
                    # nếu cắt theo K, vẽ gợi ý auto
                    fig_d.add_hline(y=cut_h, line_dash="dash", line_width=1)
                st.plotly_chart(fig_d, use_container_width=True)

            if viz_type == "3D":
                fig = px.scatter_3d(df_plot, x="Recency", y="Frequency", z="Monetary",
                                    color="Cluster", title=f"{model} (3D)")
            else:
                fig = px.scatter(df_plot, x="Recency", y="Monetary",
                                 color="Cluster", title=f"{model} (2D)")
            st.plotly_chart(fig, use_container_width=True)

show_model_curves(X)
if run_btn:
    results = run_models(X, rfm, models, k_values)
    df = show_metrics(results)
    show_metrics_chart(df, chart_type)
    show_clusters(results, X, viz_type)
else:
    st.info("Chọn mô hình và nhấn **Chạy mô hình** để bắt đầu.")
