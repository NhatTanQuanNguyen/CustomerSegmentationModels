import sys, os, time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from fcmeans import FCM
from kneed import KneeLocator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.Kmeans_final import KMeans
from src.models.unsupervised.FuzzyCMean.Fuzzy_Cmeans_final import FuzzyCMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator


# =====================================================
# Cache data
# =====================================================
@st.cache_resource(show_spinner=False)
def load_rfm_data():
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    return pre.process()


@st.cache_resource(show_spinner=False)
def run_model(name, X, n_clusters=None, eps=None):
    start = time.time()

    if name == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        labels, centers = model.labels_, model.centroids

    elif name == "FuzzyCMeans":
        model = FuzzyCMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        labels, centers = model.labels_, model.centers

    elif name == "ManualGMM":
        result = ManualGMM.run(X, n_components=n_clusters)
        labels, centers = result["labels"], result["means"]

    elif name == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X)
        centers = None

    elif name == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X)
        centers = None

    else:
        raise ValueError(f"Model {name} không được hỗ trợ.")

    duration = time.time() - start
    return labels, centers, duration


# =====================================================
# Auto select parameters per model
# =====================================================
def auto_select_k_by_model(model_name, X, k_min=2, k_max=10):
    Ks = range(k_min, k_max + 1)
    best_k, fig = 3, None
    scores = []

    if model_name == "KMeans":
        wcss = []
        for k in Ks:
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(X)
            wcss.append(model.inertia_)
        kl = KneeLocator(Ks, wcss, curve="convex", direction="decreasing")
        best_k = kl.elbow or 3
        fig = px.line(x=list(Ks), y=wcss, markers=True, title="📉 Elbow (KMeans)")
        fig.add_vline(x=best_k, line_dash="dash", line_color="red")

    elif model_name == "FuzzyCMeans":
        fpc = []
        for k in Ks:
            model = FCM(n_clusters=k, random_state=42)
            model.fit(X)
            fpc.append(model.fpc)
        best_k = Ks[np.argmax(fpc)]
        fig = px.line(x=list(Ks), y=fpc, markers=True, title="📈 FPC (Fuzzy C-Means)")
        fig.add_vline(x=best_k, line_dash="dash", line_color="red")

    elif model_name == "ManualGMM":
        bics = []
        for k in Ks:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(X)
            bics.append(gmm.bic(X))
        best_k = Ks[np.argmin(bics)]
        fig = px.line(x=list(Ks), y=bics, markers=True, title="📉 BIC (GMM)")
        fig.add_vline(x=best_k, line_dash="dash", line_color="red")

    elif model_name == "Agglomerative":
        Z = linkage(X, method="ward")
        fig = go.Figure()
        dendrogram(Z, no_plot=False)
        best_k = 3
        fig.update_layout(title="🌳 Dendrogram (Agglomerative Clustering)")

    elif model_name == "DBSCAN":
        neigh = NearestNeighbors(n_neighbors=5)
        nbrs = neigh.fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = np.sort(distances[:, 4])
        best_eps = np.percentile(distances, 95)
        fig = px.line(y=distances, title="📏 k-distance plot (DBSCAN)")
        fig.add_hline(y=best_eps, line_dash="dash", line_color="red")
        return best_eps, fig, "eps"

    return best_k, fig, "k"


# =====================================================
# Streamlit layout
# =====================================================
st.set_page_config(page_title="Unsupervised Comparison Dashboard", layout="wide")
st.title("🎛 Bảng điều khiển so sánh thuật toán không giám sát")
st.markdown("---")

# Load data
with st.spinner("📦 Đang tải dữ liệu RFM..."):
    X, rfm, rfm_scaled = load_rfm_data()
st.success(f"✅ Dữ liệu RFM có {X.shape[0]} mẫu, {X.shape[1]} thuộc tính")

uploaded = st.file_uploader("📂 Tải lên file CSV khác (tùy chọn)", type=["csv"])
if uploaded is not None:
    df_user = pd.read_csv(uploaded)
    X = df_user.values
    st.info(f"Đã tải {df_user.shape[0]} dòng, {df_user.shape[1]} cột")

# Sidebar config
st.sidebar.header("⚙️ Cấu hình mô hình")
models = st.sidebar.multiselect(
    "Chọn mô hình cần so sánh",
    ["KMeans", "FuzzyCMeans", "ManualGMM", "DBSCAN", "Agglomerative"],
    default=["KMeans", "FuzzyCMeans", "ManualGMM"],
)
viz_type = st.sidebar.radio("🎨 Kiểu hiển thị cụm", ["3D", "2D"])
metric_chart = st.sidebar.selectbox(
    "📊 Kiểu biểu đồ chỉ số",
    ["Line", "Bar", "Radar", "Parallel", "Heatmap"],
)
run_btn = st.sidebar.button("🚀 Chạy tất cả")

# =====================================================
# Train and evaluate
# =====================================================
if run_btn:
    evaluator = UnsupervisedEvaluator()
    st.info("⏳ Đang chạy mô hình...")
    progress = st.progress(0)
    all_results = {}

    for i, name in enumerate(models):
        st.write(f"🔍 **{name}** — đang chọn thông số tối ưu...")
        param, fig_param, param_type = auto_select_k_by_model(name, X)
        st.plotly_chart(fig_param, use_container_width=True)
        st.success(f"{name}: {param_type} tối ưu = **{round(param, 3)}**")

        if name == "DBSCAN":
            labels, centers, duration = run_model(name, X, eps=param)
        else:
            labels, centers, duration = run_model(name, X, n_clusters=int(param))

        metrics = evaluator.internal_metrics(X, labels)
        metrics["time"] = round(duration, 3)
        metrics[param_type] = round(param, 3)
        all_results[name] = {"labels": labels, "metrics": metrics}
        progress.progress((i + 1) / len(models))

    st.session_state["results"] = all_results
    st.success("🎉 Hoàn tất!")

# =====================================================
# Visualization and comparison
# =====================================================
if "results" in st.session_state:
    results = st.session_state["results"]
    metrics_df = pd.DataFrame(
        [{"Model": m, **r["metrics"]} for m, r in results.items()]
    ).set_index("Model")

    st.markdown("## 📈 Bảng tổng hợp chỉ số")
    st.dataframe(metrics_df.style.format(precision=4))

    # So sánh các chỉ số
    if metric_chart == "Line":
        fig = px.line(metrics_df.T, markers=True)
    elif metric_chart == "Bar":
        fig = px.bar(metrics_df.T, barmode="group")
    elif metric_chart == "Radar":
        fig = go.Figure()
        for model in metrics_df.index:
            fig.add_trace(go.Scatterpolar(
                r=metrics_df.loc[model].values,
                theta=metrics_df.columns,
                fill="toself",
                name=model))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    elif metric_chart == "Parallel":
        fig = px.parallel_coordinates(metrics_df.reset_index(), color="time")
    elif metric_chart == "Heatmap":
        fig = px.imshow(metrics_df.T, text_auto=".3f")
    else:
        fig = None

    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap label match giữa các model
    st.markdown("## 🔍 So khớp nhãn giữa các mô hình")
    models_list = list(results.keys())
    if len(models_list) >= 2:
        labels_matrix = np.zeros((len(models_list), len(models_list)))
        for i, m1 in enumerate(models_list):
            for j, m2 in enumerate(models_list):
                l1, l2 = results[m1]["labels"], results[m2]["labels"]
                if len(l1) == len(l2):
                    labels_matrix[i, j] = np.mean(l1 == l2)
        df_heat = pd.DataFrame(labels_matrix, index=models_list, columns=models_list)
        fig_h = px.imshow(df_heat, text_auto=".2f", title="Heatmap tương đồng nhãn giữa các mô hình")
        st.plotly_chart(fig_h, use_container_width=True)

    # Visualization cụm
    st.markdown("## 🎨 Hiển thị phân cụm")
    df_viz = pd.DataFrame(X, columns=["Recency", "Frequency", "Monetary"])
    tabs = st.tabs(list(results.keys()))
    for tab, model in zip(tabs, results.keys()):
        with tab:
            labels = results[model]["labels"]
            df_viz["Cluster"] = labels.astype(str)
            if viz_type == "3D":
                fig = px.scatter_3d(df_viz, x="Recency", y="Frequency", z="Monetary",
                                    color="Cluster", title=f"{model}", opacity=0.7)
            else:
                fig = px.scatter(df_viz, x="Recency", y="Monetary",
                                 color="Cluster", title=f"{model} (2D)", opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)
