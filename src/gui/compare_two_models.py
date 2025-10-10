import sys, os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ====== FIX PATH ======
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.Kmeans_final import KMeans
from src.models.unsupervised.FuzzyCMean.Fuzzy_Cmeans_final import FuzzyCMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator


# ============================================================
# ✅ Caching dữ liệu và model
# ============================================================
@st.cache_resource(show_spinner=False)
def load_data():
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, rfm, rfm_scaled = pre.process()
    return X, rfm, rfm_scaled


@st.cache_resource(show_spinner=False)
def run_model(model_name, X, n_clusters):
    if model_name == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        labels = model.labels_
        centers = model.centroids
    elif model_name == "FuzzyCMeans":
        model = FuzzyCMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        labels = model.labels_
        centers = model.centers
    elif model_name == "ManualGMM":
        result = ManualGMM.run(X, n_components=n_clusters)
        labels = result["labels"]
        centers = result["means"]
    else:
        raise ValueError("Unknown model type.")
    return labels, centers


# ============================================================
# ✅ Giao diện Streamlit
# ============================================================
st.set_page_config(page_title="Compare Two Clustering Models", layout="wide")
st.title("🤖 So sánh hai mô hình phân cụm (Không chuẩn hóa, giữ giá trị gốc)")
st.markdown("---")

# 1️⃣ Load dữ liệu
with st.spinner("📦 Đang tải dữ liệu..."):
    X, rfm, rfm_scaled = load_data()
st.success(f"✅ Dữ liệu đã sẵn sàng — {X.shape[0]} khách hàng, {X.shape[1]} đặc trưng.")

# 2️⃣ Sidebar cấu hình
col1, col2 = st.sidebar.columns(2)
with col1:
    model_a = st.selectbox("🔹 Model A", ["KMeans", "FuzzyCMeans", "ManualGMM"], index=0)
    k_a = st.slider("Số cụm (K) — Model A", 2, 10, 3)
with col2:
    model_b = st.selectbox("🔸 Model B", ["KMeans", "FuzzyCMeans", "ManualGMM"], index=1)
    k_b = st.slider("Số cụm (K) — Model B", 2, 10, 3)

viz_type = st.sidebar.radio("🎨 Kiểu biểu đồ không gian", ["3D", "2D"])
chart_type = st.sidebar.radio(
    "📊 Kiểu biểu đồ 2D đánh giá",
    ["Line", "Bar", "Radar", "Parallel", "Heatmap"],
    index=0
)

run_btn = st.sidebar.button("🚀 Chạy mô hình")

# 3️⃣ Huấn luyện model
if run_btn or "cached_results" not in st.session_state:
    with st.spinner("🚀 Đang huấn luyện mô hình..."):
        labels_a, centers_a = run_model(model_a, X, k_a)
        labels_b, centers_b = run_model(model_b, X, k_b)

        evaluator = UnsupervisedEvaluator()
        results_a = evaluator.internal_metrics(X, labels_a)
        results_b = evaluator.internal_metrics(X, labels_b)
        match_ratio = (labels_a == labels_b).mean()

        # Cache lại kết quả
        st.session_state.cached_results = {
            "labels_a": labels_a,
            "labels_b": labels_b,
            "results_a": results_a,
            "results_b": results_b,
            "match_ratio": match_ratio
        }
else:
    st.info("👈 Chọn mô hình và nhấn **Chạy mô hình** để huấn luyện hoặc đổi kiểu biểu đồ 2D để xem lại kết quả.")

# 4️⃣ Hiển thị kết quả (nếu có cache)
if "cached_results" in st.session_state:
    labels_a = st.session_state.cached_results["labels_a"]
    labels_b = st.session_state.cached_results["labels_b"]
    results_a = st.session_state.cached_results["results_a"]
    results_b = st.session_state.cached_results["results_b"]
    match_ratio = st.session_state.cached_results["match_ratio"]

    st.markdown(f"<h3 style='color:#00BFFF'>🔸 Label Match Ratio: {match_ratio:.4f}</h3>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.write(f"📊 **{model_a} Metrics**")
        st.json(results_a)
    with colB:
        st.write(f"📊 **{model_b} Metrics**")
        st.json(results_b)

    # Visualization dữ liệu cụm
    df_viz = pd.DataFrame(rfm_scaled[:, 1:], columns=["Recency", "Frequency", "Monetary"])
    df_viz[model_a] = labels_a
    df_viz[model_b] = labels_b

    st.markdown("### 🎨 Kết quả phân cụm")
    tab1, tab2 = st.tabs([model_a, model_b])

    if viz_type == "3D":
        for tab, model, k, labels in zip(
            [tab1, tab2], [model_a, model_b], [k_a, k_b], [labels_a, labels_b]
        ):
            with tab:
                fig = px.scatter_3d(
                    df_viz, x="Recency", y="Frequency", z="Monetary",
                    color=df_viz[model].astype(str),
                    title=f"{model} (K={k})", opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        for tab, model, k, labels in zip(
            [tab1, tab2], [model_a, model_b], [k_a, k_b], [labels_a, labels_b]
        ):
            with tab:
                fig = px.scatter(
                    df_viz, x="Recency", y="Monetary",
                    color=df_viz[model].astype(str),
                    title=f"{model} (2D view, K={k})", opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)

    # 5️⃣ So sánh các chỉ số
    st.markdown("### 📈 So sánh các chỉ số đánh giá")

    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
    values_a = [results_a[m] for m in metrics]
    values_b = [results_b[m] for m in metrics]

    df_metrics = pd.DataFrame({
        "Metric": metrics,
        model_a: values_a,
        model_b: values_b
    })

    # ====== LINE CHART ======
    if chart_type == "Line":
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=metrics, y=values_a, mode='lines+markers',
            name=model_a, line=dict(width=3)
        ))
        fig_line.add_trace(go.Scatter(
            x=metrics, y=values_b, mode='lines+markers',
            name=model_b, line=dict(width=3)
        ))
        fig_line.update_layout(
            title=f"📈 Biểu đồ đường — So sánh {model_a} và {model_b}",
            xaxis_title="Chỉ số",
            yaxis_title="Giá trị thực",
            template="plotly_dark"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ====== BAR CHART ======
    elif chart_type == "Bar":
        df_melt = df_metrics.melt(id_vars="Metric", var_name="Model", value_name="Score")
        fig_bar = px.bar(
            df_melt, x="Metric", y="Score", color="Model",
            barmode="group", text_auto=".3f",
            title=f"📊 Biểu đồ cột — So sánh chỉ số giữa {model_a} và {model_b}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ====== RADAR ======
    elif chart_type == "Radar":
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=values_a, theta=metrics, fill='toself', name=model_a))
        fig_radar.add_trace(go.Scatterpolar(r=values_b, theta=metrics, fill='toself', name=model_b))
        fig_radar.update_layout(
            title=f"🕸️ Biểu đồ mạng nhện (Radar) — {model_a} vs {model_b}",
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ====== PARALLEL ======
    elif chart_type == "Parallel":
        df_para = pd.DataFrame(np.vstack([values_a, values_b]), columns=metrics)
        df_para["ModelCode"] = [0, 1]
        fig_para = px.parallel_coordinates(
            df_para,
            color="ModelCode",
            dimensions=metrics,
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={"ModelCode": "Model"},
            title=f"📊 Biểu đồ song song (Parallel Coordinates): {model_a} vs {model_b}"
        )
        st.plotly_chart(fig_para, use_container_width=True)

    # ====== HEATMAP ======
    elif chart_type == "Heatmap":
        df_heat = pd.DataFrame({
            model_a: values_a,
            model_b: values_b
        }, index=metrics)
        fig_heat = px.imshow(
            df_heat.T, text_auto=".3f",
            color_continuous_scale="viridis",
            title=f"🔥 Heatmap — So sánh giá trị giữa {model_a} và {model_b}"
        )
        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)
