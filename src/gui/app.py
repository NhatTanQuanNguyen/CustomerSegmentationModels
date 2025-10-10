import sys, os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ==========================================
# 🧩 FIX IMPORT PATH
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.Kmeans_final import KMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator


# ==========================================
# ⚙️ CONFIG
# ==========================================
st.set_page_config(page_title="Unsupervised Comparison (Optimized OOP)", layout="wide")
st.title("🧠 So sánh mô hình phân cụm (Hướng đối tượng + Caching thông minh)")
st.markdown("---")


# ==========================================
# 📦 CLASS 1 — Data Manager (cache dữ liệu)
# ==========================================
class DataManager:
    """Chịu trách nhiệm load và cache dữ liệu RFM"""

    @st.cache_data(show_spinner=True)
    def load_data(_, file_path):
        pre = RFMPreprocessor(file_path)
        X, rfm, rfm_scaled = pre.process()
        return X, rfm, rfm_scaled


# ==========================================
# ⚙️ CLASS 2 — Model Runner
# ==========================================
class ModelRunner:
    """Huấn luyện & cache mô hình (KMeans, GMM)"""

    def __init__(self):
        self.results = {}

    def run_kmeans(self, X, k):
        start = time.time()
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        self.results["KMeans"] = {
            "name": f"KMeans (K={k})",
            "labels": model.labels_,
            "centroids": model.centroids,
            "runtime": round(time.time() - start, 2),
        }
        return self.results["KMeans"]

    def run_gmm(self, X, k):
        start = time.time()
        result = ManualGMM.run(X, n_components=k)
        result.update({
            "name": f"Manual GMM (K={k})",
            "runtime": round(time.time() - start, 2),
        })
        self.results["Manual GMM"] = result
        return result

    def run_parallel(self, X, jobs):
        """Chạy nhiều mô hình song song"""
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {}
            for name, k in jobs:
                if name == "KMeans":
                    futures[executor.submit(self.run_kmeans, X, k)] = name
                elif name == "Manual GMM":
                    futures[executor.submit(self.run_gmm, X, k)] = name

            progress = st.progress(0)
            done = 0
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    st.success(f"✅ {res['name']} hoàn tất (⏱ {res['runtime']}s)")
                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy {name}: {e}")
                done += 1
                progress.progress(done / len(jobs))
        return self.results


# ==========================================
# 🎨 CLASS 3 — Visualizer
# ==========================================
class Visualizer:
    """Sinh biểu đồ phân cụm tùy chọn"""

    def __init__(self, rfm_scaled):
        if isinstance(rfm_scaled, np.ndarray):
            self.df_base = pd.DataFrame(rfm_scaled[:, -3:], columns=["Recency", "Frequency", "Monetary"])
        else:
            self.df_base = rfm_scaled[["Recency", "Frequency", "Monetary"]].copy()

    def render_chart(self, results, chart_type):
        """Render biểu đồ cho tất cả mô hình"""
        df_viz = self.df_base.copy()
        for name, res in results.items():
            df_viz[name] = res["labels"]

        for name, res in results.items():
            st.subheader(f"{res['name']} Visualization")

            if chart_type == "3D Scatter":
                fig = px.scatter_3d(
                    df_viz, x="Recency", y="Frequency", z="Monetary",
                    color=df_viz[name].astype(str),
                    title=f"{res['name']}",
                    opacity=0.7
                )
            elif chart_type == "2D Scatter":
                fig = px.scatter(
                    df_viz, x="Recency", y="Monetary",
                    color=df_viz[name].astype(str),
                    title=f"{res['name']} (Recency vs Monetary)"
                )
            else:
                fig = px.density_heatmap(
                    df_viz, x="Recency", y="Monetary",
                    z=df_viz[name], color_continuous_scale="Viridis",
                    title=f"{res['name']} Heatmap"
                )
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 🚀 STREAMLIT UI
# ==========================================
st.sidebar.header("📂 Dữ liệu")
file_path = "data/raw/noLabel/Online Retail.xlsx"

try:
    data_manager = DataManager()
    X, rfm, rfm_scaled = data_manager.load_data(file_path)
    st.sidebar.success(f"✅ Dữ liệu đã tải ({X.shape[0]} KH, {X.shape[1]} đặc trưng)")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi khi load dữ liệu: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Cấu hình mô hình")

algo_choice = st.sidebar.multiselect(
    "🧩 Chọn thuật toán muốn chạy",
    ["KMeans", "Manual GMM"],
    default=st.session_state.get("algo_choice", ["KMeans", "Manual GMM"])
)
st.session_state["algo_choice"] = algo_choice

chart_type = st.sidebar.radio(
    "📊 Chọn loại biểu đồ",
    ["3D Scatter", "2D Scatter", "Heatmap"],
    index=st.session_state.get("chart_index", 0),
    key="chart_type"
)
st.session_state["chart_index"] = ["3D Scatter", "2D Scatter", "Heatmap"].index(chart_type)

colK1, colK2 = st.sidebar.columns(2)
k_kmeans = colK1.slider("K cho KMeans", 2, 10, st.session_state.get("k_kmeans", 3), 1, key="k_kmeans")
k_gmm = colK2.slider("K cho GMM", 2, 10, st.session_state.get("k_gmm", 3), 1, key="k_gmm")

run_button = st.sidebar.button("🚀 Chạy mô hình")

# ==========================================
# 🧠 MAIN EXECUTION
# ==========================================
if run_button:
    st.subheader("🚀 Đang chạy mô hình song song...")

    runner = ModelRunner()
    jobs = []
    if "KMeans" in algo_choice:
        jobs.append(("KMeans", k_kmeans))
    if "Manual GMM" in algo_choice:
        jobs.append(("Manual GMM", k_gmm))

    results = runner.run_parallel(X, jobs)

    # ==== Hiển thị tâm cụm ====
    for name, res in results.items():
        if "centroids" in res:
            df_c = pd.DataFrame(np.round(res["centroids"], 4),
                                columns=["Recency", "Frequency", "Monetary"])
        else:
            df_c = pd.DataFrame(np.round(res["means"], 4),
                                columns=["Recency", "Frequency", "Monetary"])
        st.markdown(f"#### 📍 {res['name']} — Trung bình cụm")
        st.dataframe(df_c, use_container_width=True)

    # ==== Đánh giá ====
    if len(results) == 2:
        st.markdown("### 🔍 So sánh mô hình")
        evalr = UnsupervisedEvaluator()
        eval_results = evalr.compare_models(
            X,
            np.array(results["KMeans"]["labels"]),
            np.array(results["Manual GMM"]["labels"]),
            "KMeans", "Manual GMM"
        )
        st.json(eval_results)

    # ==== Biểu đồ ====
    visualizer = Visualizer(rfm_scaled)
    visualizer.render_chart(results, chart_type)

else:
    st.info("👈 Chọn thuật toán, điều chỉnh K, rồi nhấn **Chạy mô hình** để bắt đầu.")
