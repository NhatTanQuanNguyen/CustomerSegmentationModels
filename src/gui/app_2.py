import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.main import KMeansNumpy
from src.models.supervised.SuperVectorMachine.main import MultiClassSVM
from src.models.supervised.RandomForest.main import RandomForestManual
from src.evaluation.supervised_evel import evaluate_supervised
from src.utils.profiler import profile_block

st.set_page_config(page_title="So sánh mô hình tay", layout="wide")
st.title("So sánh SVM và Random Forest")

with st.sidebar:
    source = st.radio("Nguồn dữ liệu", ["Path nội bộ", "Upload Excel"], index=0)
    default_path = "data/raw/noLabel/Online Retail.xlsx"
    file_path = st.text_input("Đường dẫn Excel", value=default_path) if source == "Path nội bộ" else None
    up_file = st.file_uploader("Chọn file Excel", type=["xlsx", "xls"]) if source != "Path nội bộ" else None

    k = st.slider("K (KMeans)", 2, 12, 4)
    test_size = st.slider("Tỷ lệ test", 0.1, 0.5, 0.3, step=0.05)
    rs = st.number_input("Random state", value=42, step=1)

    model_choice = st.radio("Chọn mô hình", ["SVM RBF tay", "Random Forest tay", "So sánh cả hai"], index=2)

    C, gamma, calib_split = 2.0, 0.5, 0.2
    n_estimators, max_depth = 100, 5

    st.markdown("---")
    if "SVM" in model_choice:
        C = st.number_input("C", value=2.0, step=0.1)
        gamma = st.number_input("gamma", value=0.5, step=0.1)
        calib_split = st.slider("Calibration split (Platt)", 0.0, 0.5, 0.2, step=0.05)
    if "Forest" in model_choice:
        n_estimators = st.number_input("Số cây", value=100, step=10)
        max_depth = st.number_input("Độ sâu tối đa", value=5, step=1)

    normalize_cm = st.checkbox("Normalize confusion matrix (%)", value=True)
    run_btn = st.button("Huấn luyện & Đánh giá", use_container_width=True)

@st.cache_resource(show_spinner=True)
def load_rfm(src):
    pre = RFMPreprocessor(src)
    X_scaled, _, _ = pre.process()
    return X_scaled

def plot_cm(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm / np.clip(row_sum, 1e-12, None) * 100.0)
        df = pd.DataFrame(np.round(cm, 1), index=class_names, columns=class_names)
        fig = px.imshow(df, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="%"))
    else:
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        fig = px.imshow(df, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig

if run_btn:
    if source == "Path nội bộ":
        X_scaled = load_rfm(file_path)
    else:
        if up_file is None:
            st.error("Hãy upload file Excel.")
            st.stop()
        X_scaled = load_rfm(up_file)

    kmeans = KMeansNumpy.fit_k(X_scaled, k=int(k), random_state=int(rs))
    y = kmeans["labels"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=float(test_size), random_state=int(rs), stratify=y)
    results = {}

    if model_choice in ["SVM RBF tay", "So sánh cả hai"]:
        model_svm = MultiClassSVM(
            C=float(C),
            gamma=float(gamma),
            max_iter=500,
            calib_split=float(calib_split),
            random_state=int(rs)
        )

        with profile_block() as prof:
            model_svm.fit(X_tr, y_tr)
            train_stats = prof()

        metrics = evaluate_supervised(
            model_svm, X_tr, y_tr, X_te, y_te, fit=False
        )

        metrics.update(train_stats)
        results["SVM RBF"] = metrics

    if model_choice in ["Random Forest tay", "So sánh cả hai"]:
        model_rf = RandomForestManual(
            n_estimators=int(n_estimators),
            max_depth=None if max_depth <= 0 else int(max_depth),
            random_state=int(rs)
        )

        with profile_block() as prof:
            model_rf.fit(X_tr, y_tr)
            train_stats = prof()

        metrics = evaluate_supervised(
            model_rf, X_tr, y_tr, X_te, y_te, fit=False
        )

        metrics.update(train_stats)
        results["Random Forest"] = metrics


    for name, metrics in results.items():
        st.subheader(name)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        c2.metric("Precision (w)", f"{metrics['precision_weighted']:.4f}")
        c3.metric("Recall (w)", f"{metrics['recall_weighted']:.4f}")
        c4.metric("F1 (w)", f"{metrics['f1_weighted']:.4f}")
        c5.metric("Time (s)", f"{metrics['train_time_sec']:.3f}")
        c6.metric("RAM (MB)", f"{metrics['memory_mb']:.2f}")
        st.plotly_chart(plot_cm(metrics["confusion_matrix"], np.unique(y), normalize_cm), use_container_width=True)

    if len(results) == 2:
        df_compare = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [results[m]["accuracy"] for m in results],
            "F1_weighted": [results[m]["f1_weighted"] for m in results],
            "Time (s)": [results[m]["train_time_sec"] for m in results],
            "RAM (MB)": [results[m]["memory_mb"] for m in results],
        })

        fig = px.bar(
            df_compare,
            x="Model",
            y=["Accuracy", "F1_weighted", "Time (s)", "RAM (MB)"],
            barmode="group",
            text_auto=".3f"
        )

        fig.update_layout(
            yaxis_title="Giá trị",
            legend_title="Chỉ số",
            bargap=0.25
        )

        st.plotly_chart(fig, use_container_width=True)

