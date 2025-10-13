# src/gui/app_2.py

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
from src.evaluation.supervised_evel import evaluate_supervised

st.set_page_config(page_title="SVM tay — GUI", layout="wide")
st.title("SVM RBF (code tay) — Train/Test & Đánh giá")

with st.sidebar:
    source = st.radio("Nguồn dữ liệu", ["Path nội bộ", "Upload Excel"], index=0)
    default_path = "data/raw/noLabel/Online Retail.xlsx"
    file_path = st.text_input("Đường dẫn Excel", value=default_path) if source == "Path nội bộ" else None
    up_file = st.file_uploader("Chọn file Excel", type=["xlsx", "xls"]) if source != "Path nội bộ" else None

    k = st.slider("K (KMeans)", 2, 12, 4)
    k_seed = st.number_input("Random state KMeans", value=42, step=1)

    test_size = st.slider("Tỷ lệ test", 0.1, 0.5, 0.3, step=0.05)
    rs = st.number_input("Random state split", value=42, step=1)
    stratify = st.checkbox("Stratify theo nhãn KMeans", value=True)

    C = st.number_input("C", value=2.0, step=0.1, format="%.3f")
    gamma = st.number_input("gamma", value=0.5, step=0.1, format="%.3f")
    max_iter = st.number_input("max_iter", value=500, step=50)
    calib_split = st.slider("Calibration split (Platt)", 0.0, 0.5, 0.2, step=0.05)

    normalize_cm = st.checkbox("Normalize confusion matrix (%)", value=True)
    show_proba = st.checkbox("Hiển thị bảng xác suất (top 15)", value=True)

    run_btn = st.button("Huấn luyện & Đánh giá", use_container_width=True)

@st.cache_resource(show_spinner=True)
def load_rfm(src):
    pre = RFMPreprocessor(src)
    X_scaled, rfm_original, _ = pre.process()
    return X_scaled, rfm_original

def plot_cm(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype(float)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm / np.clip(row_sum, 1e-12, None) * 100.0)
        df = pd.DataFrame(np.round(cm, 1), index=class_names, columns=class_names)
        fig = px.imshow(df, text_auto=True, aspect="auto",
                        labels=dict(x="Predicted", y="True", color="%"),
                        color_continuous_scale="Blues")
    else:
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        fig = px.imshow(df, text_auto=True, aspect="auto",
                        labels=dict(x="Predicted", y="True", color="Count"),
                        color_continuous_scale="Blues")
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig

if run_btn:
    if source == "Path nội bộ":
        if not os.path.exists(file_path):
            st.error(f"Không tìm thấy file: {file_path}")
            st.stop()
        X_scaled, _ = load_rfm(file_path)
    else:
        if up_file is None:
            st.error("Hãy upload file Excel.")
            st.stop()
        X_scaled, _ = load_rfm(up_file)

    st.write(f"Số mẫu: {X_scaled.shape[0]}, số đặc trưng: {X_scaled.shape[1]}")
    st.dataframe(pd.DataFrame(X_scaled[:10], columns=[f"f{i+1}" for i in range(X_scaled.shape[1])]))

    kmeans = KMeansNumpy.fit_k(X_scaled, k=int(k), random_state=int(k_seed))
    y = kmeans["labels"]
    classes = np.unique(y)
    st.write("Phân bố nhãn:", pd.Series(y).value_counts().sort_index())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=float(test_size), random_state=int(rs),
        stratify=y if stratify else None
    )
    st.write(f"Train: {X_tr.shape}, Test: {X_te.shape}")

    model = MultiClassSVM(C=float(C), gamma=float(gamma), max_iter=int(max_iter),
                          calib_split=float(calib_split), random_state=int(rs))

    with st.spinner("Đang huấn luyện..."):
        metrics = evaluate_supervised(model, X_tr, y_tr, X_te, y_te, fit=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (w)", f"{metrics['precision_weighted']:.4f}")
    c3.metric("Recall (w)", f"{metrics['recall_weighted']:.4f}")
    c4.metric("F1 (w)", f"{metrics['f1_weighted']:.4f}")

    st.markdown("**Confusion Matrix**")
    st.plotly_chart(plot_cm(metrics["confusion_matrix"], classes, normalize_cm), use_container_width=True)

    rep = pd.DataFrame(metrics["report"]).T
    st.markdown("**Classification report (đầy đủ)**")
    st.dataframe(rep, use_container_width=True)

    pc = rep.loc[classes.astype(str), ["precision", "recall", "f1-score", "support"]].rename(columns={"f1-score":"f1"})
    st.markdown("**Per-class metrics (gọn)**")
    st.dataframe(pc.style.format({"precision":"{:.4f}", "recall":"{:.4f}", "f1":"{:.4f}", "support":"{:.0f}"}),
                 use_container_width=True)

    if show_proba and hasattr(metrics["model"], "predict_proba"):
        proba = metrics["model"].predict_proba(X_te)
        proba_df = pd.DataFrame(proba, columns=[f"class_{c}" for c in classes])
        proba_df.insert(0, "y_true", y_te)
        proba_df.insert(1, "y_pred", metrics["y_pred"])
        st.markdown("**Xác suất dự đoán (top 15)**")
        st.dataframe(proba_df.head(15), use_container_width=True)

    st.info("y là nhãn KMeans (pseudo-labels) → chỉ số phản ánh mức độ tái tạo nhãn KMeans.")
