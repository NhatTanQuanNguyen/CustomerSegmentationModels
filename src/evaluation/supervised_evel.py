# src/evaluation/run_eval.py
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def evaluate_supervised(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    fit: bool = True
) -> Dict[str, Any]:
    """
    Truyền model + data, hàm sẽ (tùy chọn) fit rồi predict và trả metric.
    """
    if fit:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    return {
        "accuracy": acc,
        "precision_weighted": p,
        "recall_weighted": r,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "report": report,
        "y_pred": y_pred,   # tiện cho việc dùng tiếp
        "model": model,     # trả lại model đã fit
    }
