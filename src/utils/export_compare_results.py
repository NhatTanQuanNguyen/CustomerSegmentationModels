import os
import pandas as pd
import matplotlib.pyplot as plt

def export_compare_results(
    compare_data,
    title="So sánh kết quả mô hình",
    save_path="results/compare_summary.png",
    round_digits=4,
    show_values=True
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame(compare_data)
    for col in df.columns:
        if df[col].dtype in ["float64", "float32"]:
            df[col] = df[col].round(round_digits)

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.5 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Ảnh kết quả đã lưu tại: {save_path}")

    return save_path
