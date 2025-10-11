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
    """
    Hàm xuất ảnh bảng so sánh kết quả giữa các mô hình hoặc giữa thực tế & mô phỏng.

    Parameters
    ----------
    compare_data : list[dict]
        Dữ liệu so sánh dạng list các dict, ví dụ:
        [
            {"Tên mô hình": "KMeans", "Silhouette": 0.53, "DBI": 0.89, "CHI": 300.2, "Dunn": 0.18},
            {"Tên mô hình": "Fuzzy C-Means", "Silhouette": 0.52, "DBI": 0.91, "CHI": 289.7, "Dunn": 0.15},
            {"Tên mô hình": "GMM", "LogL": -14523.12, "MatchRate": 0.9435}
        ]
    title : str
        Tiêu đề hiển thị trên ảnh
    save_path : str
        Đường dẫn nơi lưu file ảnh (mặc định: results/compare_summary.png)
    round_digits : int
        Số chữ số sau dấu phẩy
    show_values : bool
        Có hiển thị giá trị số trên bảng hay không
    """

    # Tạo thư mục lưu nếu chưa có
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Chuyển dữ liệu sang DataFrame
    df = pd.DataFrame(compare_data)
    for col in df.columns:
        if df[col].dtype in ["float64", "float32"]:
            df[col] = df[col].round(round_digits)

    # Xuất bảng
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
    print(f"✅ Ảnh kết quả đã lưu tại: {save_path}")

    return save_path
