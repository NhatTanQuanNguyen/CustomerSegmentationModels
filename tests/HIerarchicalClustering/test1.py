import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from src.preprocesses.noLabel.cleanData import RFMPreprocessor


def main():
    # === 1️⃣ ĐỌC DỮ LIỆU ===
    file_path = "data/raw/noLabel/Online Retail.xlsx"
    pre = RFMPreprocessor(file_path)
    X = pre.process()
    df = pre.rfm

    print("✅ Đọc dữ liệu thành công! Tổng số dòng:", len(df))

    # === 2️⃣ CHỌN 1000 DÒNG ĐẦU & BỎ CỘT ID ===
    numeric_df = df.select_dtypes(include=["int64", "float64"]).dropna().head(1000)
    if "CustomerID" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["CustomerID"])
        print("🧹 Đã loại bỏ cột CustomerID khỏi dữ liệu phân cụm.")

    print(f"📊 Dùng {len(numeric_df)} dòng đầu để chạy phân cụm với {numeric_df.shape[1]} đặc trưng.")

    # === 3️⃣ CHUẨN HÓA DỮ LIỆU ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # === 4️⃣ THỰC HIỆN HIERARCHICAL CLUSTERING ===
    Z = linkage(X_scaled, method='ward')

    # === 5️⃣ CẮT CÂY THỦ CÔNG ===
    # Bạn có thể đổi t = 3, 4, 5... để chọn số cụm mong muốn
    k = int(input("👉 Nhập số cụm mong muốn: "))
    print(f"🔹 Đang cắt cây thành {k} cụm...")
    labels = fcluster(Z, t=k, criterion='maxclust')

    # === 6️⃣ GẮN NHÃN CỤM VÀ HIỂN THỊ ===
    numeric_df["Cluster"] = labels
    print("\n📊 Thống kê trung bình theo cụm:")
    print(numeric_df.groupby("Cluster").mean())

    # === 7️⃣ VẼ DENDROGRAM ===
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=k, show_leaf_counts=True)
    plt.title(f"Dendrogram - Ward Linkage ({k} cụm)")
    plt.xlabel("Samples / Clusters")
    plt.ylabel("Khoảng cách (Ward)")
    plt.grid(True)
    plt.show()

    # === 8️⃣ VẼ SCATTER PLOT ===
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=50)
    plt.title(f"Hierarchical Clustering (Ward) - {k} cụm")
    plt.xlabel(numeric_df.columns[0])
    plt.ylabel(numeric_df.columns[1])
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
