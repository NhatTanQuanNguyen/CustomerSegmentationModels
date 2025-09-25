from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main():
    file_path = "data/raw/noLabel/Online Retail.xlsx"
    pre = RFMPreprocessor(file_path)
    X = pre.process()

    # Khởi tạo KMeans với 4 cụm
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X)

    # Gán nhãn cụm cho từng khách hàng
    labels = kmeans.labels_
    print("Nhãn cụm cho 10 khách hàng đầu tiên:", labels[:10])

    # Vẽ scatter plot theo 2 chiều Recency & Monetary
    plt.scatter(X[:, 0], X[:, 2], c=labels, cmap="viridis", s=50)
    plt.xlabel("Recency (chuẩn hóa)")
    plt.ylabel("Monetary (chuẩn hóa)")
    plt.title("Phân cụm khách hàng bằng KMeans")
    plt.colorbar(label="Cluster")
    plt.show()


if __name__ == "__main__":
    main()
