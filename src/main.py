from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.Kmeans_final import KMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator
import numpy as np


def main():
    print("🚀 START UNSUPERVISED COMPARISON PIPELINE")

    # ================== 1️⃣ Chuẩn bị dữ liệu ==================
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, rfm, rfm_scaled = pre.process()
    print(f"✅ Dữ liệu RFM đã sẵn sàng — shape: {X.shape}")

    # ================== 2️⃣ KMEANS (Code Tay) ==================
    print("\n🔹 Running KMeans (NumPy implementation)...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_

    print(f"✅ KMeans Clusters Created: {len(set(kmeans_labels))}")
    print("📍 KMeans Centroids:")
    print(np.round(kmeans.centroids, 4))
    print("🏷️ KMeans First 10 Labels:", kmeans_labels[:10])

    # ================== 3️⃣ MANUAL GMM ==================
    print("\n🔹 Running Manual Gaussian Mixture Model...")
    gmm_result = ManualGMM.run(X, n_components=3)
    gmm_labels = gmm_result["labels"]

    print(f"✅ GMM Clusters Created: {len(set(gmm_labels))}")
    print("📍 GMM Means:")
    print(np.round(gmm_result["means"], 4))
    print("🏷️ GMM First 10 Labels:", gmm_labels[:10])

    # ================== 4️⃣ ĐÁNH GIÁ & SO SÁNH ==================
    print("\n🔹 Evaluating and Comparing Models...")
    evaluator = UnsupervisedEvaluator()
    results = evaluator.compare_models(X, kmeans_labels, gmm_labels,
                                       name_a="KMeans", name_b="ManualGMM")

    print("\n===== ✅ SUMMARY =====")
    print(f"🔸 Label Match Ratio: {results['match_ratio']:.4f}")
    print("\n📊 Internal Metrics:")
    for model_name in ["KMeans", "ManualGMM"]:
        metrics = results[model_name]
        print(f"  {model_name}:")
        for key, value in metrics.items():
            print(f"    {key:<20}: {value:.4f}")

    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
