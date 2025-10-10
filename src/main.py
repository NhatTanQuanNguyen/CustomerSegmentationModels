from src.preprocesses.noLabel.cleanData import RFMPreprocessor
from src.models.unsupervised.Kmean.Kmeans_final import KMeans
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM
from src.models.unsupervised.FuzzyCMean.Fuzzy_Cmeans_final import FuzzyCMeans
from src.evaluation.unsupervised_eval import UnsupervisedEvaluator
import numpy as np


def main():
    print("🚀 START UNSUPERVISED COMPARISON PIPELINE")

    # ================== 1️⃣ Chuẩn bị dữ liệu ==================
    pre = RFMPreprocessor("data/raw/noLabel/Online Retail.xlsx")
    X, rfm, rfm_scaled = pre.process()
    print(f"✅ Dữ liệu RFM đã sẵn sàng — shape: {X.shape}")

    # ================== 2️⃣ KMEANS ==================
    print("\n🔹 Running KMeans (NumPy implementation)...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_

    print(f"✅ KMeans Clusters Created: {len(set(kmeans_labels))}")
    print("📍 KMeans Centroids:")
    print(np.round(kmeans.centroids, 4))
    print("🏷️ KMeans First 10 Labels:", kmeans_labels[:10])

    # ================== 3️⃣ FUZZY C-MEANS ==================
    print("\n🔹 Running Fuzzy C-Means...")
    fcm = FuzzyCMeans(n_clusters=3, random_state=42)
    fcm.fit(X)
    fcm_labels = fcm.labels_

    print(f"✅ Fuzzy C-Means Clusters Created: {len(set(fcm_labels))}")
    print("📍 Fuzzy C-Means Centers:")
    print(np.round(fcm.centroids, 4))
    print("🏷️ Fuzzy C-Means First 10 Labels:", fcm_labels[:10])

    # ================== 4️⃣ MANUAL GMM ==================
    print("\n🔹 Running Manual Gaussian Mixture Model...")
    gmm_result = ManualGMM.run(X, n_components=3)
    gmm_labels = gmm_result["labels"]

    print(f"✅ GMM Clusters Created: {len(set(gmm_labels))}")
    print("📍 GMM Means:")
    print(np.round(gmm_result["means"], 4))
    print("🏷️ GMM First 10 Labels:", gmm_labels[:10])

    # ================== 5️⃣ ĐÁNH GIÁ & SO SÁNH ==================
    print("\n🔹 Evaluating and Comparing Models...")
    evaluator = UnsupervisedEvaluator()

    results_k_gmm = evaluator.compare_models(X, kmeans_labels, gmm_labels,
                                             name_a="KMeans", name_b="ManualGMM")
    results_k_fcm = evaluator.compare_models(X, kmeans_labels, fcm_labels,
                                             name_a="KMeans", name_b="FuzzyCMeans")
    results_fcm_gmm = evaluator.compare_models(X, fcm_labels, gmm_labels,
                                               name_a="FuzzyCMeans", name_b="ManualGMM")

    # ================== 6️⃣ TỔNG HỢP ==================
    print("\n===== ✅ SUMMARY =====")
    for res, name in zip(
        [results_k_gmm, results_k_fcm, results_fcm_gmm],
        ["KMeans ↔ GMM", "KMeans ↔ Fuzzy C-Means", "Fuzzy C-Means ↔ GMM"]
    ):
        print(f"\n📊 {name}")
        print(f"  🔸 Label Match Ratio: {res['match_ratio']:.4f}")
        for model_name in res:
            if model_name in ["KMeans", "ManualGMM", "FuzzyCMeans"]:
                print(f"  {model_name}:")
                for key, value in res[model_name].items():
                    print(f"    {key:<20}: {value:.4f}")

    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
