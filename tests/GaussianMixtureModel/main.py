import numpy as np
from sklearn.mixture import GaussianMixture
from src.models.unsupervised.GaussianMixtureModel.main import ManualGMM


class LibraryGMM:
    """
    Gaussian Mixture Model dùng thư viện sklearn.
    Dùng để so sánh với ManualGMM (code tay).
    Trả về:
        - labels: nhãn cụm
        - means: trung tâm cụm
        - weights: trọng số cụm
        - covariances: ma trận hiệp phương sai
        - score: log-likelihood trung bình
    """

    @staticmethod
    def run(X, n_components=3, random_state=42, max_iters=100):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
            max_iter=max_iters
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        return {
            "labels": labels,
            "means": gmm.means_,
            "weights": gmm.weights_,
            "covariances": gmm.covariances_,
            "score": gmm.score(X),
        }


def compare_manual_and_library(X, n_components=3):
    """
    So sánh kết quả giữa ManualGMM (code tay) và GaussianMixture (sklearn)
    """

    print("🔹 Manual GMM (Code Tay)")
    manual_result = ManualGMM.run(X, n_components=n_components)
    print("Means (Manual):")
    print(np.round(manual_result["means"], 4))
    print("Weights (Manual):", np.round(manual_result["weights"], 4))

    print("\n🔹 Library GMM (sklearn)")
    lib_result = LibraryGMM.run(X, n_components=n_components)
    print("Means (Library):")
    print(np.round(lib_result["means"], 4))
    print("Weights (Library):", np.round(lib_result["weights"], 4))

    # So sánh
    print("\n===== COMPARISON =====")
    manual_ll = manual_result["log_likelihoods"][-1] if manual_result["log_likelihoods"] else np.nan
    print(f"ManualGMM log-likelihood: {manual_ll:.4f}")
    print(f"LibraryGMM avg log-likelihood: {lib_result['score']:.4f}")

    # So nhãn
    match = (manual_result["labels"] == lib_result["labels"]).mean()
    print(f"Cluster label match ratio: {match:.4f}")

    return {
        "manual": manual_result,
        "library": lib_result,
        "match_ratio": match
    }


# =================== TEST TRỰC TIẾP ===================
if __name__ == "__main__":
    # Nếu bạn đã có ma trận X thật, chỉ cần import và truyền vào:
    # from Normalization.RFM.cleanData import RFMPreprocessor
    # pre = RFMPreprocessor("data/raw/Online Retail.xlsx")
    # X, _, _ = pre.process()

    # Test nhanh với dữ liệu giả lập
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, n_features=3, centers=3, random_state=42)

    print("🚀 BẮT ĐẦU SO SÁNH MANUAL vs LIBRARY GMM\n")
    result = compare_manual_and_library(X, n_components=3)
    print("\n🎯 HOÀN THÀNH SO SÁNH")
