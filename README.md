# 🧭 ElevatorModelProject — Phân Cụm Khách Hàng & So Sánh Hiệu Năng Các Thuật Toán Không Giám Sát

## 🎯 Mục tiêu

Dự án nhằm **so sánh hiệu năng của các thuật toán phân cụm không giám sát (unsupervised clustering)** trên tập dữ liệu khách hàng để **chọn ra mô hình phân cụm tối ưu**, từ đó **làm nhãn cho mô hình giám sát (supervised learning)** giúp **dự đoán hành vi khách hàng trong tương lai**.

---

## 📚 1. Giới thiệu

Phân cụm khách hàng (Customer Segmentation) là bước quan trọng trong lĩnh vực **Marketing Data Science**.  
Mục tiêu là nhóm khách hàng dựa trên:
- Mức độ mua hàng gần đây (Recency)
- Tần suất giao dịch (Frequency)
- Tổng chi tiêu (Monetary)
- Và các yếu tố mở rộng khác như Length, Score (LRFMS, MTS)

Dự án kết hợp:
- Tiền xử lý và chuẩn hóa dữ liệu đầu vào,
- Ứng dụng nhiều thuật toán phân cụm không giám sát (KMeans, FCM, GMM, LCM, Hierarchical...),
- Đánh giá chất lượng phân cụm bằng các chỉ số nội tại,
- Sau đó sử dụng nhãn cụm tốt nhất để huấn luyện mô hình giám sát.

---

## 🔄 2. Pipeline tổng thể

### Quy trình hoạt động

1. **Tiền xử lý dữ liệu (Preprocessing)**  
   Làm sạch, chuẩn hóa và tính các chỉ số RFM / LRFMS / MTS.  
   Dùng các module `RFMPreprocessor`, `LRFMSPreprocessor`.

2. **Phân cụm không giám sát (Unsupervised Clustering)**  
   Thực nghiệm với nhiều thuật toán khác nhau:
   - KMeans (NumPy)
   - Fuzzy C-Means
   - Gaussian Mixture Model (EM)
   - Latent Class Model (LCM)
   - Hierarchical Clustering (Ward method)

3. **Đánh giá nội tại (Internal Evaluation)**  
   Dùng các chỉ số:
   - Silhouette Score  
   - Davies-Bouldin Index (DBI)  
   - Calinski-Harabasz Index (CHI)  
   - Dunn Index  

4. **Chọn mô hình tối ưu (Model Selection)**  
   Lựa chọn thuật toán có Silhouette cao, DBI thấp, CHI cao nhất.

5. **Huấn luyện mô hình giám sát (Supervised Learning)**  
   Dùng nhãn cụm tối ưu làm ground truth để huấn luyện:
   - SVM_RBF  
   - Decision Tree  
   - Random Forest  

6. **Dự đoán & Trực quan hóa**  
   Thực hiện dự đoán cho khách hàng mới và hiển thị kết quả trên giao diện Streamlit.

---

## ⚙️ 3. Các thành phần chính

### 🧠 A. Tiền xử lý dữ liệu
- Làm sạch, xử lý giá trị khuyết, loại bỏ ngoại lai (z-score, IQR).
- Chuẩn hóa dữ liệu bằng StandardScaler hoặc MinMaxScaler.
- Xây dựng đặc trưng RFM / LRFMS / MTS (Multi-Time-Series).

### 🤖 B. Các thuật toán phân cụm không giám sát

| Thuật toán | Đặc điểm nổi bật |
|-------------|------------------|
| **KMeans (NumPy)** | Tự cài đặt thủ công, tốc độ cao, dễ kiểm chứng |
| **Fuzzy C-Means (FCM)** | Mỗi điểm có thể thuộc nhiều cụm, tăng tính linh hoạt |
| **Gaussian Mixture Model (GMM)** | Dựa trên phân phối chuẩn hỗn hợp, huấn luyện bằng EM |
| **Latent Class Model (LCM)** | Mô hình phân phối xác suất tiềm ẩn |
| **Hierarchical Clustering (Ward)** | Không cần chọn số cụm trước, tạo cấu trúc cây phân cấp |

---

## 🧮 4. Các chỉ số đánh giá mô hình

Dùng module `UnsupervisedEvaluator` để tính toán:

| Chỉ số | Ý nghĩa | Đánh giá tốt |
|--------|----------|---------------|
| **Silhouette Score** | Đo độ tách biệt giữa các cụm | Càng gần 1 càng tốt |
| **Davies-Bouldin Index (DBI)** | Đo độ tương đồng giữa các cụm | Càng thấp càng tốt |
| **Calinski-Harabasz Index (CHI)** | Độ phân tán giữa cụm | Càng cao càng tốt |
| **Dunn Index** | Tỷ lệ giữa khoảng cách cụm và đường kính cụm | Càng cao càng tốt |

---

## 📘 5. Giai đoạn học giám sát (Supervised Phase)

Sau khi xác định cụm tối ưu, kết quả phân cụm sẽ được dùng làm nhãn để huấn luyện mô hình giám sát.

### Mô hình sử dụng:
- **SVM_RBF**: phân tách cụm phức tạp, phi tuyến tính  
- **DecisionTree**: dễ giải thích, trực quan  
- **RandomForestCluster**: tổng hợp nhiều cây quyết định, tăng độ chính xác

Mục tiêu là dự đoán cụm khách hàng mới dựa trên hành vi giao dịch.

---

## 🧠 6. Kỹ thuật xử lý đầu vào cho KMeans

- **Chuẩn hóa dữ liệu (StandardScaler)** tránh thuộc tính chi phối.
- **Loại nhiễu bằng PCA hoặc Feature Selection** giảm chiều không quan trọng.
- **Khởi tạo centroid thông minh (k-means++)** tăng tốc hội tụ.
- **Chạy song song (multi-threaded)** bằng ThreadPool để tối ưu tốc độ.

---

## 🧪 7. Kết quả & Đánh giá

- Thực hiện đánh giá 5 thuật toán trên cùng một tập dữ liệu.
- So sánh theo Silhouette, DBI, CHI, Dunn.
- Chọn mô hình tối ưu nhất làm nhãn huấn luyện.
- Trực quan kết quả bằng biểu đồ scatter, heatmap, radar và biểu đồ 3D.

---

## 💻 8. Cách triển khai & chạy

### Bước 1 — Cài môi trường

```bash
git clone https://github.com/NhatTanQuanNguyen/ElevatorModelProject.git
cd ElevatorModelProject
python -m venv venv
venv\Scripts\activate        # (Windows)
# hoặc
source venv/bin/activate     # (Linux / Mac)
pip install -r requirements.txt

### Bước 2 — Chạy ứng dụng

Dự án có **3 ứng dụng Streamlit chính**:

- **App 1:** `src/gui/app.py`  
  → So sánh các thuật toán phân cụm không giám sát  

- **App 2:** `src/gui/app_2.py`  
  → Huấn luyện và đánh giá mô hình giám sát  

- **App 3:** `src/gui/app_3.py`  
  → Phân tích LRFMS + MTS và hiển thị biểu đồ trực quan  

Chạy bằng lệnh:

```bash
streamlit run src/gui/app.py
streamlit run src/gui/app_2.py
streamlit run src/gui/app_3.py


---

## 🔄 Sơ đồ sử dụng

```mermaid
flowchart TD
    A[📂 Dữ liệu đầu vào (Transaction Data)] --> B[🧹 Tiền xử lý dữ liệu (RFM / LRFMS / MTS)]
    B --> C[🤖 Phân cụm không giám sát: KMeans / FCM / GMM / LCM / Hierarchical]
    C --> D[📈 Đánh giá chất lượng cụm: Silhouette, DBI, CHI, Dunn]
    D --> E[🏆 Chọn mô hình tốt nhất]
    E --> F[🎓 Huấn luyện mô hình giám sát: SVM, DecisionTree, RandomForest]
    F --> G[📊 Trực quan hóa kết quả & dự đoán khách hàng mới]
    G --> H[💡 Ứng dụng trong marketing và phân tích hành vi]
