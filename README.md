# PTITHCM Project - Customer Segmentation (RFM + KMeans)

Dự án này phân tích dữ liệu bán hàng để tính toán **RFM (Recency, Frequency, Monetary)** và áp dụng **KMeans clustering** nhằm phân cụm khách hàng.  
Code được viết theo chuẩn module để dễ dàng phát triển mở rộng và làm việc nhóm.

---

## 1. Yêu cầu hệ thống

- [Python Python 3.12.8](https://www.python.org/downloads/windows/)
- [Git](https://git-scm.com/download/win)
- [Visual Studio Code](https://code.visualstudio.com/)

Khuyến nghị cài thêm extension trong VS Code:
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance

---

## 2. Tạo và quản lý môi trường ảo (Windows)

### 2.1. Tạo môi trường ảo
```cmd
python -m venv .venv
```

### 2.2. Kích hoạt môi trường ảo
```cmd
.venv\Scripts\activate.bat
```

👉 Khi kích hoạt thành công, bạn sẽ thấy `(venv)` hiện ở đầu dòng terminal.

### 2.3. Thoát môi trường ảo
```cmd
deactivate
```

---

## 3. Cài đặt thư viện

Trong khi môi trường `.venv` đang kích hoạt:

```cmd
pip install -r requirements.txt
```

---

## 4. Mở project trong VS Code

1. `File > Open Folder...` → chọn thư mục `PTITHCM_Project`.
2. Nhấn `Ctrl+Shift+P` → gõ **Python: Select Interpreter**.
3. Chọn môi trường `.venv` vừa tạo.
4. Giờ bạn có thể chạy code trực tiếp trong VS Code (Debug hoặc Terminal).

---


## 5. Cấu trúc thư mục chính

```
PTITHCM_Project/
│── data/                  # Dữ liệu thô & đã xử lý
│── docs/                  # Tài liệu thiết kế, báo cáo
│── notebooks/             # Jupyter Notebook thử nghiệm
│── src/
│   ├── preprocesses/      # Tiền xử lý dữ liệu 
│   ├── models/            # Thuật toán ML/DL
│   │   └── unsupervised/  # Thuật toán học không giám sát (KMeans, ...) . Mỗi người code sẽ code trong thư mục có tên thuật toán của mính
│   └── visualization/     # Code trực quan hóa
│── tests/                 # Unit tests
│── requirements.txt       # Danh sách thư viện
│── README.md              # Hướng dẫn sử dụng
```

---

## 6. Thêm thư viện mới

Nếu bạn cần thêm thư viện:
```cmd
pip install <ten-thu-vien>
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Cập nhật requirements.txt"
git push origin main
```

