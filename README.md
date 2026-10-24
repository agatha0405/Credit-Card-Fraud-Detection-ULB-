# Phát hiện gian lận giao dịch thẻ tín dụng

**Credit Card Fraud Detection (ULB)**  
Bài tập lớn môn Học máy — Trường Đại học Phenikaa.

## Giới thiệu project

Project xây dựng hệ thống phát hiện giao dịch thẻ tín dụng có dấu hiệu gian lận bằng Machine Learning. Nội dung gồm khám phá dữ liệu, tiền xử lý, huấn luyện và so sánh ba thuật toán **Logistic Regression, Random Forest, XGBoost**, đánh giá trên dữ liệu mất cân bằng và xây dựng ứng dụng minh họa bằng Streamlit.

Toàn bộ quá trình phân tích, huấn luyện và đánh giá được trình bày trong Jupyter Notebook.

## Demo Streamlit

Ứng dụng cho phép:

- Tải CSV giao dịch hoặc sử dụng dữ liệu demo có sẵn.
- Hiển thị điểm dự đoán, phân loại giao dịch bình thường và nghi gian lận.
- Xem thống kê, biểu đồ, lọc và sắp xếp kết quả.
- Đánh giá Precision, Recall, F1 và ma trận nhầm lẫn khi CSV có đủ nhãn hợp lệ của cả hai lớp.
- Tải kết quả phân tích về dưới dạng CSV.

Chọn **Dùng dữ liệu demo**, sau đó nhấn **Phân tích giao dịch** để trải nghiệm ứng dụng. File demo gồm 20 giao dịch thật, với 10 giao dịch mỗi lớp; kết quả trên demo chỉ phục vụ minh họa.

Ứng dụng sử dụng mô hình đã lưu, không cần huấn luyện lại. CSV đầu vào cần các cột `Time`, `V1`–`V28`, `Amount`; cột `Class` là tùy chọn.

## Dữ liệu và phương pháp

### Dữ liệu

Sử dụng bộ dữ liệu [Credit Card Fraud Detection của ULB/MLG trên Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), gồm **284.807 giao dịch**, 30 đặc trưng và một cột nhãn:

| Thành phần | Ý nghĩa |
| --- | --- |
| `Time` | Số giây tính từ giao dịch đầu tiên |
| `V1`–`V28` | Các đặc trưng đã được biến đổi bằng PCA để bảo mật thông tin |
| `Amount` | Giá trị giao dịch |
| `Class` | 0: bình thường; 1: gian lận |

Để chạy notebook, tải và giải nén dataset từ liên kết trên, sau đó đặt file tại `data/creditcard.csv`.

### Phương pháp

1. Khám phá phân phối nhãn, dữ liệu thiếu, trùng lặp và các đặc trưng.
2. Loại bản sao trùng hoàn toàn, giữ lần xuất hiện đầu; không loại giao dịch chỉ vì có giá trị bất thường.
3. Chia train/test theo tỷ lệ 80/20 với `stratify=y`, `random_state=42`; tách validation từ tập train.
4. Huấn luyện Logistic Regression, Random Forest và XGBoost trên cùng phân hoạch. Logistic Regression sử dụng StandardScaler cho `Time`, `Amount` trong Pipeline; Random Forest và XGBoost sử dụng dữ liệu thô.
5. Chọn mô hình và ngưỡng phân loại trên validation, sau đó huấn luyện lại mô hình được chọn trên toàn bộ train và đánh giá trên test.

Các bước chuẩn hóa chỉ được fit trên dữ liệu huấn luyện. Validation và test giữ nguyên phân phối nhãn; tập test không được sử dụng để lựa chọn mô hình hoặc ngưỡng.

## Kết quả chính

Mô hình được lựa chọn là **Random Forest**, với 100 cây, `min_samples_leaf=2`, `class_weight="balanced"` và `random_state=42`. Ngưỡng phân loại được chọn trên validation là **0,37**.

Kết quả trên tập test gồm **56.746 giao dịch**, trong đó có **95 giao dịch gian lận**:

| Chỉ số | Giá trị |
| --- | ---: |
| Precision | 0,9589 |
| Recall | 0,7368 |
| F1-score | 0,8333 |
| Average Precision (AP) | 0,8128 |
| ROC-AUC | 0,9293 |

Mô hình phát hiện đúng **70/95** giao dịch gian lận, bỏ sót **25** giao dịch và cảnh báo nhầm **3** giao dịch bình thường.

Các kết quả trên được lấy từ [test_metrics.csv](reports/test_metrics.csv). Bảng so sánh mô hình, kết quả validation và biểu đồ được trình bày trong notebook và thư mục `reports/`.

## Cài đặt và chạy

### Cài đặt môi trường

Môi trường thực nghiệm: **Python 3.13.5**, **Scikit-learn 1.7.0**.

Mở thư mục project trong VS Code và chạy các lệnh sau trong terminal PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Truy cập địa chỉ hiển thị trong terminal, mặc định là [http://localhost:8501](http://localhost:8501).

Ứng dụng tải ba file `models/best_model.joblib`, `models/threshold.json` và `models/features.json`. Dữ liệu demo có sẵn trong `data/demo_transactions.csv`, nên có thể chạy ứng dụng mà không cần tải toàn bộ dataset.

### Chạy notebook

Cài extension **Python** và **Jupyter** trong VS Code, chọn kernel `.venv` và chạy các notebook theo thứ tự:

| Notebook | Nội dung |
| --- | --- |
| `01_eda.ipynb` | Khám phá và trực quan hóa dữ liệu |
| `02_preprocessing.ipynb` | Xử lý trùng lặp, chia dữ liệu và chuẩn hóa |
| `03_model_training.ipynb` | Huấn luyện và so sánh ba thuật toán |
| `04_model_evaluation.ipynb` | Chọn ngưỡng, đánh giá test và lưu mô hình cuối |

Các notebook sử dụng `data/creditcard.csv` và đầu ra của các bước trước. Chạy tuần tự **01 → 02 → 03 → 04** để tái tạo đầy đủ dữ liệu xử lý, mô hình trung gian và kết quả thực nghiệm. Thư mục làm việc của kernel là thư mục gốc project hoặc `notebooks/`.

## Kiểm tra

Chạy bộ kiểm thử ứng dụng:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các kiểm thử bao gồm kiểm tra dữ liệu CSV, thứ tự đặc trưng, dự đoán từ mô hình đã lưu, tính chỉ số đánh giá, xuất kết quả và thao tác trên giao diện Streamlit. Bộ kiểm thử sử dụng dữ liệu demo và không huấn luyện lại mô hình.

## Cấu trúc project

```text
credit-card-fraud-detection/
├── data/
│   └── demo_transactions.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── models/
│   ├── best_model.joblib
│   ├── threshold.json
│   ├── features.json
│   └── standard_scaler.joblib
├── reports/
│   └── figures/
├── Thuyết trình/
│   ├── Credit_Card_Fraud_Detection_Presentation.pptx
│   └── Credit_Card_Fraud_Detection_Presentation.pdf
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_app.py
├── app.py
├── requirements.txt
└── README.md
```

## Nhóm thực hiện

**Nhóm 6 — Trường Đại học Phenikaa**

| Thành viên | Mã sinh viên |
| --- | --- |
| Trương Việt Thành | 24106898 |
| Bùi Khánh Linh | 24100543 |
| Nguyễn Kinh Bang | 24101477 |

## Giảng viên hướng dẫn

**Nguyễn Ngọc Giang** — Môn Học máy.