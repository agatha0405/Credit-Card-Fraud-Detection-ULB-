# Quy ước project

- Viết phần Machine Learning trong Jupyter Notebook (`.ipynb`), không tách phân tích, tiền xử lý, huấn luyện hoặc đánh giá sang các file `.py` riêng.
- Mỗi cell thực hiện một nhiệm vụ rõ ràng; dùng biến có tên dễ hiểu và cách dùng Pandas, NumPy, Scikit-learn thông dụng.
- Markdown tiếng Việt chỉ dùng cho tiêu đề và nhận xét cần thiết. Không giải thích lệnh Python cơ bản, không comment từng dòng, không thêm hàm/class khi chưa cần.
- Hiển thị trực tiếp bảng và biểu đồ; biểu đồ có tiêu đề, nhãn trục và lưu tại `reports/figures/`.
- Chỉ báo cáo kết quả đã chạy thực tế. Chạy lại các cell bị ảnh hưởng; khi thay đổi luồng notebook, chạy tuần tự toàn bộ và lưu output.
- Không chỉnh sửa dataset gốc, không tạo dữ liệu giả. Thiếu dataset phải báo rõ.
- Giữ chất lượng đánh giá cho dữ liệu mất cân bằng: precision, recall, F1, PR-AUC và confusion matrix; không chỉ accuracy.
- Chia dữ liệu trước khi học các bước tiền xử lý hoặc chọn đặc trưng cho mô hình. Chỉ fit trên tập huấn luyện; không cân bằng lại validation/test, không dùng test để chọn mô hình hoặc ngưỡng. Tránh các bản sao lọt sang nhiều tập.
- Chỉ thực hiện giai đoạn người dùng đã yêu cầu. Hiện tại đã được yêu cầu giai đoạn 1–5; giai đoạn 5 chỉ tạo ứng dụng Streamlit. Không sửa bốn notebook, không huấn luyện lại hoặc thay đổi best_model.joblib, threshold.json, features.json. Không tự tạo báo cáo hoặc slide.
- Giai đoạn 4: xác nhận tham số rf_leaf2 từ giai đoạn 3, chọn ngưỡng F1 trên validation rồi refit trên toàn bộ train. Chỉ sau khi cố định mô hình và ngưỡng mới đọc test để đánh giá; không dùng test để thay đổi quyết định. Lưu mô hình cuối, ngưỡng và thứ tự đặc trưng cho giai đoạn sau.
- Dùng cùng phân hoạch train/test cho các mô hình. Nếu cross-validation, dùng dữ liệu train chưa chuẩn hóa với Pipeline để fit scaler riêng từng fold; không dùng dữ liệu đã chuẩn hóa trên toàn bộ train cho cross-validation.
- Streamlit dùng DataFrame thô theo features.json, predict_proba và ngưỡng trong threshold.json. Không scaler hoặc predict() mặc định; không lưu dữ liệu tải lên ra đĩa. Class chỉ dùng để đánh giá dữ liệu tải lên, không đưa vào đầu vào mô hình.
