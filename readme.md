# 🛡️ Ứng dụng Phát hiện Gian lận Giao dịch & Rủi ro Tín dụng

Ứng dụng web thông minh được phát triển dựa trên Streamlit và Scikit-Learn nhằm tự động hóa quy trình huấn luyện mô hình học máy, kiểm tra hiệu năng định danh, và chấm điểm rủi ro tài chính/gian lận tín dụng (biến mục tiêu: `default`).

## ✨ Tính năng chính

- **Cấu hình động (Sidebar):** Cho phép tùy biến tải tệp dữ liệu, hoán đổi linh hoạt giữa 3 thuật toán trong thực nghiệm (*Random Forest*, *Decision Tree*, *Logistic Regression*) và tinh chỉnh siêu tham số trực tiếp.
- **Tổng quan dữ liệu:** Xem trước cấu trúc tập dữ liệu thô kết hợp tự động trích xuất thống kê mô tả.
- **Trực quan hóa đồ thị tương tác:** Đồ thị hóa tự động bằng Plotly hiển thị phân phối của biến mục tiêu và tập biến đặc trưng X theo dạng lưới 2x2 thông minh.
- **Thử nghiệm & Đánh giá:** Hiển thị trực quan bảng Classification Report cùng Ma trận nhầm lẫn (Confusion Matrix) dạng biểu đồ nhiệt ngay sau khi huấn luyện.
- **Dự báo đa chế độ:** Hỗ trợ nhập thông số thủ công phân tích giao dịch đơn lẻ hoặc tải tệp Excel/CSV mẫu `X_new` để chấm điểm hàng loạt rồi kết xuất file báo cáo.

## 🛠️ Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Chuẩn bị môi trường máy trạm
Đảm bảo máy tính của bạn đã được cài đặt Python (phiên bản khuyến nghị `>= 3.9`).

### Bước 2: Cài đặt các thư viện bổ trợ cần thiết
Di chuyển terminal vào thư mục chứa mã nguồn của ứng dụng và chạy lệnh sau để thiết lập thư viện:
```bash
pip install -r requirements.txt
