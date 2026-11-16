# Khai báo sử dụng công cụ AI (AI Usage Declaration)

Tài liệu này khai báo minh bạch việc sử dụng công cụ trí tuệ nhân tạo hỗ trợ trong quá trình thực hiện Bài tập lớn **Đề tài 12: Phát hiện giao dịch thẻ tín dụng gian lận** (Nhóm 12 — Học phần Nhập môn Học máy).

## 1. Công cụ đã sử dụng
- **Trợ lý AI lập trình:** Google Antigravity (hỗ trợ gợi ý cấu trúc dự án, rà soát code, tạo giao diện demo và định dạng báo cáo LaTeX).

## 2. Phạm vi hỗ trợ và phần nhóm tự chịu trách nhiệm
- **Ý tưởng & Thiết kế thí nghiệm:** Nhóm phân tích yêu cầu bài toán mất cân bằng lớp, xác định các rủi ro rò rỉ dữ liệu (1.081 dòng trùng lặp, thứ tự tách train/val/test trước khi chuẩn hóa) và lựa chọn so sánh 2 luồng tiền xử lý (Luồng A vs Luồng B) cùng các thuật toán K-NN, Naive Bayes, Decision Tree và Ensemble mở rộng.
- **Mã nguồn:** AI hỗ trợ gợi ý khung module (`src/`, `app/`, `notebooks/`) và rà soát lỗi cú pháp. Toàn bộ mã nguồn được nhóm kiểm tra, chạy thực nghiệm trực tiếp trên tệp `creditcard.csv` và kiểm chứng số liệu đầu ra thực tế trong `outputs/metrics/`.
- **Báo cáo & Diễn giải:** Nhóm tự đối chiếu kết quả thực nghiệm, phân tích lỗi sai (False Negative / False Positive), và chịu trách nhiệm hoàn toàn về tính chính xác cũng như khả năng giải trình từng dòng code và kết quả khi bảo vệ.

## 3. Các nhóm câu lệnh / prompt chính đã sử dụng
1. *"Dựa vào đề tài Phát hiện giao dịch thẻ tín dụng gian lận, hướng dẫn bài tập lớn của giảng viên và khảo sát tệp creditcard.csv, đề xuất kế hoạch thực nghiệm so sánh 2 luồng tiền xử lý chống data leakage."*
2. *"Xây dựng khung pipeline Python tách module (EDA, preprocessing, training, evaluation, error analysis) cho K-NN, Gaussian Naive Bayes, Decision Tree và Ensemble."*
3. *"Xây dựng giao diện demo dự đoán giao dịch gian lận và khung báo cáo LaTeX 6 phần theo đúng yêu cầu học phần."*
