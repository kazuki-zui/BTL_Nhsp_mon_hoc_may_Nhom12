# Hướng dẫn chuẩn bị dữ liệu (`creditcard.csv`)

Theo quy định nộp bài, tệp dữ liệu gốc dung lượng lớn (~144 MB) không được đẩy trực tiếp lên GitHub.

## Thông tin bộ dữ liệu
- **Tên:** Credit Card Fraud Detection (ULB Machine Learning Group)
- **Kích thước:** 284.807 giao dịch × 31 cột (`Time`, `V1`–`V28`, `Amount`, `Class`)
- **Nguồn dữ liệu:** Bộ dữ liệu được giảng viên chỉ định cho **Đề tài 12** (nguồn gốc Kaggle / ULB Machine Learning Group: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`).

## Cách đặt tệp dữ liệu để chạy pipeline
Đặt tệp `creditcard.csv` vào một trong các vị trí sau:
1. `data/creditcard.csv` trong thư mục gốc của repository, **hoặc**
2. Truyền trực tiếp đường dẫn khi chạy lệnh:
   ```bash
   python -m src.run_pipeline --data-path "D:\creditcard.csv\creditcard.csv"
   ```
3. Hoặc đặt biến môi trường `CREDITCARD_DATA_PATH`.
