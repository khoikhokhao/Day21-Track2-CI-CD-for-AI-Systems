# Dự án MLOps: Pipeline Dự đoán Chất lượng Rượu (Wine Quality Prediction)

## 👤 Thông tin sinh viên

- **Họ và tên:** Phạm Minh Khôi
- **MSSV:** 2A202600262
- **Cohort:** AI-K1
- **Lớp:** E403

---

## 📝 Giới thiệu dự án

Dự án này xây dựng một hệ thống **MLOps Pipeline** hoàn chỉnh, tự động hóa quy trình từ khâu kiểm thử, huấn luyện mô hình cho đến triển khai lên máy chủ thực tế (AWS EC2). Hệ thống đảm bảo rằng chỉ những mô hình đạt tiêu chuẩn chất lượng mới được phép triển khai đến người dùng cuối.

---

## 🛠 Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.10
- **Framework:** FastAPI, Scikit-learn
- **Quản lý dữ liệu & mô hình:** DVC (Data Version Control), AWS S3
- **Theo dõi thí nghiệm:** MLflow
- **CI/CD:** GitHub Actions
- **Hạ tầng:** AWS EC2 (Ubuntu 24.04), AWS S3

---

## ⚙️ Quy trình Pipeline (CI/CD)

Pipeline được thiết lập tự động kích hoạt mỗi khi có thay đổi trong mã nguồn hoặc tham số (`params.yaml`).

1. **Unit Test**  
   Chạy kiểm thử tự động với `pytest` để đảm bảo logic code không bị lỗi.

2. **Train**  
   Tải dữ liệu từ S3 qua DVC, huấn luyện mô hình và lưu trữ kết quả vào MLflow.

3. **Eval (Quality Gate)**  
   Kiểm tra độ chính xác (Accuracy). Nếu **Accuracy < 0.70**, pipeline sẽ tự động dừng lại và hủy bỏ quá trình triển khai.

4. **Deploy**  
   Tự động SSH vào AWS EC2, cập nhật mô hình mới nhất và khởi động lại service API.

---

## 🚀 Hướng dẫn sử dụng

### 1. Kiểm tra trạng thái hệ thống

Để kiểm tra server đã hoạt động hay chưa, sử dụng lệnh:

```bash
curl http://32.236.18.122:8000/health
```

---

### 2. Gửi yêu cầu dự đoán

Sử dụng lệnh sau (trên Linux/Ubuntu) để dự đoán chất lượng rượu từ một mẫu dữ liệu:

```bash
curl -X POST http://32.236.18.122:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

Kết quả trả về mẫu:

```json
{
  "prediction": 0,
  "label": "thap"
}
```

---

## 📊 Kết quả đạt được

- Triển khai thành công hệ thống CI/CD tự động hóa 100%.
- Mô hình được lưu trữ và quản lý phiên bản chặt chẽ trên AWS S3 qua DVC.
- Hệ thống API có khả năng phản hồi nhanh và xử lý dự đoán thời gian thực.
