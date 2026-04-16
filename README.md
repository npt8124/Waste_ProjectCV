# ♻️ Waste Detection & Classification System

## 📌 Giới thiệu

Dự án xây dựng hệ thống Phân loại rác bằng YOLO26.


Hệ thống sử dụng:

* YOLOv11 cho **phát hiện vật thể (Detection)**
* YOLO26-CLS cho **phân loại (Classification)**

Ứng dụng có thể:

* Nhận diện nhiều loại rác khác nhau
* Xử lý ảnh và video
* Hiển thị kết quả trực tiếp trên web (Streamlit)

---

##  Pipeline hệ thống

### 🔹 Pipeline kết hợp (Detection + Classification)

```text
Input (Image/Video)
        ↓
YOLOv11 (Detect Object)
        ↓
Crop từng object
        ↓
YOLO26-CLS (Classify)
        ↓
Hiển thị Bounding Box + Label
```

 Đây là pipeline chính của hệ thống, giúp:

* Tăng độ chính xác
* Phân loại chi tiết từng vật thể

---

### 🔹 Pipeline chỉ dùng YOLO26 (Classification)

```text
Input Image
     ↓
YOLO26-CLS
     ↓
Dự đoán nhãn (label)
```

 Dùng khi:

* Ảnh chỉ có 1 đối tượng chính
* Cần tốc độ nhanh

##  Công nghệ sử dụng

* Python
* OpenCV
* Ultralytics YOLO
* Streamlit (Web UI)
* PyTorch

---

##  Cài đặt

```bash
pip install -r requirements.txt
```

---

## ▶ Chạy ứng dụng

```bash
streamlit run app.py
```

---

##  Demo

* Upload ảnh hoặc video
* Hệ thống sẽ:

  * Detect object
  * Classify loại rác
  * Hiển thị kết quả trực quan

---

##  Kết quả

* Detection: YOLOv11
* Classification: YOLO26-CLS (~93% accuracy)

---

##  Hạn chế

* Một số object nhỏ có thể bị miss
* Nhầm lẫn giữa các loại rác tương tự (glass vs plastic)

---

##  Hướng phát triển

* Cải thiện dataset
* Tối ưu tracking và counting
* Deploy web online
* Thêm real-time camera





📁 detect/
Dùng cho YOLO detect (bbox)
Format:
class x_center y_center w h



📁 classify/
Dùng cho YOLO classify
Không cần label .txt
Phân loại bằng folder


📁 src/
toàn bộ code
không để code lẫn dataset (tránh lỗi path)


tách riêng detect / classify
dùng path rõ ràng
kiểm tra dataset trước train



1. chuẩn bị dataset  
2. kiểm tra dataset, chạy clean_dataset.py sau đó chạy check_dataset.py
Sau khi clean: XÓA cache 
datasets/detect/train/labels.cache
datasets/detect/valid/labels.cache
3. train detect    python src/train_detect.py


src/utils/split_classify_dataset.py để CHIA TRAIN / VALID
4. train classify
5. inference pipeline

