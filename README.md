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

