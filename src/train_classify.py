from ultralytics import YOLO

def main():
    model = YOLO("yolo26x-cls.pt")

    model.train(
        data="datasets/classify",
        epochs=100,          # số vòng lặp training
        imgsz=256,           # kích thước ảnh đầu vào (classification không cần quá lớn như detect)
        batch=32,            # số ảnh mỗi batch (classification nhẹ -> batch lớn giúp train nhanh hơn)
        device=0,            # dùng GPU

        patience=10,         # early stopping: nếu 10 epoch không cải thiện → dừng

        optimizer="AdamW", # optimizer tốt cho deep learning hiện đại (ổn định + generalize tốt)
        lr0=0.0005, # learning rate ban đầu (nhỏ → học ổn định, tránh overshoot)

        # augmentation mạnh hơn
        fliplr=0.5, # lật ngang 50% ảnh → tăng đa dạng dữ liệu
        hsv_h=0.02, # thay đổi hue (màu sắc nhẹ) → giúp model không phụ thuộc màu
        hsv_s=0.8, # thay đổi độ bão hòa → mô phỏng ánh sáng khác nhau
        hsv_v=0.5, # thay đổi độ sáng → robust hơn với môi trường thực
        erasing=0.4,  # random xóa 1 phần ảnh giúp model học tổng thể object, không phụ thuộc 1 vùng

        verbose=True # in log chi tiết khi train
    )

if __name__ == "__main__":
    main()