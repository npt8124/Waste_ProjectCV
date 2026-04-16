from ultralytics import YOLO

def main():
    model = YOLO("yolo11m.pt")  # best cho RTX 3050

    model.train(
        data="datasets/detect/data.yaml",

        # ===== CORE =====
        epochs=120,    # số epoch (120 là đủ để model học tốt với dataset ~5k ảnh)
        imgsz=640, # kích thước ảnh đầu vào (640 = tiêu chuẩn YOLO, giữ chi tiết tốt)
        batch=4, # số ảnh mỗi batch 
        device=0, #GPU

        # ===== QUAN TRỌNG CHO CLASSIFY =====
        box=10.0,          # tăng độ chính xác bbox
        cls=0.3,           # giảm trọng số classify của detect
        dfl=1.5, #Distribution Focal Loss (giúp định vị bbox chính xác hơn)

        # ===== AUGMENT =====
        hsv_h=0.015,   # thay đổi hue nhẹ -> chống phụ thuộc màu
        hsv_s=0.7,  # thay đổi độ bão hòa -> robust ánh sáng
        hsv_v=0.4, # thay đổi độ sáng -> phù hợp môi trường thực tế
        fliplr=0.5,  # lật ngang 50% → tăng đa dạng dữ liệu
        mosaic=1.0, # ghép nhiều ảnh lại → tăng khả năng detect nhiều object
        mixup=0.1,         # giảm mixup (tránh làm bbox bị "ảo", ảnh hưởng crop)
        copy_paste=0.2,  # copy object từ ảnh khác → tăng số lượng object
        scale=0.5, # zoom in/out → model học đa kích thước
        translate=0.1, # dịch chuyển object → tăng robustness

        # =====  GIỮ BOX SẠCH =====
        close_mosaic=10,   #  tắt mosaic ở 10 epoch cuối,  giúp bbox giống dữ liệu thật 
        overlap_mask=False, # không cho phép overlap mask (giữ object rõ ràng)

        # ===== TRAIN =====
        optimizer="AdamW", # optimizer hiện đại -> hội tụ ổn định + generalization tốt
        lr0=0.0005,  # learning rate ban đầu (nhỏ -> tránh học quá nhanh gây sai)
        cos_lr=True, # cosine decay learning rate -> giúp training mượt hơn
        weight_decay=0.0005, # regularization -> giảm overfitting
        patience=25,  # early stopping: 25 epoch không cải thiện -> dừng

        # ===== PERFORMANCE =====
        workers=4, # số luồng load dữ liệu (tăng tốc training)
        verbose=True  # in log chi tiết khi train
    )

if __name__ == "__main__":
    main()