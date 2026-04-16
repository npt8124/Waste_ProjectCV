import cv2
import os
import json
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
IMAGE_PATH = "src/test.jpg"

DETECT_MODEL = "runs/detect/train3/weights/best.pt"
CLASSIFY_MODEL = "runs/classify/train/weights/best.pt"

CONF_DETECT = 0.25
CONF_CLASSIFY = 0.5
MIN_AREA = 400
PAD = 10

# =========================
# COLORS
# =========================
COLORS = {
    "plastic": (0,255,0),
    "metal": (255,0,0),
    "paper": (0,255,255),
    "glass": (255,0,255),
    "cardboard": (255,255,0),
    "trash": (0,0,255),
    "unknown": (128,128,128)
}

# =========================
# LOAD MODEL
# =========================
detect_model = YOLO(DETECT_MODEL)
classify_model = YOLO(CLASSIFY_MODEL)

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Cannot load image")
    exit()

h, w = img.shape[:2]

# =========================
# OUTPUT
# =========================
os.makedirs("outputs", exist_ok=True)
output_img_path = "outputs/result_image.jpg"
output_json_path = "outputs/result_image.json"

results_json = []
counter = {}

# =========================
# DETECT
# =========================
results = detect_model(img, conf=CONF_DETECT)[0]

if results.boxes is not None:
    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # filter size
        if (x2-x1)*(y2-y1) < MIN_AREA:
            continue

        # padding
        x1p = max(0, x1-PAD)
        y1p = max(0, y1-PAD)
        x2p = min(w, x2+PAD)
        y2p = min(h, y2+PAD)

        crop = img[y1p:y2p, x1p:x2p]
        if crop.size == 0:
            continue

        # =========================
        # CLASSIFY
        # =========================
        cls_result = classify_model(crop)[0]
        cls_id = int(cls_result.probs.top1)
        cls_name = classify_model.names[cls_id]
        cls_conf = float(cls_result.probs.top1conf)

        if cls_conf < CONF_CLASSIFY:
            cls_name = "unknown"

        # =========================
        # COUNT
        # =========================
        counter[cls_name] = counter.get(cls_name, 0) + 1

        # =========================
        # DRAW
        # =========================
        color = COLORS.get(cls_name, (255,255,255))
        label = f"{cls_name} {cls_conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label,
                                     cv2.FONT_HERSHEY_SIMPLEX,
                                     0.6, 2)

        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)

        cv2.rectangle(img,
                      (x1, y1-th-10),
                      (x1+tw, y1),
                      color,
                      -1)

        cv2.putText(img, label,
                    (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2)

        # =========================
        # SAVE JSON
        # =========================
        results_json.append({
            "bbox": [x1, y1, x2, y2],
            "detect_conf": conf,
            "class": cls_name,
            "class_conf": cls_conf
        })

# =========================
# DRAW COUNT
# =========================
y = 30
for k, v in counter.items():
    cv2.putText(img, f"{k}: {v}",
                (10,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),2)
    y += 30

# =========================
# SAVE
# =========================
cv2.imwrite(output_img_path, img)

with open(output_json_path, "w") as f:
    json.dump(results_json, f, indent=4)

print("✅ DONE")
print("Image:", output_img_path)
print("JSON:", output_json_path)