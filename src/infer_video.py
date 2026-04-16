import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================
VIDEO_PATH = "src/input.mp4"

DETECT_MODEL = "runs/detect/train3/weights/best.pt"
CLASSIFY_MODEL = "runs/classify/train/weights/best.pt"

CONF_DETECT = 0.15   #  tăng recall
CONF_CLASSIFY = 0.5
IOU_NMS = 0.5
MIN_AREA = 400
PAD = 10
SMOOTH = 5

# =========================
# COLOR
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
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(3))
h = int(cap.get(4))

os.makedirs("outputs", exist_ok=True)

out = cv2.VideoWriter(
    "outputs/result_final.mp4",
    cv2.VideoWriter_fourcc(*"avc1"),
    fps, (w, h)
)

# =========================
# TRACK + COUNT
# =========================
counted_ids = set()
counter = defaultdict(int)
history = defaultdict(lambda: deque(maxlen=SMOOTH))
last_boxes = {}   # 🔥 lưu box để chống miss

# =========================
# NMS FUNCTION
# =========================
def nms(boxes, scores, iou_threshold=0.5):
    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w_ = np.maximum(0, xx2-xx1)
        h_ = np.maximum(0, yy2-yy1)

        inter = w_ * h_
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # MULTI-SCALE DETECT
    # =========================
    res1 = detect_model(frame, conf=CONF_DETECT, imgsz=640)[0]
    res2 = detect_model(frame, conf=CONF_DETECT, imgsz=960)[0]

    boxes = []
    scores = []

    for r in [res1, res2]:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            conf = float(b.conf[0])
            boxes.append([x1,y1,x2,y2])
            scores.append(conf)

    # =========================
    # NMS
    # =========================
    if len(boxes) > 0:
        keep = nms(boxes, scores, IOU_NMS)
        boxes = [boxes[i] for i in keep]

    # =========================
    # TRACK
    # =========================
    results = detect_model.track(
        frame,
        conf=CONF_DETECT,
        persist=True,
        tracker="bytetrack.yaml"
    )[0]

    current_ids = set()

    if results.boxes is not None:
        for box in results.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            current_ids.add(track_id)
            last_boxes[track_id] = (x1,y1,x2,y2)

            # FILTER
            if (x2-x1)*(y2-y1) < MIN_AREA:
                continue

            # CROP
            crop = frame[max(0,y1-PAD):min(h,y2+PAD),
                         max(0,x1-PAD):min(w,x2+PAD)]

            if crop.size == 0:
                continue

            # =========================
            # CLASSIFY
            # =========================
            cls_result = classify_model(crop)[0]
            cls_id = int(cls_result.probs.top1)
            cls_name = classify_model.names[cls_id]
            cls_conf = float(cls_result.probs.top1conf)

            # SMOOTH
            history[track_id].append(cls_name)
            final_cls = max(set(history[track_id]),
                            key=history[track_id].count)

            # COUNT
            if track_id not in counted_ids:
                counter[final_cls] += 1
                counted_ids.add(track_id)

            # DRAW
            color = COLORS.get(final_cls, (255,255,255))
            label = f"{final_cls} ID:{track_id}"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # =========================
    # FALLBACK (CHỐNG MISS)
    # =========================
    for tid, bbox in last_boxes.items():
        if tid not in current_ids:
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,100,100),1)

    # =========================
    # DRAW COUNT
    # =========================
    y = 30
    for cls, cnt in counter.items():
        cv2.putText(frame,f"{cls}: {cnt}",
                    (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)
        y += 30

    out.write(frame)

cap.release()
out.release()

print("DONE → outputs/result_final.mp4")