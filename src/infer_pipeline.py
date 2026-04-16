import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

# =========================
# CONFIG
# =========================
DETECT_MODEL = "runs/detect/train3/weights/best.pt"
CLASSIFY_MODEL = "runs/classify/train/weights/best.pt"

CONF_DETECT = 0.15
CONF_CLASSIFY = 0.5
MIN_AREA = 400
PAD = 10
SMOOTH = 5

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
# LOAD MODEL (CACHE)
# =========================
detect_model = YOLO(DETECT_MODEL)
classify_model = YOLO(CLASSIFY_MODEL)


# =========================
# MAIN PIPELINE FUNCTION
# =========================
def run_pipeline(input_path, output_path,
                 progress_callback=None,
                 preview_callback=None):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(3))
    h = int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    os.makedirs("outputs", exist_ok=True)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),  # 🔥 streamlit fix
        fps, (w, h)
    )

    counted_ids = set()
    counter = defaultdict(int)
    history = defaultdict(lambda: deque(maxlen=SMOOTH))
    last_boxes = {}

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

                # FILTER SIZE
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

                # SMOOTH LABEL
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
        # FALLBACK (ANTI-MISS)
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

        frame_id += 1

        # =========================
        # CALLBACKS
        # =========================
        if progress_callback:
            progress_callback(frame_id / total)

        if preview_callback:
            preview_callback(frame)

    cap.release()
    out.release()

    return dict(counter)