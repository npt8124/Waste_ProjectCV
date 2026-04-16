# import streamlit as st
# import tempfile
# import cv2
# from src.infer_pipeline import run_pipeline

# st.set_page_config(layout="wide")

# st.title("♻️ Waste Detection AI")
# st.caption("YOLO26 Production Demo")

# uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

# if uploaded_file:

#     # =========================
#     # SAVE INPUT
#     # =========================
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Input")
#         st.video(tfile.name)

#     if st.button("Run Waste Detect and  Classify"):

#         # =========================
#         # UI ELEMENTS
#         # =========================
#         progress_bar = st.progress(0)
#         preview = st.empty()

#         def update_progress(p):
#             progress_bar.progress(min(p,1.0))

#         def update_preview(frame):
#             preview.image(frame, channels="BGR")

#         output_path = "outputs/final_streamlit.mp4"

#         with st.spinner("Processing..."):

#             run_pipeline(
#                 tfile.name,
#                 output_path,
#                 progress_callback=update_progress,
#                 preview_callback=update_preview
#             )

#         st.success("Done!")

#         with col2:
#             st.subheader("Output")
#             video_bytes = open(output_path, "rb").read()
#             st.video(video_bytes)


import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
from src.infer_pipeline import run_pipeline

st.set_page_config(layout="wide")

st.title("Waste Detection AI")
st.caption("YOLO26 Production Demo")

# =========================
# VIDEO
# =========================
uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.video(tfile.name)

    if st.button("Run Waste Detect and Classify"):

        progress_bar = st.progress(0)
        preview = st.empty()

        def update_progress(p):
            progress_bar.progress(min(p,1.0))

        def update_preview(frame):
            preview.image(frame, channels="BGR")

        output_path = "outputs/final_streamlit.mp4"

        with st.spinner("Processing..."):

            run_pipeline(
                tfile.name,
                output_path,
                progress_callback=update_progress,
                preview_callback=update_preview
            )

        st.success("Done!")

        with col2:
            st.subheader("Output")
            video_bytes = open(output_path, "rb").read()
            st.video(video_bytes)

# =========================
# IMAGE 
# =========================
import subprocess
import time
import os

st.divider()
st.subheader("Image Detection")

uploaded_image = st.file_uploader("Upload Image", type=["jpg","png"], key="image")

if uploaded_image:

    os.makedirs("src", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_path = "src/input.jpg"

    # save ảnh
    with open(input_path, "wb") as f:
        f.write(uploaded_image.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        st.image(input_path)

    if st.button("Run Image Detect and Classify"):

        with st.spinner("Processing image..."):

            # gọi file của bạn
            subprocess.run(["python", "src/infer_image.py"])

            time.sleep(1)  # đợi ghi file

        st.success("Done!")

        output_img = "outputs/result_image.jpg"

        if os.path.exists(output_img):

            with col2:
                st.subheader("Output")
                st.image(output_img)

        else:
            st.error("Output image not found")

        # JSON nếu có
        json_path = "outputs/result_image.json"
        if os.path.exists(json_path):
            st.subheader("Statistics")
            with open(json_path) as f:
                st.json(f.read())