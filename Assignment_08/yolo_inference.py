import os

import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Pencil & Charger Detector", layout="centered")

st.title("Pencil & Phone Charger Detector")
st.write("Upload an image and the trained YOLO model will detect pencils and phone chargers.")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

confidence = st.slider("Confidence threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    if st.button("Detect objects"):
        results = model.predict(image, conf=confidence)
        result = results[0]

        # result.plot() returns a BGR numpy array -> convert to RGB
        annotated = result.plot()[:, :, ::-1]

        with col2:
            st.subheader("Detections")
            st.image(annotated, use_container_width=True)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            st.warning("No objects detected. Try lowering the confidence threshold.")
        else:
            st.success(f"{len(boxes)} object(s) detected")
            rows = []
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                rows.append({"Class": model.names[int(cls_id)], "Confidence": round(conf, 2)})
            st.table(rows)
