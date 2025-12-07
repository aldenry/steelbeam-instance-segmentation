import os
import tempfile

import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Steel Beam Instance Segmentation",
    layout="wide",
)

st.title("Steel Beam Instance Segmentation")
st.write(
    """
App ini menggunakan YOLOv8 Instance Segmentation untuk mendeteksi dan
memberikan mask pada beberapa jenis profil baja:

- I-beam  
- L-beam  
- O-beam  
- O-pipe  
- T-beam  
- Square-bar  
- Square-pipe  

Upload gambar, lalu model akan menampilkan hasil segmentasi.
"""
)


@st.cache_resource
def load_model():
    model_path = "weights/best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model tidak ditemukan di: {model_path}")
        st.stop()
    return YOLO(model_path)


model = load_model()

# Sidebar pengaturan
with st.sidebar:
    st.header("Settings")

    conf = st.slider(
        "Confidence threshold",
        0.05,
        0.95,
        0.35,
        0.05,
        help="Semakin besar maka prediksi lebih tepat, tapi objek kecil bisa terlewat.",
    )

    iou = st.slider(
        "IoU threshold",
        0.05,
        0.95,
        0.50,
        0.05,
        help="Mengatur seberapa besar overlap yang dianggap deteksi yang sama.",
    )

    imgsz = st.selectbox(
        "Image size",
        [320, 416, 512],
        index=0,
        help="320 lebih cepat, 512 lebih detail namun lebih berat.",
    )

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_choice = st.radio(
        "Device",
        ["auto", "cuda", "cpu"],
        index=0,
        help="Auto akan menggunakan GPU jika tersedia, jika tidak akan memakai CPU.",
    )

    if device_choice == "auto":
        device = default_device
    else:
        device = device_choice


uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)

    # Simpan sementara untuk diproses YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    results = model.predict(
        source=temp_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    os.remove(temp_path)

    out_img = results[0].plot()          # BGR (numpy array)
    out_img = Image.fromarray(out_img[:, :, ::-1])  # ke RGB

    with col2:
        st.subheader("Segmentation result")
        st.image(out_img, use_column_width=True)
else:

    st.info("Silakan upload gambar untuk mulai melakukan segmentasi.")
