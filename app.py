import streamlit as st
import numpy as np
import torch
import os
import gdown
from PIL import Image
import streamlit as st
import torch
from vit_model import model as base_model


# -------------------------------
# Download model if not present
# -------------------------------
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1V7M85rGTMWwHyjwVfLVAp_5Qj6oh0SOS/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model():
    model = base_model   # initialize model structure

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model
# -------------------------------
# Load model
# -------------------------------
# @st.cache_resource
# def load_model():
#     model = torch.load(MODEL_PATH, map_location="cpu")
#     model.eval()
#     return model


# @st.cache_resource
# def load_model():
#     model = base_model  # initialize architecture

#     # state_dict = torch.load("best_vit_finetuned.pth", map_location="cpu")
#     # model.load_state_dict(state_dict)

#     model.eval()
#     return model
# model = load_model()

# -------------------------------
# UI
# -------------------------------
st.title("🧠 NeuroScan AI")
st.subheader("Brain Tumor Classification + XAI + Report")

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Analyze MRI"):
        st.write("Analyzing...")

        # Dummy prediction (replace with your model logic)
        prediction = "Meningioma"
        confidence = 94.2

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence}%")

        # Later you can call:
        # shap_model
        # generate_report()
