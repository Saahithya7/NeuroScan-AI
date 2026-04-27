# # import streamlit as st
# # import numpy as np
# # import torch
# # import os
# # import gdown
# # from PIL import Image
# # import streamlit as st
# # import torch
# # from vit_model import model as base_model


# # # -------------------------------
# # # Download model if not present
# # # -------------------------------
# # MODEL_PATH = "model.pth"

# # if not os.path.exists(MODEL_PATH):
# #     url = "https://drive.google.com/file/d/1V7M85rGTMWwHyjwVfLVAp_5Qj6oh0SOS"
# #     gdown.download(url, MODEL_PATH, quiet=False)


# # @st.cache_resource
# # def load_model():
# #     model = base_model   # initialize model structure

# #     state_dict = torch.load(MODEL_PATH, map_location="cpu")
# #     model.load_state_dict(state_dict)

# #     model.eval()
# #     return model
# # # -------------------------------
# # # Load model
# # # -------------------------------
# # # @st.cache_resource
# # # def load_model():
# # #     model = torch.load(MODEL_PATH, map_location="cpu")
# # #     model.eval()
# # #     return model


# # # @st.cache_resource
# # # def load_model():
# # #     model = base_model  # initialize architecture

# # #     # state_dict = torch.load("best_vit_finetuned.pth", map_location="cpu")
# # #     # model.load_state_dict(state_dict)

# # #     model.eval()
# # #     return model
# # # model = load_model()

# # # -------------------------------
# # # UI
# # # -------------------------------
# # st.title("🧠 NeuroScan AI")
# # st.subheader("Brain Tumor Classification + XAI + Report")

# # uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file)
# #     st.image(image, caption="Uploaded MRI", use_container_width=True)

# #     if st.button("🔍 Analyze MRI"):
# #         st.write("Analyzing...")

# #         # Dummy prediction (replace with your model logic)
# #         prediction = "Meningioma"
# #         confidence = 94.2

# #         st.success(f"Prediction: {prediction}")
# #         st.info(f"Confidence: {confidence}%")

# #         # Later you can call:
# #         # shap_model
# #         # generate_report()

# import streamlit as st
# import numpy as np
# import torch
# import os
# import gdown
# from PIL import Image
# from vit_model import model as base_model

# import matplotlib
# matplotlib.use("Agg")

# from vit_model import model, predict, CLASS_NAMES, device
# from shap_model import explain
# from attention_map import generate_attention_map
# from llm_report import generate_report
# # -------------------------------
# # Download model if not present
# # -------------------------------
# MODEL_PATH = "model.pth"

# if not os.path.exists(MODEL_PATH):
#     # ✅ Fixed: use the direct download URL format
#     FILE_ID = "1V7M85rGTMWwHyjwVfLVAp_5Qj6oh0SOS"
#     url = f"https://drive.google.com/uc?id={FILE_ID}"
#     gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# # -------------------------------
# # Load model
# # -------------------------------
# @st.cache_resource
# def load_model():
#     model = base_model
#     state_dict = torch.load(MODEL_PATH, map_location="cpu")
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# model = load_model()  # ✅ Fixed: this was commented out before

# # -------------------------------
# # UI
# # -------------------------------
# st.title("🧠 NeuroScan AI")
# st.subheader("Brain Tumor Classification + XAI + Report")

# uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded MRI", use_container_width=True)

#     if st.button("🔍 Analyze MRI"):
#         st.write("Analyzing...")

#         # Dummy prediction (replace with your model logic)
#         prediction = "Meningioma"
#         confidence = 94.2

#         st.success(f"Prediction: {prediction}")
#         st.info(f"Confidence: {confidence}%")


import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from vit_model import model, predict, CLASS_NAMES, device
from shap_model import explain
from attention_map import generate_attention_map
from llm_report import generate_report

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 NeuroScan AI")
st.subheader("Brain Tumor Classification · XAI · LLM Report Generation")

# ── Groq API Key ──────────────────────────────────────────────────────────────
api_key = st.text_input("Enter your Groq API key", type="password",
                         placeholder="gsk_xxxxxxxxxxxxxxxxxx")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["image"] = image

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Input MRI", use_container_width=True)

    with col2:
        # ── Classify ──────────────────────────────────────────────────────────
        if st.button("🔍 Classify Tumor"):
            with st.spinner("Running ViT inference..."):
                label, conf = predict(image)
            st.session_state["result"]     = label
            st.session_state["confidence"] = conf

        # ── Attention Map ─────────────────────────────────────────────────────
        if st.button("🌡 Gradient Attention Map"):
            with st.spinner("Computing gradients..."):
                attn = generate_attention_map(model, image, device)
            st.session_state["attn"] = attn

        # ── SHAP ──────────────────────────────────────────────────────────────
        st.warning("⏱ SHAP takes ~6 minutes on CPU. Click and wait.")
        if st.button("🔬 SHAP Explanation (~6 min)"):
            with st.spinner("Running SHAP — please wait ~6 minutes..."):
                try:
                    fig, shap_map = explain(model, image, device)
                    st.session_state["shap_fig"] = fig
                    st.session_state["shap_map"] = shap_map
                    st.session_state.pop("shap_err", None)
                except Exception as e:
                    st.session_state["shap_err"] = str(e)

        # ── LLM Report ────────────────────────────────────────────────────────
        if st.button("📋 Generate Neurosurgical Report"):
            if not api_key:
                st.warning("Please enter your Groq API key above.")
            elif "shap_map" not in st.session_state:
                st.warning("Please run SHAP Explanation first.")
            elif "result" not in st.session_state:
                st.warning("Please run Classify Tumor first.")
            else:
                with st.spinner("Generating report via Llama 3.1 70B..."):
                    try:
                        report, region_info = generate_report(
                            api_key,
                            st.session_state["result"],
                            st.session_state["confidence"],
                            st.session_state["shap_map"]
                        )
                        st.session_state["report"]      = report
                        st.session_state["region_info"] = region_info
                        st.session_state.pop("report_err", None)
                    except Exception as e:
                        st.session_state["report_err"] = str(e)

    # ── Result ────────────────────────────────────────────────────────────────
    if "result" in st.session_state:
        st.success(f"Prediction: {st.session_state['result']}")
        st.info(f"Confidence: {st.session_state['confidence']:.1f}%")

    # ── Attention Map ─────────────────────────────────────────────────────────
    if "attn" in st.session_state:
        st.subheader("Gradient Attention Map")
        attn    = st.session_state["attn"]
        img_np  = np.array(st.session_state["image"].resize((224, 224)))
        heatmap = plt.cm.jet(attn)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)
        c1, c2  = st.columns(2)
        c1.image(np.uint8(255 * attn), caption="Raw Attention", use_container_width=True, clamp=True)
        c2.image(overlay, caption="Overlay", use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if "shap_err" in st.session_state:
        st.error(f"SHAP error: {st.session_state['shap_err']}")
    if "shap_fig" in st.session_state:
        st.subheader("SHAP Explanation")
        st.pyplot(st.session_state["shap_fig"])

    # ── Report ────────────────────────────────────────────────────────────────
    if "report_err" in st.session_state:
        st.error(f"Report error: {st.session_state['report_err']}")
    if "report" in st.session_state:
        st.subheader("📋 Neurosurgical Report")
        st.write(st.session_state["report"])
        st.download_button("⬇️ Download Report",
                           data=st.session_state["report"],
                           file_name="neurosurgical_report.txt",
                           mime="text/plain")
else:
    st.info("Upload an MRI scan to begin.")
