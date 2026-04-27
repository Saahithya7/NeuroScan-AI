# import streamlit as st
# from PIL import Image
# import torch
# import numpy as np

# import matplotlib
# matplotlib.use("Agg")

# from vit_model import model, predict, CLASS_NAMES, device
# from shap_model import explain
# from attention_map import generate_attention_map

# st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

# html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#080c10; color:#e8edf2; }
# .stApp { background:#080c10; }

# .hero-title {
#     font-size:2.4rem; font-weight:800; letter-spacing:-1px;
#     background:linear-gradient(135deg,#00d4ff 0%,#7b61ff 60%,#ff6b9d 100%);
#     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
# }
# .hero-sub {
#     font-family:'Space Mono',monospace; font-size:0.75rem;
#     color:#3a5060; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1.8rem;
# }
# .result-card {
#     background:linear-gradient(135deg,#0f1a2e,#111827);
#     border:1px solid #1e3a5f; border-radius:16px; padding:1.2rem 1.6rem; margin:1rem 0;
# }
# .result-label { font-family:'Space Mono',monospace; font-size:0.68rem; color:#4a6080;
#     text-transform:uppercase; letter-spacing:0.14em; margin-bottom:4px; }
# .result-class { font-size:1.9rem; font-weight:800; }
# .result-conf  { font-family:'Space Mono',monospace; font-size:0.85rem; color:#4a7aaa; margin-top:2px; }

# .clr-Glioma     { color:#ff6b6b; }
# .clr-Meningioma { color:#ffd93d; }
# .clr-NoTumor    { color:#6bffb8; }
# .clr-Pituitary  { color:#c77dff; }

# .sec { font-family:'Space Mono',monospace; font-size:0.7rem; color:#3a5060;
#     text-transform:uppercase; letter-spacing:0.16em; margin:1.4rem 0 0.5rem;
#     display:flex; align-items:center; gap:8px; }
# .sec::after { content:''; flex:1; height:1px; background:#1a2a3a; }

# .warning-box {
#     background:#1a1200; border:1px solid #7a5800; border-radius:10px;
#     padding:0.8rem 1rem; margin-bottom:0.8rem;
#     font-family:'Space Mono',monospace; font-size:0.75rem; color:#c8950a;
# }

# div.stButton > button {
#     background:linear-gradient(135deg,#0077cc,#7b61ff); color:white;
#     border:none; border-radius:10px; font-family:'Space Mono',monospace;
#     font-size:0.8rem; font-weight:700; padding:0.55rem 1.2rem; width:100%;
# }
# [data-testid="stFileUploader"] { background:#0d1520; border:1.5px dashed #1e3a5f; border-radius:14px; padding:0.8rem; }
# [data-testid="stImage"] img    { border-radius:12px; border:1px solid #1e3a5f; }
# </style>
# """, unsafe_allow_html=True)

# # ── Header ────────────────────────────────────────────────────────────────────
# st.markdown('<div class="hero-title">🧠 NeuroScan AI</div>', unsafe_allow_html=True)
# st.markdown('<div class="hero-sub">Brain Tumor Classification · XAI Dashboard</div>', unsafe_allow_html=True)

# # ── Upload ────────────────────────────────────────────────────────────────────
# st.markdown('<div class="sec">Upload MRI Scan</div>', unsafe_allow_html=True)
# uploaded_file = st.file_uploader("JPG or PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.session_state["image"] = image

#     col1, col2 = st.columns([1, 1.3])

#     with col1:
#         st.image(image, caption="Input MRI", use_container_width=True)

#     with col2:
#         st.markdown('<div class="sec">Actions</div>', unsafe_allow_html=True)

#         # ── Classify ──────────────────────────────────────────────────────────
#         if st.button("🔍  Classify Tumor"):
#             with st.spinner("Running ViT inference..."):
#                 label, conf = predict(image)
#             st.session_state["result"]     = label
#             st.session_state["confidence"] = conf

#         # ── Attention Map ─────────────────────────────────────────────────────
#         if st.button("🌡  Gradient Attention Map"):
#             with st.spinner("Computing gradients..."):
#                 attn = generate_attention_map(model, image, device)
#             st.session_state["attn"] = attn

#         # ── SHAP ──────────────────────────────────────────────────────────────
#         st.markdown("""
#         <div class="warning-box">
#             ⏱ SHAP takes ~6 minutes on CPU.<br>
#             Click and wait — do not close the browser.
#         </div>""", unsafe_allow_html=True)

#         if st.button("🔬  SHAP Explanation  (slow ~6 min)"):
#             with st.spinner("Running SHAP — please wait ~6 minutes on CPU..."):
#                 try:
#                     fig = explain(model, image, device)
#                     st.session_state["shap_fig"] = fig
#                     st.session_state.pop("shap_err", None)
#                 except Exception as e:
#                     st.session_state["shap_err"] = str(e)

#     # ── Result card ───────────────────────────────────────────────────────────
#     if "result" in st.session_state:
#         label = st.session_state["result"]
#         conf  = st.session_state["confidence"]
#         css   = "clr-" + label.replace(" ", "")

#         st.markdown(f"""
#         <div class="result-card">
#             <div class="result-label">Predicted Diagnosis</div>
#             <div class="result-class {css}">{label}</div>
#             <div class="result-conf">Confidence: {conf:.1f}%</div>
#         </div>""", unsafe_allow_html=True)
#         st.info("⚠️ Research tool only — not for clinical diagnosis.")

#     # ── Attention map ─────────────────────────────────────────────────────────
#     if "attn" in st.session_state:
#         st.markdown('<div class="sec">Gradient Attention Map</div>', unsafe_allow_html=True)
#         attn   = st.session_state["attn"]
#         img_np = np.array(st.session_state["image"].resize((224, 224)))


#         c1, c2 = st.columns(2)
#         c1.image(np.uint8(255 * attn), caption="Raw Attention", use_container_width=True, clamp=True)
#         c2.image(overlay, caption="Overlay", use_container_width=True)

#     # ── SHAP error ────────────────────────────────────────────────────────────
#     if "shap_err" in st.session_state:
#         st.error(f"SHAP error: {st.session_state['shap_err']}")

#     # ── SHAP result ───────────────────────────────────────────────────────────
#     if "shap_fig" in st.session_state:
#         st.markdown('<div class="sec">SHAP Explanation</div>', unsafe_allow_html=True)
#         st.pyplot(st.session_state["shap_fig"])
#         st.caption("🔴 Red = pushed prediction toward this class  ·  🔵 Blue = pushed away")

# else:
#     st.markdown("""
#     <div style="text-align:center;padding:3rem 1rem;color:#1e3040;">
#         <div style="font-size:3rem;">🧬</div>
#         <div style="font-family:'Space Mono',monospace;font-size:0.78rem;letter-spacing:0.1em;">
#             Upload an MRI scan to begin
#         </div>
#     </div>""", unsafe_allow_html=True)

import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

from vit_model import model, predict, CLASS_NAMES, device
from shap_model import explain
from attention_map import generate_attention_map
from llm_report import generate_report
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

heatmap = plt.cm.jet(attn)[:, :, :3]
heatmap = (heatmap * 255).astype(np.uint8)

overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)

st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#080c10; color:#e8edf2; }
.stApp { background:#080c10; }

.hero-title {
    font-size:2.4rem; font-weight:800; letter-spacing:-1px;
    background:linear-gradient(135deg,#00d4ff 0%,#7b61ff 60%,#ff6b9d 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub {
    font-family:'Space Mono',monospace; font-size:0.75rem;
    color:#3a5060; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:1.8rem;
}
.result-card {
    background:linear-gradient(135deg,#0f1a2e,#111827);
    border:1px solid #1e3a5f; border-radius:16px; padding:1.2rem 1.6rem; margin:1rem 0;
}
.result-label { font-family:'Space Mono',monospace; font-size:0.68rem; color:#4a6080;
    text-transform:uppercase; letter-spacing:0.14em; margin-bottom:4px; }
.result-class { font-size:1.9rem; font-weight:800; }
.result-conf  { font-family:'Space Mono',monospace; font-size:0.85rem; color:#4a7aaa; margin-top:2px; }

.clr-Glioma     { color:#ff6b6b; }
.clr-Meningioma { color:#ffd93d; }
.clr-NoTumor    { color:#6bffb8; }
.clr-Pituitary  { color:#c77dff; }

.location-card {
    background:#0a1628;
    border:1px solid #1e3a5f;
    border-left: 4px solid #00d4ff;
    border-radius:12px; padding:1rem 1.4rem; margin:0.8rem 0;
}
.location-title { font-family:'Space Mono',monospace; font-size:0.68rem;
    color:#4a6080; text-transform:uppercase; letter-spacing:0.14em; margin-bottom:8px; }
.location-region { font-size:1.3rem; font-weight:800; color:#00d4ff; }
.location-sub { font-family:'Space Mono',monospace; font-size:0.78rem; color:#7aaacc; margin-top:4px; }

.report-card {
    background:#070f1a;
    border:1px solid #1e3a5f;
    border-radius:14px; padding:1.4rem 1.8rem; margin:1rem 0;
    font-size:0.9rem; line-height:1.7; color:#c8d8e8;
    white-space: pre-wrap;
}

.sec { font-family:'Space Mono',monospace; font-size:0.7rem; color:#3a5060;
    text-transform:uppercase; letter-spacing:0.16em; margin:1.4rem 0 0.5rem;
    display:flex; align-items:center; gap:8px; }
.sec::after { content:''; flex:1; height:1px; background:#1a2a3a; }

.warning-box {
    background:#1a1200; border:1px solid #7a5800; border-radius:10px;
    padding:0.8rem 1rem; margin-bottom:0.8rem;
    font-family:'Space Mono',monospace; font-size:0.75rem; color:#c8950a;
}

div.stButton > button {
    background:linear-gradient(135deg,#0077cc,#7b61ff); color:white;
    border:none; border-radius:10px; font-family:'Space Mono',monospace;
    font-size:0.8rem; font-weight:700; padding:0.55rem 1.2rem; width:100%;
    margin-bottom: 6px;
}
[data-testid="stFileUploader"] { background:#0d1520; border:1.5px dashed #1e3a5f; border-radius:14px; padding:0.8rem; }
[data-testid="stImage"] img    { border-radius:12px; border:1px solid #1e3a5f; }
[data-testid="stTextInput"] input {
    background:#0d1520; border:1px solid #1e3a5f; border-radius:8px;
    color:#e8edf2; font-family:'Space Mono',monospace; font-size:0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 NeuroScan AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Brain Tumor Classification · XAI · LLM Report Generation</div>', unsafe_allow_html=True)

# ── Groq API Key input ────────────────────────────────────────────────────────
st.markdown('<div class="sec">Groq API Key</div>', unsafe_allow_html=True)
api_key = st.secrets["GROQ_API_KEY"]
# api_key = st.text_input("Enter your Groq API key", type="password",
#                          placeholder="gsk_xxxxxxxxxxxxxxxxxx",
#                          label_visibility="collapsed")

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Upload MRI Scan</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("JPG or PNG", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state["image"] = image

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Input MRI", use_container_width=True)

    with col2:
        st.markdown('<div class="sec">Actions</div>', unsafe_allow_html=True)

        # ── Classify ──────────────────────────────────────────────────────────
        if st.button("🔍  Classify Tumor"):
            with st.spinner("Running ViT inference..."):
                label, conf = predict(image)
            st.session_state["result"]     = label
            st.session_state["confidence"] = conf

        # ── Attention Map ─────────────────────────────────────────────────────
        if st.button("🌡  Gradient Attention Map"):
            with st.spinner("Computing gradients..."):
                attn = generate_attention_map(model, image, device)
            st.session_state["attn"] = attn

        # ── SHAP ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div class="warning-box">
            ⏱ SHAP takes ~6 minutes on CPU.<br>
            Click and wait — do not close the browser.
        </div>""", unsafe_allow_html=True)

        if st.button("🔬  SHAP Explanation  (~6 min)"):
            with st.spinner("Running SHAP — please wait ~6 minutes..."):
                try:
                    fig, shap_map = explain(model, image, device)
                    st.session_state["shap_fig"] = fig
                    st.session_state["shap_map"] = shap_map
                    st.session_state.pop("shap_err", None)
                except Exception as e:
                    st.session_state["shap_err"] = str(e)

        # ── LLM Report ────────────────────────────────────────────────────────
        if st.button("📋  Generate Neurosurgical Report"):
            if not api_key:
                st.warning("Please enter your Groq API key above.")
            elif "shap_map" not in st.session_state:
                st.warning("Please run SHAP Explanation first — the report needs the heatmap.")
            elif "result" not in st.session_state:
                st.warning("Please run Classify Tumor first.")
            else:
                with st.spinner("Generating neurosurgical report via Llama 3.1 70B..."):
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

    # ── Result card ───────────────────────────────────────────────────────────
    if "result" in st.session_state:
        label = st.session_state["result"]
        conf  = st.session_state["confidence"]
        css   = "clr-" + label.replace(" ", "")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Diagnosis</div>
            <div class="result-class {css}">{label}</div>
            <div class="result-conf">Confidence: {conf:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        st.info("⚠️ Research tool only — not for clinical diagnosis.")

    # ── Attention map ─────────────────────────────────────────────────────────
    if "attn" in st.session_state:
        st.markdown('<div class="sec">Gradient Attention Map</div>', unsafe_allow_html=True)
        attn   = st.session_state["attn"]
        img_np = np.array(st.session_state["image"].resize((224, 224)))


        heatmap = plt.cm.jet(attn)[:, :, :3]   # apply colormap
        heatmap = (heatmap * 255).astype(np.uint8)

        overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)
        c1, c2  = st.columns(2)
        c1.image(np.uint8(255 * attn), caption="Raw Attention", use_container_width=True, clamp=True)
        c2.image(overlay, caption="Overlay", use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if "shap_err" in st.session_state:
        st.error(f"SHAP error: {st.session_state['shap_err']}")

    if "shap_fig" in st.session_state:
        st.markdown('<div class="sec">SHAP Explanation</div>', unsafe_allow_html=True)
        st.pyplot(st.session_state["shap_fig"])
        st.caption("🔴 Red = pushed prediction toward this class  ·  🔵 Blue = pushed away")

    # ── Location card ─────────────────────────────────────────────────────────
    if "region_info" in st.session_state:
        r = st.session_state["region_info"]
        st.markdown(f"""
        <div class="location-card">
            <div class="location-title">📍 Detected Anatomical Location</div>
            <div class="location-region">{r['hemisphere']} {r['region']}</div>
            <div class="location-sub">Specific area: {r['subregion']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Report error ──────────────────────────────────────────────────────────
    if "report_err" in st.session_state:
        st.error(f"Report error: {st.session_state['report_err']}")

    # ── LLM Report ────────────────────────────────────────────────────────────
    if "report" in st.session_state:
        st.markdown('<div class="sec">📋 Neurosurgical Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="report-card">{st.session_state["report"]}</div>',
                    unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download Report",
            data=st.session_state["report"],
            file_name="neurosurgical_report.txt",
            mime="text/plain"
        )

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#1e3040;">
        <div style="font-size:3rem;">🧬</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.78rem;letter-spacing:0.1em;">
            Upload an MRI scan to begin
        </div>
    </div>""", unsafe_allow_html=True)
