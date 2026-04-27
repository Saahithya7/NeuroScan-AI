# import torch
# import shap
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from PIL import Image

# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# CLASS_NAMES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


# def explain(model, image, device):
#     model.eval()

#     # ── 1. Prepare image ──────────────────────────────────────────────────────
#     img_pil = image.convert("RGB").resize((224, 224))
#     img_np  = np.array(img_pil, dtype=np.float32) / 255.0   # (224,224,3)

#     # ── 2. predict_fn — exact same as working Colab cell ─────────────────────
#     def predict_fn(images):
#         normed = (images - IMAGENET_MEAN) / IMAGENET_STD
#         x = torch.tensor(normed, dtype=torch.float32).permute(0,3,1,2).to(device)
#         with torch.no_grad():
#             probs = torch.softmax(model(x), dim=1)
#         return probs.cpu().numpy()

#     # ── 3. Predicted class ────────────────────────────────────────────────────
#     preds      = predict_fn(img_np[np.newaxis])
#     pred_class = int(np.argmax(preds[0]))
#     confidence = float(preds[0, pred_class]) * 100

#     # ── 4. SHAP ───────────────────────────────────────────────────────────────
#     masker    = shap.maskers.Image("blur(64,64)", img_np.shape)
#     explainer = shap.Explainer(predict_fn, masker)

#     shap_values = explainer(
#         img_np[np.newaxis],
#         max_evals=500,
#         batch_size=50
#     )

#     # ── 5. Build heatmap — exact same as working Colab cell ───────────────────
#     sv     = shap_values.values[0, :, :, :, pred_class]  # (224,224,3)
#     signed = sv.sum(axis=2)                               # (224,224)

#     low  = np.percentile(signed, 1)
#     high = np.percentile(signed, 99)
#     signed_clipped = np.clip(signed, low, high)
#     abs_max = max(abs(low), abs(high)) + 1e-8
#     signed_norm = signed_clipped / abs_max                # spans [-1, 1]

#     # ── 6. Plot ───────────────────────────────────────────────────────────────
#     fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
#     fig.patch.set_facecolor("#0d0d0d")

#     for ax in axes:
#         ax.axis("off")
#         ax.set_facecolor("#0d0d0d")

#     axes[0].imshow(img_np)
#     axes[0].set_title("Original MRI", color="#aabbcc", fontsize=11, pad=8)

#     im = axes[1].imshow(signed_norm, cmap="RdBu_r", vmin=-1, vmax=1)
#     axes[1].set_title("SHAP Heatmap", color="#aabbcc", fontsize=11, pad=8)
#     cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
#     cbar.ax.tick_params(colors="#aabbcc", labelsize=7)
#     cbar.set_label("SHAP value", color="#aabbcc", fontsize=8)

#     axes[2].imshow(img_np)
#     axes[2].imshow(signed_norm, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.55)
#     axes[2].set_title(
#         f"{CLASS_NAMES[pred_class]}  ·  {confidence:.1f}%",
#         color="#00d4ff", fontsize=11, pad=8
#     )

#     plt.tight_layout(pad=1.5)
#     return fig

import torch
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def explain(model, image, device):
    """
    Returns:
        fig      : matplotlib Figure  (the visual plot)
        shap_map : (224,224) numpy array of signed importance values
                   — passed to generate_report() in llm_report.py
    """
    model.eval()

    # ── 1. Prepare image ──────────────────────────────────────────────────────
    img_pil = image.convert("RGB").resize((224, 224))
    img_np  = np.array(img_pil, dtype=np.float32) / 255.0   # (224,224,3)

    # ── 2. predict_fn ─────────────────────────────────────────────────────────
    def predict_fn(images):
        normed = (images - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.tensor(normed, dtype=torch.float32).permute(0,3,1,2).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        return probs.cpu().numpy()

    # ── 3. Predicted class ────────────────────────────────────────────────────
    preds      = predict_fn(img_np[np.newaxis])
    pred_class = int(np.argmax(preds[0]))
    confidence = float(preds[0, pred_class]) * 100

    # ── 4. SHAP ───────────────────────────────────────────────────────────────
    masker    = shap.maskers.Image("blur(64,64)", img_np.shape)
    explainer = shap.Explainer(predict_fn, masker)

    shap_values = explainer(
        img_np[np.newaxis],
        max_evals=500,
        batch_size=50
    )

    # ── 5. Build heatmap ──────────────────────────────────────────────────────
    sv     = shap_values.values[0, :, :, :, pred_class]  # (224,224,3)
    signed = sv.sum(axis=2)                               # (224,224)

    low  = np.percentile(signed, 1)
    high = np.percentile(signed, 99)
    signed_clipped = np.clip(signed, low, high)
    abs_max = max(abs(low), abs(high)) + 1e-8
    signed_norm = signed_clipped / abs_max                # [-1, 1]

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("#0d0d0d")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    axes[0].imshow(img_np)
    axes[0].set_title("Original MRI", color="#aabbcc", fontsize=11, pad=8)

    im = axes[1].imshow(signed_norm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("SHAP Heatmap", color="#aabbcc", fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#aabbcc", labelsize=7)
    cbar.set_label("SHAP value", color="#aabbcc", fontsize=8)

    axes[2].imshow(img_np)
    axes[2].imshow(signed_norm, cmap="RdBu_r", vmin=-1, vmax=1, alpha=0.55)
    axes[2].set_title(
        f"{CLASS_NAMES[pred_class]}  ·  {confidence:.1f}%",
        color="#00d4ff", fontsize=11, pad=8
    )

    plt.tight_layout(pad=1.5)

    # Return BOTH the figure and the raw signed map for the LLM report
    return fig, signed_norm