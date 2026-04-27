import numpy as np
import os
from groq import Groq

# ── Brain region atlas (224×224 image space) ──────────────────────────────────
# Divides the MRI into anatomical zones based on typical axial slice coordinates
# These zones are approximate for axial brain MRI views

BRAIN_ATLAS = [
    # (x_min, x_max, y_min, y_max, region_name, hemisphere)
    # Frontal lobe
    (0,   112, 0,   80,  "Frontal Lobe",          "Left"),
    (112, 224, 0,   80,  "Frontal Lobe",          "Right"),

    # Parietal lobe
    (0,   112, 80,  140, "Parietal Lobe",          "Left"),
    (112, 224, 80,  140, "Parietal Lobe",          "Right"),

    # Temporal lobe
    (0,   70,  100, 180, "Temporal Lobe",          "Left"),
    (154, 224, 100, 180, "Temporal Lobe",          "Right"),

    # Occipital lobe
    (0,   112, 160, 224, "Occipital Lobe",         "Left"),
    (112, 224, 160, 224, "Occipital Lobe",         "Right"),

    # Central structures
    (70,  154, 80,  160, "Central Structures",     "Midline"),

    # Cerebellum
    (40,  184, 170, 224, "Cerebellum",             "Midline"),

    # Brainstem
    (90,  134, 150, 200, "Brainstem",              "Midline"),
]

SUBREGION_MAP = {
    "Frontal Lobe": [
        "Prefrontal Cortex",
        "Primary Motor Cortex (Precentral Gyrus)",
        "Broca's Area (dominant hemisphere)",
        "Orbitofrontal Cortex",
    ],
    "Parietal Lobe": [
        "Primary Somatosensory Cortex (Postcentral Gyrus)",
        "Superior Parietal Lobule",
        "Inferior Parietal Lobule",
        "Precuneus",
    ],
    "Temporal Lobe": [
        "Superior Temporal Gyrus",
        "Wernicke's Area (dominant hemisphere)",
        "Hippocampus",
        "Amygdala",
    ],
    "Occipital Lobe": [
        "Primary Visual Cortex (V1)",
        "Visual Association Areas",
        "Cuneus",
    ],
    "Central Structures": [
        "Thalamus",
        "Basal Ganglia",
        "Internal Capsule",
        "Corpus Callosum",
    ],
    "Cerebellum": [
        "Cerebellar Hemispheres",
        "Vermis",
        "Cerebellar Tonsils",
    ],
    "Brainstem": [
        "Midbrain",
        "Pons",
        "Medulla Oblongata",
    ],
}


def get_brain_region(shap_map):
    """
    Finds peak activation in SHAP heatmap and maps to brain region.
    shap_map: (224,224) numpy array, values in [-1,1]
    Returns: (region, hemisphere, subregion, peak_x, peak_y)
    """
    # Find top 5% activation area (more robust than single peak)
    threshold = np.percentile(shap_map, 95)
    hot_pixels = np.argwhere(shap_map >= threshold)   # (N,2) [row,col]

    if len(hot_pixels) == 0:
        # fallback to absolute peak
        peak_y, peak_x = np.unravel_index(shap_map.argmax(), shap_map.shape)
    else:
        peak_y = int(hot_pixels[:, 0].mean())
        peak_x = int(hot_pixels[:, 1].mean())

    # Match to atlas
    region     = "Unspecified Region"
    hemisphere = "Unknown"

    for (x0, x1, y0, y1, reg, hemi) in BRAIN_ATLAS:
        if x0 <= peak_x < x1 and y0 <= peak_y < y1:
            region     = reg
            hemisphere = hemi
            break

    # Pick most relevant subregion
    subregions = SUBREGION_MAP.get(region, ["Cortical Surface"])
    # Choose based on position within region
    idx        = int((peak_y / 224) * len(subregions))
    idx        = min(idx, len(subregions) - 1)
    subregion  = subregions[idx]

    return region, hemisphere, subregion, peak_x, peak_y


def generate_report(api_key, tumor_class, confidence, shap_map):
    """
    Maps SHAP heatmap → brain region → calls Groq LLM → returns clinical report.

    Parameters:
        api_key     : Groq API key string
        tumor_class : e.g. "Meningioma"
        confidence  : float e.g. 94.2
        shap_map    : (224,224) numpy array from shap_model.py

    Returns:
        report_text : str — full neurosurgical report
        region_info : dict — anatomical location details
    """
    # ── 1. Map heatmap to brain region ────────────────────────────────────────
    region, hemisphere, subregion, px, py = get_brain_region(shap_map)

    region_info = {
        "region":     region,
        "hemisphere": hemisphere,
        "subregion":  subregion,
        "peak_x":     px,
        "peak_y":     py,
    }

    # ── 2. Build prompt ───────────────────────────────────────────────────────
    prompt = f"""You are an expert neurosurgeon and medical educator writing a structured 
educational report for neurosurgery students.

Based on AI analysis of an MRI scan, the following findings were identified:

TUMOR TYPE     : {tumor_class}
CONFIDENCE     : {confidence:.1f}%
BRAIN REGION   : {hemisphere} {region}
SPECIFIC AREA  : {subregion}

Write a detailed educational neurosurgical report with these exact sections:

1. DIAGNOSIS
   - State the tumor type and what it means

2. ANATOMICAL LOCATION
   - Describe the exact location using proper neuroanatomical terminology
   - Mention the {hemisphere} {region}, specifically the {subregion}
   - Describe surrounding structures at risk

3. CLINICAL PRESENTATION
   - What symptoms would a patient with this tumor in this location typically show?
   - Include neurological deficits specific to this brain region

4. RADIOLOGICAL FEATURES
   - Describe typical MRI appearance of this tumor type
   - What to look for on T1, T2, and contrast sequences

5. SURGICAL CONSIDERATIONS
   - Typical surgical approach for this location
   - Key risks and eloquent areas to preserve
   - Expected surgical outcome

6. EDUCATIONAL NOTE FOR STUDENTS
   - Key learning points about this tumor type and location
   - Why this location makes it clinically significant

Write in a professional but educational tone suitable for neurosurgery students.
Be specific with anatomical terms. Keep each section concise but informative."""

    # ── 3. Call Groq API ──────────────────────────────────────────────────────
    client = Groq(api_key="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")

    chat   = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,      # low temp = more factual, less creative
        max_tokens=1500,
    )

    report_text = chat.choices[0].message.content
    return report_text, region_info