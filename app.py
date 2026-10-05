import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json


# ============================================================
#  PAGE SETUP
# ============================================================
st.set_page_config(page_title="Skin Tone Segmentation (Final)", layout="wide")
st.title("🧍 Final Skin Tone Segmentation System")
st.write("""
This system performs skin detection using **HSV**, **YCrCb**,  
logical combinations (**AND**, **OR**), and an **Automatic Selector**.
Now enhanced with **Pre-processing**, **Post-processing**, **Quality Score**,  
and **Comparison Panel**.
""")

# ============================================================
#  OUTPUT DIRECTORY
# ============================================================
RESULTS_DIR = r"c:\\xampp\\htdocs\\image_processing\\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
#  PREPROCESSING FUNCTION
# ============================================================
def preprocess_image(img, blur=False, bright=0, cont=0, wb=False, hist_eq=False):
    out = img.copy()

    # Brightness + Contrast
    if bright != 0 or cont != 0:
        out = cv2.convertScaleAbs(out, alpha=1 + (cont / 100), beta=bright)

    # Gaussian blur
    if blur:
        out = cv2.GaussianBlur(out, (5, 5), 0)

    # White balance
    if wb:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Histogram equalization Y channel
    if hist_eq:
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycc)
        y = cv2.equalizeHist(y)
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    return out

# ============================================================
#  POSTPROCESS FUNCTION
# ============================================================
def postprocess_mask(mask, open_m=False, close_m=False, smooth=False):
    kernel = np.ones((5, 5), np.uint8)
    out = mask.copy()

    if open_m:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_m:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if smooth:
        out = cv2.medianBlur(out, 7)

    return out

# ============================================================
#  QUALITY SCORE FUNCTION
# ============================================================
def evaluate_mask(mask):
    m = mask.astype(np.float32) / 255.0

    confidence = m.mean()
    noise = m.var()
    edges = cv2.Canny(mask, 50, 150)
    clarity = edges.mean() / 255.0

    return {
        "Confidence": round(confidence, 3),
        "Noise": round(noise, 3),
        "Clarity": round(clarity, 3)
    }

# ============================================================
#  AUTO METHOD DECIDER
# ============================================================
def choose_best_method(mask_hsv, mask_ycc):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
    clean_ycc = cv2.morphologyEx(mask_ycc, cv2.MORPH_CLOSE, kernel)

    score_hsv = cv2.countNonZero(clean_hsv)
    score_ycc = cv2.countNonZero(clean_ycc)

    return "YCrCb" if score_ycc > score_hsv else "HSV"

# ============================================================
#  SIDEBAR — METHOD & OPTIONS
# ============================================================
st.sidebar.title("⚙️ Method & Options")

# -------------------- Preprocessing --------------------
st.sidebar.subheader("🧼 Pre-processing")
apply_blur = st.sidebar.checkbox("Gaussian Blur", False)
brightness = st.sidebar.slider("Brightness", -50, 50, 0)
contrast = st.sidebar.slider("Contrast", -50, 50, 0)
white_balance = st.sidebar.checkbox("White Balance", False)
hist_equal = st.sidebar.checkbox("Histogram Equalization (Y-channel)", False)

# -------------------- Method --------------------
method = st.sidebar.radio(
    "Segmentation Method",
    ("HSV", "YCrCb", "AND", "OR", "AUTO")
)

# -------------------- HSV sliders --------------------
st.sidebar.subheader("🎚 HSV Range")
h_min = st.sidebar.slider("Hue Min", 0, 179, st.session_state.get("h_min", 0), key="h_min")
h_max = st.sidebar.slider("Hue Max", 0, 179, st.session_state.get("h_max", 25), key="h_max")
s_min = st.sidebar.slider("Saturation Min", 0, 255, st.session_state.get("s_min", 20), key="s_min")
s_max = st.sidebar.slider("Saturation Max", 0, 255, st.session_state.get("s_max", 255), key="s_max")
v_min = st.sidebar.slider("Value Min", 0, 255, st.session_state.get("v_min", 70), key="v_min")
v_max = st.sidebar.slider("Value Max", 0, 255, st.session_state.get("v_max", 255), key="v_max")

# -------------------- YCrCb sliders --------------------
st.sidebar.subheader("🎚 YCrCb Range")
cr_min = st.sidebar.slider("Cr Min", 0, 255, st.session_state.get("cr_min", 135), key="cr_min")
cr_max = st.sidebar.slider("Cr Max", 0, 255, st.session_state.get("cr_max", 180), key="cr_max")
cb_min = st.sidebar.slider("Cb Min", 0, 255, st.session_state.get("cb_min", 85), key="cb_min")
cb_max = st.sidebar.slider("Cb Max", 0, 255, st.session_state.get("cb_max", 135), key="cb_max")


# -------------------- Small area remove --------------------
min_area = st.sidebar.slider(
    "Minimum Area to Keep (px)",
    100, 2000,
    st.session_state.get("min_area", 500),
    key="min_area"
)

# -------------------- Post-processing --------------------
st.sidebar.subheader("✨ Post-processing")
apply_open = st.sidebar.checkbox("Morphology Opening")
apply_close = st.sidebar.checkbox("Morphology Closing")
apply_smooth = st.sidebar.checkbox("Smooth Mask (Median Blur)")

# ============================================================
# DEFAULT VALUES FOR SLIDERS (Store in session_state)
# ============================================================
if "defaults" not in st.session_state:
    st.session_state.defaults = {
        "h_min": 0,
        "h_max": 25,
        "s_min": 20,
        "s_max": 255,
        "v_min": 70,
        "v_max": 255,
        "cr_min": 135,
        "cr_max": 180,
        "cb_min": 85,
        "cb_max": 135,
        "min_area": 500
    }

# Reset button
if st.sidebar.button("🔄 Reset Sliders to Default"):
    for k, v in st.session_state.defaults.items():
        st.session_state[k] = v
    st.experimental_rerun()


# ============================================================
#  FILE UPLOADER
# ============================================================
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("⬆️ Upload an image to start.")
    st.stop()

# Read image
image = np.array(Image.open(uploaded).convert("RGB"))

# Apply preprocessing
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image_pre = preprocess_image(
    image_bgr,
    blur=apply_blur,
    bright=brightness,
    cont=contrast,
    wb=white_balance,
    hist_eq=hist_equal
)

image = cv2.cvtColor(image_pre, cv2.COLOR_BGR2RGB)

st.image(image, caption="Pre-processed Image", use_column_width=True)

# ============================================================
#  COLOR SPACE CONVERSION
# ============================================================
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# Build ranges
lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)

lower_ycc = np.array([0, cr_min, cb_min], dtype=np.uint8)
upper_ycc = np.array([255, cr_max, cb_max], dtype=np.uint8)

# Create masks
mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
mask_ycc = cv2.inRange(ycrcb, lower_ycc, upper_ycc)

# Logical methods
if method == "HSV":
    mask = mask_hsv
elif method == "YCrCb":
    mask = mask_ycc
elif method == "AND":
    mask = cv2.bitwise_and(mask_hsv, mask_ycc)
elif method == "OR":
    mask = cv2.bitwise_or(mask_hsv, mask_ycc)
else:
    best = choose_best_method(mask_hsv, mask_ycc)
    mask = mask_hsv if best == "HSV" else mask_ycc

# ============================================================
#  POST-PROCESS MASK
# ============================================================
mask = postprocess_mask(mask, apply_open, apply_close, apply_smooth)

# Remove small blobs
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < min_area:
        cv2.drawContours(mask, [c], -1, 0, -1)

# Apply mask
result = cv2.bitwise_and(image, image, mask=mask)

# Skin ratio
skin_pixels = cv2.countNonZero(mask)
ratio = (skin_pixels / mask.size) * 100

# ============================================================
#  DISPLAY RESULTS
# ============================================================
st.header(f"📊 Segmentation Result — Method: **{method}**")

col1, col2 = st.columns(2)

with col1:
    st.image(result, caption=f"Final Segmented Result — {ratio:.2f}% Skin", use_column_width=True)

with col2:
    st.image(mask, caption="Final Mask", use_column_width=True)

# ============================================================
#  QUALITY SCORE
# ============================================================
st.subheader("📈 Quality Score")
scores = evaluate_mask(mask)
st.json(scores)

# ============================================================
#  COMPARISON PANEL
# ============================================================
st.subheader("🖼️ Comparison Panel")

hsv_res = cv2.bitwise_and(image, image, mask=mask_hsv)
ycc_res = cv2.bitwise_and(image, image, mask=mask_ycc)
and_res = cv2.bitwise_and(hsv_res, ycc_res)
or_res = cv2.bitwise_or(hsv_res, ycc_res)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(hsv_res, caption="HSV Output")
with col2:
    st.image(ycc_res, caption="YCrCb Output")
with col3:
    st.image(and_res, caption="AND Output")
with col4:
    st.image(or_res, caption="OR Output")
    
# ============================================================
#  SAVE JSON SETTINGS
# ============================================================
if st.button("💾 Export Settings to JSON"):
    settings = {
        "method": method,
        "HSV": {
            "h_min": int(h_min),
            "h_max": int(h_max),
            "s_min": int(s_min),
            "s_max": int(s_max),
            "v_min": int(v_min),
            "v_max": int(v_max),
        },
        "YCrCb": {
            "cr_min": int(cr_min),
            "cr_max": int(cr_max),
            "cb_min": int(cb_min),
            "cb_max": int(cb_max),
        },
        "min_area": int(min_area),
        "skin_ratio_percent": float(ratio)
    }

    json_path = os.path.join(RESULTS_DIR, f"settings_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(json_path, "w") as f:
        json.dump(settings, f, indent=4)

    st.success(f"Settings exported to JSON:\n{json_path}")
# ============================================================
#  SAVE OUTPUTS
# ============================================================
if st.button("💾 Save Results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(RESULTS_DIR, f"mask_{method}_{ts}.jpg"), mask)
    cv2.imwrite(os.path.join(RESULTS_DIR, f"segmented_{method}_{ts}.jpg"),
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    st.success(f"Results saved in {RESULTS_DIR}")
