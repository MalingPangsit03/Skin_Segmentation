import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import gc  # garbage collector

# =====================================================================
# CONFIGURATION
# =====================================================================
DATASET_ROOT = r"C:\xampp\htdocs\image_processing\ISIC2018"
OUTPUT_FILE = "trained_kaggle_dataset.json"

# STORE ONLY 1 OUT OF N PIXELS
PIXEL_SAMPLE_RATE = 20   # lower = more memory used, higher = safer

print("\n📌 ISIC Training (Ultra Memory-Safe Sampling Mode)")
print(f"📌 Sampling rate: 1/{PIXEL_SAMPLE_RATE} pixels\n")


# =====================================================================
# FIND IMAGE/MASK PAIRS
# =====================================================================
def find_pairs(root_dir):
    image_files = {}
    mask_files = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            low = file.lower()

            if low.endswith((".jpg", ".jpeg", ".png")) and "_segmentation" not in low:
                base = os.path.splitext(low)[0]
                image_files[base] = os.path.join(dirpath, file)

            elif "_segmentation" in low and low.endswith((".png", ".jpg")):
                base = low.replace("_segmentation", "").replace(".png", "").replace(".jpg", "")
                mask_files[base] = os.path.join(dirpath, file)

    pairs = []
    for base, img_path in image_files.items():
        if base in mask_files:
            pairs.append((img_path, mask_files[base]))

    return pairs


# =====================================================================
# BUFFERS
# =====================================================================
h_vals, s_vals, v_vals = [], [], []
cr_vals, cb_vals = [], []


# =====================================================================
# LOAD DATASET
# =====================================================================
pairs = find_pairs(DATASET_ROOT)
print(f"🔎 Found {len(pairs)} image/mask pairs.\n")

if len(pairs) == 0:
    print("❌ ERROR: No matching pairs found.")
    exit()


# =====================================================================
# PROCESS IMAGES (ULTRA MEMORY SAFE)
# =====================================================================
MAX_DIM = 800
MAX_FILE_SIZE = 12_000_000

for img_path, mask_path in tqdm(pairs, desc="Processing"):

    # Skip huge files
    if os.path.getsize(img_path) > MAX_FILE_SIZE:
        print(f"\n⚠️ Skipped large image (>12MB): {img_path}")
        continue

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"\n⚠️ Skipping invalid files:\n{img_path}\n{mask_path}")
        continue

    # Resize large images
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Mask threshold
    _, mask_bin = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Extract skin pixels
    skin_pixels = img[mask_bin == 255]

    if skin_pixels.size == 0:
        continue

    # ===== MEMORY-SAFE SAMPLING ==========
    skin_pixels = skin_pixels[::PIXEL_SAMPLE_RATE]
    # =====================================

    # Convert to HSV & YCrCb
    skin_pixels = skin_pixels.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2YCrCb)

    # Append values
    h_vals.extend(hsv[:, 0, 0])
    s_vals.extend(hsv[:, 0, 1])
    v_vals.extend(hsv[:, 0, 2])
    cr_vals.extend(ycc[:, 0, 1])
    cb_vals.extend(ycc[:, 0, 2])

    # CLEAN MEMORY
    del img, mask, mask_bin, skin_pixels, hsv, ycc
    gc.collect()


# =====================================================================
# PERCENTILES
# =====================================================================
def pct_range(values):
    if len(values) == 0:
        return 0, 0
    arr = np.array(values)
    return int(np.percentile(arr, 2)), int(np.percentile(arr, 98))


h_min, h_max = pct_range(h_vals)
s_min, s_max = pct_range(s_vals)
v_min, v_max = pct_range(v_vals)

cr_min, cr_max = pct_range(cr_vals)
cb_min, cb_max = pct_range(cb_vals)


# =====================================================================
# SAVE RESULTS
# =====================================================================
result = {
    "HSV": {
        "h_min": h_min, "h_max": h_max,
        "s_min": s_min, "s_max": s_max,
        "v_min": v_min, "v_max": v_max
    },
    "YCrCb": {
        "cr_min": cr_min, "cr_max": cr_max,
        "cb_min": cb_min, "cb_max": cb_max
    }
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=4)

print("\n🎉 TRAINING COMPLETE! (Ultra Memory Safe)")
print("➡", OUTPUT_FILE)
print(json.dumps(result, indent=4))
