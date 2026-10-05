import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === CONFIG ===
BASE_DIR = r"c:\xampp\htdocs\image_processing"
IMAGE_PATH = os.path.join(BASE_DIR, "images", "sample1.jpeg")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Choose method: "HSV", "YCrCb", "AND", or "OR" ===
METHOD = "AND"   # change to "HSV" or "YCrCb" or "OR" to test other modes

# === Default ranges (will be used if no trained HSV .npy present) ===
HSV_DEFAULT_LOWER = np.array([0, 20, 70], dtype=np.uint8)
HSV_DEFAULT_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# Standard YCrCb skin range (common starting point)
YCC_DEFAULT_LOWER = np.array([0, 133, 77], dtype=np.uint8)
YCC_DEFAULT_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# === Helper: load trained HSV if exists ===
lower_hsv_path = os.path.join(MODEL_DIR, "trained_lower_skin.npy")
upper_hsv_path = os.path.join(MODEL_DIR, "trained_upper_skin.npy")
if os.path.exists(lower_hsv_path) and os.path.exists(upper_hsv_path):
    HSV_LOWER = np.load(lower_hsv_path)
    HSV_UPPER = np.load(upper_hsv_path)
    print("✅ Loaded trained HSV range from model/")
else:
    HSV_LOWER = HSV_DEFAULT_LOWER
    HSV_UPPER = HSV_DEFAULT_UPPER
    print("⚠️ Using default HSV range (no trained model found).")

# === Load image ===
if not os.path.exists(IMAGE_PATH):
    print("❌ Image not found! Check your /images/ folder path.")
    exit()

image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Failed to load image. Make sure the file is valid.")
    exit()

print(f"📸 Processing image: {IMAGE_PATH}")

# === Convert color spaces ===
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# === Create masks ===
mask_hsv = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
mask_ycc = cv2.inRange(ycrcb, YCC_DEFAULT_LOWER, YCC_DEFAULT_UPPER)

if METHOD == "HSV":
    mask = mask_hsv
elif METHOD == "YCrCb":
    mask = mask_ycc
elif METHOD == "AND":
    mask = cv2.bitwise_and(mask_hsv, mask_ycc)
elif METHOD == "OR":
    mask = cv2.bitwise_or(mask_hsv, mask_ycc)
else:
    raise ValueError("METHOD must be one of: 'HSV','YCrCb','AND','OR'")

# === Clean mask (noise reduction) ===
mask = cv2.GaussianBlur(mask, (7, 7), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# === Optionally remove small contours (improve precision) ===
min_area = 500  # tune this: remove small blobs under this area
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < min_area:
        cv2.drawContours(mask, [c], -1, 0, -1)

# === Apply mask to image ===
segmented = cv2.bitwise_and(image, image, mask=mask)

# === Measurements ===
skin_pixels = cv2.countNonZero(mask)
total_pixels = mask.size
skin_percentage = (skin_pixels / total_pixels) * 100
print(f"✅ Method: {METHOD}")
print(f"✅ Skin area detected: {skin_percentage:.2f}% of total image.")

# === Save results ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
mask_file = os.path.join(OUTPUT_DIR, f"mask_{METHOD}_{timestamp}.jpg")
seg_file = os.path.join(OUTPUT_DIR, f"segmented_{METHOD}_{timestamp}.jpg")
cv2.imwrite(mask_file, mask)
cv2.imwrite(seg_file, segmented)
print(f"💾 Saved mask -> {mask_file}")
print(f"💾 Saved segmented -> {seg_file}")

# === Display ===
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title(f"Mask ({METHOD})")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented")
plt.axis("off")

plt.tight_layout()
plt.show()
