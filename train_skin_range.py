import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),

    # Lighting changes
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),

    # Blur & noise to simulate bad cameras
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),

    # Random crop + resize
    A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5),

    # Make sure masks stay binary
], is_check_shapes=True)

# === CONFIG ===
input_folder = r"c:\xampp\htdocs\image_processing\images"
output_folder = r"c:\xampp\htdocs\image_processing\model"
os.makedirs(output_folder, exist_ok=True)

# Initial (broad) HSV range for skin
lower_init = np.array([0, 20, 70], dtype=np.uint8)
upper_init = np.array([25, 255, 255], dtype=np.uint8)

skin_pixels = []

print("🔍 Scanning images for skin pixels...")

# === LOOP THROUGH ALL IMAGES ===
for file in glob(os.path.join(input_folder, "*.jpg")) + \
            glob(os.path.join(input_folder, "*.jpeg")) + \
            glob(os.path.join(input_folder, "*.png")):
    img = cv2.imread(file)
    if img is None:
        print(f"⚠️ Cannot read {file}")
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_init, upper_init)

    # Optional: clean mask a bit
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract skin pixels
    pixels = hsv[mask > 0]
    if len(pixels) > 0:
        skin_pixels.append(pixels)
        print(f"✅ Collected {len(pixels)} pixels from {os.path.basename(file)}")
    else:
        print(f"⚠️ No skin pixels detected in {os.path.basename(file)}")

if not skin_pixels:
    print("❌ No skin pixels found in any image. Try adjusting your initial range.")
    exit()

# Combine all pixels
skin_pixels = np.vstack(skin_pixels)

# === COMPUTE TRAINED RANGE (using percentiles to ignore outliers) ===
lower_skin = np.percentile(skin_pixels, 2, axis=0)
upper_skin = np.percentile(skin_pixels, 98, axis=0)

lower_skin = np.clip(lower_skin, 0, 255).astype(np.uint8)
upper_skin = np.clip(upper_skin, 0, 255).astype(np.uint8)

plt.imshow(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
plt.title("Last Processed Sample")
plt.show()

# === SAVE TRAINED MODEL ===
np.save(os.path.join(output_folder, "trained_lower_skin.npy"), lower_skin)
np.save(os.path.join(output_folder, "trained_upper_skin.npy"), upper_skin)

print("\n🎯 Training complete!")
print(f"Lower HSV: {lower_skin}")
print(f"Upper HSV: {upper_skin}")
print(f"💾 Saved to: {output_folder}")
