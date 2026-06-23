#!/usr/bin/env python
# coding: utf-8

# # 🎬 Video Preprocessing — Frame Extraction & Enhancement
# 
# Extract sharp, well-exposed frames from a video at configurable intervals, auto-enhance brightness/contrast/sharpness, and save to `output1/`.

# In[ ]:


# Create a new code cell in your notebook and run this:
get_ipython().system('pip install yt-dlp')


# In[ ]:


import sys
import subprocess

print("Downloading the test video directly into input/video.mp4 using robust fallback formats...")

# Reconfigured to look for standard multi-compatible MP4 formats first
command = [
    sys.executable, "-m", "yt_dlp", 
    "-f", "mp4/best",  # Fallback to the cleanest combined MP4 available
    "https://www.youtube.com/watch?v=LVgRaVhnp9I", 
    "-o", "input/video.mp4"
]

result = subprocess.run(command, capture_output=True, text=True)

if result.returncode == 0:
    print("\n🎉 Success! The rainy traffic video has been downloaded safely to input/video.mp4")
else:
    print("\n❌ Error during download:")
    print(result.stderr)


# In[ ]:


# ─── CONFIG ────────────────────────────────────────────────────────────
VIDEO_PATH       = "input/video.mp4"     # Path to your video file
OUTPUT_DIR       = "output1"              # Where processed frames are saved
MIN_INTERVAL_SEC = 1.5                    # Minimum gap between extracted frames (seconds)
BLUR_THRESHOLD   = 80.0                   # Below this = blurry; flag + restore, do not discard
HARD_DROP_BLUR_THRESHOLD = 5.0            # Only skip frames this low (completely unreadable noise)
TARGET_BRIGHTNESS = 127                   # Target mean brightness (0-255)
MAX_FRAMES       = None                   # Set an int to cap total saved frames, or None for all
METRICS_CSV      = f"{OUTPUT_DIR}/preprocessing_metrics.csv"

import os, cv2, math, csv
import numpy as np
from pathlib import Path
from IPython.display import display, Image as IPImage

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


# In[ ]:


import os, cv2, math
import numpy as np
from pathlib import Path
from IPython.display import display, Image as IPImage

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")


# In[ ]:


# ─── HELPERS ────────────────────────────────────────────────────────────

def laplacian_variance(frame):
    """Higher = sharper. A blurry frame scores low."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def frame_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": edge_density(frame),
    }


def rain_streak_mask(frame):
    """Detect thin bright rain-like streaks using oriented top-hat filters."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    responses = []
    for angle in (65, 90, 115):
        kernel = np.zeros((15, 15), dtype=np.uint8)
        cv2.line(kernel, (7, 1), (7, 13), 1, 1)
        matrix = cv2.getRotationMatrix2D((7, 7), angle - 90, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (15, 15))
        response = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        responses.append(response)
    response = np.maximum.reduce(responses)
    threshold = max(18, np.percentile(response, 98.5))
    mask = (response >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return mask


def remove_rain_streaks(frame):
    """Classical derain: detect bright streaks and inpaint only those pixels."""
    mask = rain_streak_mask(frame)
    ratio = float(np.count_nonzero(mask) / mask.size)
    if ratio < 0.0005:
        return frame, ratio
    derained = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return derained, ratio


def normalize_shadows(frame):
    """Lift locally dark regions while preserving the rest of the image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    local_mean = cv2.GaussianBlur(v, (0, 0), sigmaX=21)
    shadow = ((v < local_mean * 0.78) & (v < 135)).astype(np.uint8) * 255
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    alpha = cv2.GaussianBlur(shadow.astype(np.float32) / 255.0, (0, 0), sigmaX=5)
    ratio = float(np.count_nonzero(shadow) / shadow.size)
    if ratio < 0.01:
        return frame, ratio
    gain = np.clip(local_mean / np.maximum(v, 1), 1.0, 1.75)
    hsv[:, :, 2] = np.clip(v * (1 - alpha) + (v * gain) * alpha, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), ratio


def motion_kernel(size=11, angle=0):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(kernel, (1, center), (size - 2, center), 1.0, 1)
    matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    return kernel / max(kernel.sum(), 1e-6)


def wiener_deconvolve_channel(channel, kernel, noise=0.015):
    channel = channel.astype(np.float32) / 255.0
    padded = np.zeros(channel.shape, dtype=np.float32)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.fft.ifftshift(padded)
    image_fft = np.fft.fft2(channel)
    kernel_fft = np.fft.fft2(padded)
    restored = np.fft.ifft2(image_fft * np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise))
    return np.clip(np.real(restored) * 255, 0, 255).astype(np.uint8)


def attempt_motion_deblur(frame):
    """Try simple Wiener deconvolution at a few road-CCTV motion directions."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    candidates = [frame]
    for angle in (0, 45, 90, 135):
        restored_y = wiener_deconvolve_channel(y, motion_kernel(11, angle))
        restored = cv2.cvtColor(cv2.merge([restored_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
        candidates.append(cv2.fastNlMeansDenoisingColored(restored, None, 3, 3, 7, 21))
    return max(candidates, key=laplacian_variance)


def auto_enhance(frame, blur_score=None, prev_accumulated=None, is_video=True):
    """
    High-fidelity enhancement pipeline.
    Uses target CLAHE normalization and a Laplacian kernel pass to ensure 
    sharp structural edges without any blurring artifacts.
    """
    # 1. Temporal Video Consistency
    if is_video and prev_accumulated is not None:
        frame = cv2.addWeighted(frame, 0.85, prev_accumulated, 0.15, 0)
    accumulated_back = frame.copy()

    # 2. Balanced Local Contrast Enhancement (LAB Space)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # ClipLimit=1.5 brings out details safely without introducing noise halos
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. High-Pass Laplacian Edge Sharpening
    # This directly targets sharp boundaries (lines, wires, text) and avoids blurring tree layers
    laplacian = cv2.Laplacian(img_enhanced, cv2.CV_16S, ksize=3)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    
    # Scale and add the crisp edge details back onto the contrast-enhanced base image
    processed_frame = cv2.addWeighted(img_enhanced, 1.0, laplacian_8u, 0.4, 0)

    return processed_frame, accumulated_back, {
        "rain_mask_ratio": 0.0,
        "shadow_mask_ratio": 0.0,
        "blur_flag": False,
        "deblur_attempted": False,
    }


# In[ ]:


# ─── EXTRACT & PROCESS ──────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps
frame_gap = max(1, int(fps * MIN_INTERVAL_SEC))  # Frames to skip between candidates

print(f"Video : {VIDEO_PATH}")
print(f"FPS   : {fps:.1f}")
print(f"Frames: {total_frames}  ({duration_sec:.1f}s)")
print(f"Sampling every {frame_gap} frames (~{frame_gap/fps:.2f}s)\n")

saved, skipped_blur, idx = 0, 0, 0
preview_paths = []

# CRITICAL: Initialize the temporal frame accumulator variable before the loop begins
prev_frame = None

print(f"Beginning clean frame extraction sequence with Temporal Accumulation...")

# Open file session and keep it open across the entire extraction loop
with open(METRICS_CSV, mode='w', newline='') as csv_file:
    metrics_writer = csv.writer(csv_file)
    # Write structural tracking headers
    metrics_writer.writerow(["frame_id", "timestamp_sec", "orig_laplacian_var", "rain_mask_ratio", "shadow_mask_ratio", "deblur_triggered"])

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        score = laplacian_variance(frame)

        # FIX: Drop ONLY if the frame is completely black or corrupted noise
        if score < HARD_DROP_BLUR_THRESHOLD:
            skipped_blur += 1
            idx += frame_gap
            continue

        # Standardize dimensions uniformly so downstream detection models look uniform
        frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

        # Run the new temporal enhancement pipeline, passing and updating the frame memory
        enhanced, prev_frame, meta = auto_enhance(frame_resized, blur_score=score, prev_accumulated=prev_frame)

        timestamp = idx / fps
        out_name = f"frame_{saved:04d}_t{timestamp:06.2f}s.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Write data metrics row directly inside the open file session
        metrics_writer.writerow([
            saved, 
            f"{timestamp:.2f}", 
            f"{score:.2f}", 
            f"{meta['rain_mask_ratio']:.4f}", 
            f"{meta['shadow_mask_ratio']:.4f}", 
            meta['deblur_attempted']
        ])

        saved += 1
        if saved <= 6:
            preview_paths.append(out_path)

        if saved % 10 == 0:
            print(f"  Processed and saved {saved} clean frames...")

        if MAX_FRAMES and saved >= MAX_FRAMES:
            break

        idx += frame_gap

cap.release()

print(f"\n🎉 Extraction Finished! Clear, artifact-free images saved directly to: {OUTPUT_DIR}")
print(f"  Enhanced frames saved    : {saved}")
print(f"  Corrupted frames dropped : {skipped_blur}")
print(f"  Metrics exported to     : {os.path.abspath(METRICS_CSV)}")


# In[ ]:


# ─── PREVIEW ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

if preview_paths:
    cols = min(3, len(preview_paths))
    rows = math.ceil(len(preview_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for ax, p in zip(axes, preview_paths):
        img = cv2.imread(p)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(p).name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(preview_paths):]:
        ax.axis("off")
    plt.suptitle(f"First {len(preview_paths)} extracted frames (enhanced)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("No frames extracted — check VIDEO_PATH and HARD_DROP_BLUR_THRESHOLD.")


# In[ ]:


import matplotlib.pyplot as plt

# 1. Re-open the video to grab the exact matching raw frames
cap_raw = cv2.VideoCapture(VIDEO_PATH)
raw_frames = []

# We will pull the exact frames corresponding to your first few saved samples
for i in range(3):
    idx = i * frame_gap
    cap_raw.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap_raw.read()
    if ret:
        # Standardize raw frame size to match output
        frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        raw_frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
cap_raw.release()

# 2. Load your newly enhanced post-processed images from output1/
enhanced_frames = []
for path in preview_paths[:3]:
    img = cv2.imread(path)
    if img is not None:
        enhanced_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# 3. Plot them side-by-side
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
plt.subplots_adjust(wspace=0.05, hspace=0.15)

for i in range(min(len(raw_frames), len(enhanced_frames))):
    # Left Column: Raw Inputs
    axes[i, 0].imshow(raw_frames[i])
    axes[i, 0].set_title(f"RAW INPUT (Frame {i})", fontsize=14, color='red', fontweight='bold')
    axes[i, 0].axis('off')
    
    # Right Column: Your Enhanced Pipeline
    axes[i, 1].imshow(enhanced_frames[i])
    axes[i, 1].set_title(f"ENHANCED OUTPUT (Frame {i})", fontsize=14, color='green', fontweight='bold')
    axes[i, 1].axis('off')

plt.show()


# In[ ]:


import os

# Define the absolute root where Kaggle mounts input datasets
kaggle_input_root = "/kaggle/input"

print("Scanning Kaggle Input Datasets...")
for root, dirs, files in os.walk(kaggle_input_root):
    # Filter out common deep subdirectories to keep the output clean and fast
    if "image_archive" in root:
        print(f"\n📍 EXACT MATCH FOUND:")
        print(f"Absolute Path: {os.path.abspath(root)}")
        print(f"Sample Files: {os.listdir(root)[:3]}")
        break


# In[ ]:





# In[ ]:


import os, cv2, random
import numpy as np
import matplotlib.pyplot as plt
import shutil

# 1. Paths targeting Kaggle input mount folder
IDD_LOCAL_SOURCE = "/kaggle/input/datasets/mitanshuchakrawarty/new-idd-dataset/IDD_RESIZED/image_archive"
IDD_OUT_DIR = "idd_local_enhanced"

# Wipe the old directory to ensure old ghosted frames are permanently gone
if os.path.exists(IDD_OUT_DIR):
    shutil.rmtree(IDD_OUT_DIR)
os.makedirs(IDD_OUT_DIR, exist_ok=True)

# 2. Gather all image files
all_images = [f for f in os.listdir(IDD_LOCAL_SOURCE) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(all_images)} authentic IDD images locally.")

# 3. Randomly select 30 unique images
random.seed(42)  
selected_images = random.sample(all_images, min(30, len(all_images)))

# Keep track of exactly what pairs successfully processed
processed_pairs = []

print("Processing 30 randomized local IDD files (Standalone Mode)...")

for idx, filename in enumerate(selected_images):
    raw_path = os.path.join(IDD_LOCAL_SOURCE, filename)
    out_name = f"idd_enhanced_{idx:02d}.jpg"
    out_path = os.path.join(IDD_OUT_DIR, out_name)
    
    img = cv2.imread(raw_path)
    if img is None:
        continue
        
    img_resized = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    
    # FORCE prev_accumulated=None to stop the ghosting immediately
    # If you updated auto_enhance with the flag, add: is_video=False
    enhanced, _, meta = auto_enhance(
        img_resized, 
        blur_score=100.0, 
        prev_accumulated=None
    )
    
    cv2.imwrite(out_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
    processed_pairs.append((raw_path, out_path, filename))

print(f"\n🎉 Finished! Processed {len(processed_pairs)} images cleanly.")

# ─── BULLETPROOF PLOT VISUALIZATION ───────────────────────────────────────
if len(processed_pairs) >= 3:
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    for i in range(3):
        raw_path, enhanced_path, orig_filename = processed_pairs[i]
        
        r_img = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
        e_img = cv2.cvtColor(cv2.imread(enhanced_path), cv2.COLOR_BGR2RGB)
        
        axes[i, 0].imshow(r_img)
        axes[i, 0].set_title(f"Raw IDD Asset ({orig_filename})", fontsize=11, fontweight='bold', color='darkred')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(e_img)
        axes[i, 1].set_title(f"Clean Standalone Processed Frame {i}", fontsize=11, fontweight='bold', color='darkgreen')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

