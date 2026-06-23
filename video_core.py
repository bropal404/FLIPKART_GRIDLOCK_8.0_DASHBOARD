"""
video_core.py — Frame extraction & enhancement helpers for the GRIDLOCK pipeline.
All functions derived from video_processing.ipynb.
"""

import cv2
import numpy as np

# ── Config constants (used by app.py / process_video_task) ──────────────────
MIN_INTERVAL_SEC         = 1.5    # seconds between sampled frames
HARD_DROP_BLUR_THRESHOLD = 5.0    # frames below this Laplacian score are dropped (corrupted)
MAX_FRAMES               = 20     # cap on frames extracted per video


# ── Quality metrics ──────────────────────────────────────────────────────────

def laplacian_variance(frame):
    """Higher = sharper. Blurry / black frames score very low."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def edge_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def frame_metrics(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "brightness":    float(gray.mean()),
        "contrast":      float(gray.std()),
        "laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density":  edge_density(frame),
    }


# ── Rain & shadow correction ─────────────────────────────────────────────────

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
    response  = np.maximum.reduce(responses)
    threshold = max(18, np.percentile(response, 98.5))
    mask = (response >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return mask


def remove_rain_streaks(frame):
    """Classical derain: detect bright streaks and inpaint only those pixels."""
    mask  = rain_streak_mask(frame)
    ratio = float(np.count_nonzero(mask) / mask.size)
    if ratio < 0.0005:
        return frame, ratio
    derained = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
    return derained, ratio


def normalize_shadows(frame):
    """Lift locally dark regions while preserving the rest of the image."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    v   = hsv[:, :, 2]
    local_mean = cv2.GaussianBlur(v, (0, 0), sigmaX=21)
    shadow = ((v < local_mean * 0.78) & (v < 135)).astype(np.uint8) * 255
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    alpha  = cv2.GaussianBlur(shadow.astype(np.float32) / 255.0, (0, 0), sigmaX=5)
    ratio  = float(np.count_nonzero(shadow) / shadow.size)
    if ratio < 0.01:
        return frame, ratio
    gain = np.clip(local_mean / np.maximum(v, 1), 1.0, 1.75)
    hsv[:, :, 2] = np.clip(v * (1 - alpha) + (v * gain) * alpha, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), ratio


# ── Motion deblur ────────────────────────────────────────────────────────────

def motion_kernel(size=11, angle=0):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(kernel, (1, center), (size - 2, center), 1.0, 1)
    matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    return kernel / max(kernel.sum(), 1e-6)


def wiener_deconvolve_channel(channel, kernel, noise=0.015):
    channel = channel.astype(np.float32) / 255.0
    padded  = np.zeros(channel.shape, dtype=np.float32)
    kh, kw  = kernel.shape
    padded[:kh, :kw] = kernel
    padded     = np.fft.ifftshift(padded)
    image_fft  = np.fft.fft2(channel)
    kernel_fft = np.fft.fft2(padded)
    restored   = np.fft.ifft2(
        image_fft * np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise)
    )
    return np.clip(np.real(restored) * 255, 0, 255).astype(np.uint8)


def attempt_motion_deblur(frame):
    """Try simple Wiener deconvolution at a few road-CCTV motion directions."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    candidates = [frame]
    for angle in (0, 45, 90, 135):
        restored_y = wiener_deconvolve_channel(y, motion_kernel(11, angle))
        restored   = cv2.cvtColor(cv2.merge([restored_y, cr, cb]), cv2.COLOR_YCrCb2BGR)
        candidates.append(cv2.fastNlMeansDenoisingColored(restored, None, 3, 3, 7, 21))
    return max(candidates, key=laplacian_variance)


# ── Main enhancement pipeline ────────────────────────────────────────────────

def auto_enhance(frame, blur_score=None, prev_accumulated=None, is_video=True):
    """
    High-fidelity enhancement pipeline (from video_processing.ipynb).
    Uses CLAHE normalization + Laplacian sharpening.
    Optionally blends with the previous frame for temporal consistency.
    """
    # 1. Temporal consistency — blend with previous frame to reduce flicker
    if is_video and prev_accumulated is not None:
        frame = cv2.addWeighted(frame, 0.85, prev_accumulated, 0.15, 0)
    accumulated_back = frame.copy()

    # 2. Local contrast enhancement in LAB space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. High-pass Laplacian edge sharpening
    laplacian   = cv2.Laplacian(img_enhanced, cv2.CV_16S, ksize=3)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    processed_frame = cv2.addWeighted(img_enhanced, 1.0, laplacian_8u, 0.4, 0)

    return processed_frame, accumulated_back, {
        "rain_mask_ratio":  0.0,
        "shadow_mask_ratio": 0.0,
        "blur_flag":        False,
        "deblur_attempted": False,
    }
