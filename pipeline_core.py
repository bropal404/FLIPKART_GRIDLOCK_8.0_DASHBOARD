#!/usr/bin/env python
# coding: utf-8

# # Traffic Violation Challan Pipeline
# 
# Photos in `input/` → trained YOLO → (optional Florence-2 VLM fallback) → ANPR OCR → challans in `outputs/challan_UID_TYPE/`.
# 
# **Architecture:**
# - YOLO is the primary violation detector (no_helmet, bad_helmet, no_seatbelt, triple_riding).
# - Florence-2 is a **fallback only** — invoked when YOLO finds a rider/vehicle context but no direct violation class.
# - VLM never overrides or suppresses YOLO's direct detections.
# - `norm_class_name` includes `motorcyclist → no_helmet` (nckh-2023 dataset key fix).
# 

# In[1]:


# get_ipython().run_line_magic('pip', '-q install -U "ultralytics>=8.3.0" pillow pandas opencv-python')
# Florence-2 is optional — install only if you want VLM fallback:
# %pip -q install "transformers==4.49.0"


# In[2]:


import json, os, re, shutil, time, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
import ultralytics
from ultralytics import YOLO

# Florence-2 is optional — imported only if available
try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    print("[INFO] transformers not installed — Florence-2 VLM disabled (YOLO-only mode).")

print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics: {ultralytics.__version__}")


# In[3]:


# =============================
# CONFIG - local + Kaggle paths
# =============================
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
RUNNING_ON_KAGGLE = Path('/kaggle/input').exists()

INPUT_FOLDER = Path(os.getenv('PIPELINE_INPUT_FOLDER', ROOT / 'input'))
OUTPUT_FOLDER = Path(os.getenv('PIPELINE_OUTPUT_FOLDER', ROOT / 'outputs'))
MODELS_FOLDER = Path(os.getenv('PIPELINE_MODELS_FOLDER', ROOT / 'models'))

# Local defaults:
#   models/traffic_yolo_best.pt  -> your trained helmet/seatbelt/triple-riding YOLO
#   models/anpr_yolo_best.pt     -> your ANPR character YOLO from anpr_complete notebook
# Kaggle defaults are used only when running on Kaggle.
LOCAL_TRAFFIC_WEIGHTS = MODELS_FOLDER / 'traffic_yolo_best.pt'
LOCAL_ANPR_WEIGHTS = MODELS_FOLDER / 'anpr_yolo_best.pt'
KAGGLE_TRAFFIC_WEIGHTS = Path('/kaggle/input/traffic-violation-yolo/best.pt')
KAGGLE_ANPR_WEIGHTS = Path('/kaggle/input/anpr-yolo-character-model/best.pt')

TRAFFIC_YOLO_WEIGHTS = Path(os.getenv(
    'TRAFFIC_YOLO_WEIGHTS',
    str(KAGGLE_TRAFFIC_WEIGHTS if RUNNING_ON_KAGGLE else LOCAL_TRAFFIC_WEIGHTS),
))
ANPR_YOLO_WEIGHTS = Path(os.getenv(
    'ANPR_YOLO_WEIGHTS',
    str(KAGGLE_ANPR_WEIGHTS if RUNNING_ON_KAGGLE else LOCAL_ANPR_WEIGHTS),
))
VLM_MODEL_ID = os.getenv('VLM_MODEL_ID', 'microsoft/Florence-2-large')

# Confidence threshold below which a no_helmet detection is sent to VLM
# for verification (even when YOLO already tagged it directly).
NO_HELMET_CONF_THRESHOLD = 0.50  # tune: detections below this → VLM review

YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640

ANPR_IMGSZ = 640
ANPR_IOU = 0.45
# Sweep fewer thresholds and exit early once a valid plate is found — avoids CPU hang
ANPR_CONF_CANDIDATES = [0.35, 0.25, 0.20, 0.15, 0.10]
MAX_PLATE_CHARS = 11
ROW_THRESHOLD = 0.12

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

FINE_MAP = {
    'Riding without helmet': 1000,
    'Improperly worn helmet': 500,
    'Triple Riding': 2000,
    'Not wearing seatbelt': 1000,
}

for folder in [INPUT_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

print(f'ROOT={ROOT.resolve()}')
print(f'RUNNING_ON_KAGGLE={RUNNING_ON_KAGGLE}')
print(f'INPUT_FOLDER={INPUT_FOLDER.resolve()}')
print(f'OUTPUT_FOLDER={OUTPUT_FOLDER.resolve()}')
print(f'MODELS_FOLDER={MODELS_FOLDER.resolve()}')
print(f'TRAFFIC_YOLO_WEIGHTS={TRAFFIC_YOLO_WEIGHTS}')
print(f'ANPR_YOLO_WEIGHTS={ANPR_YOLO_WEIGHTS}')
print(f'VLM={VLM_MODEL_ID} on {DEVICE}')


# In[4]:


# ── Load all models ────────────────────────────────────────────────────────

def require_file(path, label):
    if path.exists():
        return path
    alt = Path('/kaggle/input') / path.name
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"{label} weights not found at {path} (or {alt}).\n"
        f"Upload them to Kaggle as a dataset or place them in {MODELS_FOLDER}."
    )

def load_yolo(weights_path, label):
    try:
        m = YOLO(str(weights_path))
        return m
    except Exception as e:
        raise RuntimeError(
            f"{label}: load failed. "
            f"Ultralytics {ultralytics.__version__} — try: pip install -U 'ultralytics>=8.3.0'\n{e}"
        ) from e

TRAFFIC_YOLO_WEIGHTS = require_file(TRAFFIC_YOLO_WEIGHTS, "Traffic YOLO")
ANPR_YOLO_WEIGHTS    = require_file(ANPR_YOLO_WEIGHTS,    "ANPR YOLO")

print("Loading traffic YOLO...")
traffic_model = load_yolo(TRAFFIC_YOLO_WEIGHTS, "traffic YOLO")
print(f"  classes: {traffic_model.names}")

print("Loading ANPR YOLO...")
anpr_model = load_yolo(ANPR_YOLO_WEIGHTS, "ANPR YOLO")
print(f"  classes: {anpr_model.names}")

# Florence-2 — optional fallback; load only if transformers is installed
VLM_AVAILABLE = False
vlm_processor = None
vlm_model = None
if _TRANSFORMERS_AVAILABLE:
    try:
        print(f"Loading Florence-2 ({VLM_MODEL_ID})...")
        vlm_processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
        vlm_model = (
            AutoModelForCausalLM
            .from_pretrained(VLM_MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True, attn_implementation="eager")
            .to(DEVICE)
        )
        vlm_model.eval()
        VLM_AVAILABLE = True
        print("Florence-2 loaded — VLM fallback active.")
    except Exception as e:
        print(f"[WARN] Florence-2 load failed ({e}) — running YOLO-only.")
else:
    print("[INFO] VLM disabled (transformers not installed).")

print("All YOLO models loaded.")


# In[5]:


def norm_class_name(name):
    """
    Normalise a raw YOLO class name to MASTER_CLASSES vocabulary.
    Handles camelCase, hyphens, spaces, and all known synonyms including
    the critical 'motorcyclist' → 'no_helmet' mapping (nckh-2023 dataset).
    """
    s = str(name).strip()
    # camelCase split: 'FaceWithNoHelmet' → 'Face_With_No_Helmet'
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    s = s.lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    aliases = {
        # ── KEY FIX: nckh-2023 dataset uses 'motorcyclist' for riders WITHOUT helmets ──
        "motorcyclist":               "no_helmet",
        # no helmet synonyms
        "nohelmet":                   "no_helmet",
        "without_helmet":             "no_helmet",
        "withouthelmet":              "no_helmet",
        "facewithnohelmet":           "no_helmet",
        "head_without_helmet":        "no_helmet",
        "not_wearing_helmet":         "no_helmet",
        "helmet_violation":           "no_helmet",
        # bad helmet
        "badhelmet":                  "bad_helmet",
        "facewithbadhelmet":          "bad_helmet",
        "incorrect_helmet":           "bad_helmet",
        # helmet compliant
        "with_helmet":                "helmet",
        "good_helmet":                "helmet",
        "facewithgoodhelmet":         "helmet",
        "head_with_helmet":           "helmet",
        "wearing_helmet":             "helmet",
        "yes_helmet":                 "helmet",
        # rider context
        "biker":                      "rider",
        "bike_rider":                 "rider",
        "person_on_bike":             "rider",
        # vehicles
        "motorcycle":                 "vehicle",
        "motorbike":                  "vehicle",
        "bike":                       "vehicle",
        "scooter":                    "vehicle",
        "two_wheeler":                "vehicle",
        "car":                        "vehicle",
        "truck":                      "vehicle",
        "bus":                        "vehicle",
        # plates
        "number_plate":               "license_plate",
        "numberplate":                "license_plate",
        "licence_plate":              "license_plate",
        "plate":                      "license_plate",
        # seatbelt compliant
        "seat_belt":                  "seatbelt",
        "with_seatbelt":              "seatbelt",
        "wearing_seatbelt":           "seatbelt",
        "person_seatbelt":            "seatbelt",
        # no seatbelt — violation
        "no_seat_belt":               "no_seatbelt",
        "without_seatbelt":           "no_seatbelt",
        "not_wearing_seatbelt":       "no_seatbelt",
        "person_noseatbelt":          "no_seatbelt",
        "2":                          "no_seatbelt",  # confirmed numeric from seatbelt yaml
        # triple riding
        "triple_ride":                "triple_riding",
        "tripleriding":               "triple_riding",
        "three_person":               "triple_riding",
        "triple":                     "triple_riding",
        "3person":                    "triple_riding",
        "more_than_2_person_on_2_wheeler": "triple_riding",
    }
    return aliases.get(s, s)

LABEL_COLORS = {
    "license_plate":  (255, 215,   0),
    "no_helmet":      (255,   0,   0),
    "helmet":         (  0, 180,   0),
    "bad_helmet":     (255, 140,   0),
    "rider":          (  0, 150, 255),
    "seatbelt":       (  0, 180,   0),
    "no_seatbelt":    (255,  60,  60),
    "triple_riding":  (200,   0, 200),
    "vehicle":        ( 80, 180, 255),
}
YOLO_DIRECT_VIOLATION_MAP = {
    "no_helmet":     "Riding without helmet",
    "bad_helmet":    "Improperly worn helmet",
    "no_seatbelt":   "Not wearing seatbelt",
    "triple_riding": "Triple Riding",
}

def get_font(size=18):
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _box_contains(outer, inner, min_overlap=0.40):
    """
    Return True if *inner* overlaps *outer* by at least min_overlap of inner's area.
    Used to detect when a helmet box sits inside a no_helmet box.
    """
    xA = max(outer[0], inner[0]); yA = max(outer[1], inner[1])
    xB = min(outer[2], inner[2]); yB = min(outer[3], inner[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    inner_area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (inter / inner_area) >= min_overlap


def filter_genuine_no_helmet(detections):
    """
    Keep only no_helmet detections that do NOT have a co-located helmet box.

    A no_helmet detection is suppressed when any helmet box overlaps it by
    >= 40 % of the helmet box area — meaning the same head region was also
    labelled with a helmet, indicating a model inconsistency rather than a
    real violation.

    All other class detections are returned unchanged.
    """
    helmet_boxes = [d["box"] for d in detections if d["class_name"] == "helmet"]
    filtered = []
    for d in detections:
        if d["class_name"] == "no_helmet":
            suppressed = any(_box_contains(d["box"], hb) for hb in helmet_boxes)
            if suppressed:
                print(f"  [filter] Suppressed no_helmet @ {d['box']} "
                      f"(overlaps a co-located helmet detection)")
                continue
        filtered.append(d)
    return filtered


def crop_and_save_no_helmets(image, detections, challan_dir, pad_frac=0.10):
    """
    Crop each confirmed no_helmet region from *image* and save to challan_dir.

    Returns a list of dicts with keys:
        path        – saved file path (str)
        detection   – the detection dict
        crop_pil    – PIL.Image of the crop (for immediate VLM use)
    """
    crops = []
    no_helmet_dets = [d for d in detections if d["class_name"] == "no_helmet"]
    for idx, d in enumerate(no_helmet_dets):
        x1, y1, x2, y2 = d["box"]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        px, py = int(w * pad_frac), int(h * pad_frac)
        crop = image.crop((
            max(0, x1 - px), max(0, y1 - py),
            min(image.width, x2 + px), min(image.height, y2 + py),
        ))
        crop_path = challan_dir / f"no_helmet_crop_{idx:02d}_conf{d['confidence']:.2f}.jpg"
        crop.save(crop_path, quality=95)
        crops.append({"path": str(crop_path), "detection": d, "crop_pil": crop})
    return crops


def run_traffic_yolo(image_path):
    result = traffic_model.predict(
        str(image_path), conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, verbose=False
    )[0]
    image = Image.open(image_path).convert("RGB")
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = get_font(18)
    detections, yolo_violations = [], []

    boxes = result.boxes if result.boxes is not None else []
    for box in boxes:
        cls_id = int(box.cls[0])
        raw_name = result.names.get(cls_id, str(cls_id))
        class_name = norm_class_name(raw_name)
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        color = LABEL_COLORS.get(class_name, (220, 220, 220))
        label = f"{class_name} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        draw.rectangle([x1, max(0, y1 - th - 6), x1 + tw + 8, y1], fill=(0, 0, 0))
        draw.text((x1 + 4, max(0, y1 - th - 4)), label, fill=color, font=font)
        detections.append({
            "class_id": cls_id,
            "raw_class_name": str(raw_name),
            "class_name": class_name,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
        })
        if class_name in YOLO_DIRECT_VIOLATION_MAP:
            yolo_violations.append(YOLO_DIRECT_VIOLATION_MAP[class_name])

    # ── NEW: remove no_helmet detections that co-localise with a helmet ──
    detections = filter_genuine_no_helmet(detections)
    # Re-derive violations from the filtered detection set
    yolo_violations = sorted(set(
        YOLO_DIRECT_VIOLATION_MAP[d["class_name"]]
        for d in detections
        if d["class_name"] in YOLO_DIRECT_VIOLATION_MAP
    ))
    return annotated, detections, yolo_violations


# In[6]:


# ── VLM helpers (Florence-2 fallback) ──────────────────────────────────────

def _crop_to_context(image, detections, pad_frac=0.15):
    """
    Crop the PIL image to the union of rider/vehicle/violation bounding boxes.
    Gives the VLM a tighter, more informative region instead of the full frame.
    Returns the full image if no context boxes exist.
    """
    context_classes = {"rider", "vehicle", "no_helmet", "bad_helmet", "triple_riding", "no_seatbelt"}
    boxes = [d["box"] for d in detections if d["class_name"] in context_classes]
    if not boxes:
        return image
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    return image.crop((
        max(0, x1 - pw), max(0, y1 - ph),
        min(image.width, x2 + pw), min(image.height, y2 + ph),
    ))

def run_vlm_caption(image):
    """Run Florence-2 DETAILED_CAPTION on a (pre-cropped) PIL image."""
    if not VLM_AVAILABLE:
        return ""
    task = "<DETAILED_CAPTION>"
    inputs = vlm_processor(text=task, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.no_grad():
        generated_ids = vlm_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=3, do_sample=False,
        )
    generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = vlm_processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    return parsed.get(task, "").strip()

def matches_any(text, patterns):
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)

HELMET_NOT_WORN = [
    r"no helmet", r"without (?:a )?helmets?", r"not wearing (?:a )?helmets?",
    r"bare head", r"bareheaded", r"exposed head", r"none.*(?:wearing|have).*helmets?",
    r"without safety helmets?", r"no safety helmets?", r"nobody.*wearing.*helmet",
    r"without their helmets?"
]
NO_SEATBELT = [
    r"no seat ?belts?", r"without (?:a )?seat ?belts?",
    r"not wearing (?:a )?seat ?belts?", r"unfastened seat ?belts?",
]
TRIPLE_RIDING = [
    r"three (?:people|persons|riders|passengers|men|women|children|guys|girls|boys|individuals)",
    r"more than two (?:people|persons|riders|passengers|men|women|children|guys|girls|boys)",
    r"triple riding",
    r"three on (?:a )?(?:bike|motorcycle|scooter)",
    r"3 (?:people|persons|riders|passengers|men|women|children|guys|girls|boys|individuals)",
    r"a group of three",
    r"three friends",
    r"three young"
]

# YOLO violation classes that YOLO directly predicts — these are never overridden by VLM
_YOLO_HELMET_VIOLATIONS  = {"Riding without helmet", "Improperly worn helmet"}
_YOLO_SEATBELT_VIOLATION = {"Not wearing seatbelt"}
_YOLO_TRIPLE_VIOLATION   = {"Triple Riding"}

def get_vlm_verdict(image_path, detections, yolo_violations):
    """
    Determine final violation verdict.

    Decision hierarchy:
      1. YOLO direct detections are authoritative — never suppressed by VLM.
      2. VLM is invoked ONLY as a fallback when YOLO found a rider/vehicle
         context class but NO corresponding direct violation class.  This
         catches cases where the model outputs 'rider' without 'no_helmet'.
      3. For triple_riding VLM is always allowed to supplement (harder to
         detect from a single box).
      4. If VLM is not available, YOLO-only verdict is returned directly.
    """
    violations = set(yolo_violations)   # locked — YOLO is ground truth
    evidence   = [f"YOLO detected: {v}" for v in yolo_violations]
    caption    = ""

    detected_classes = {d["class_name"] for d in detections}
    has_rider_context   = bool(detected_classes & {"rider", "no_helmet", "bad_helmet"})
    has_vehicle_context = bool(detected_classes & {"vehicle", "no_seatbelt"})

    # Decide whether VLM is worth running at all
    yolo_covers_helmet   = bool(violations & _YOLO_HELMET_VIOLATIONS)
    yolo_covers_seatbelt = bool(violations & _YOLO_SEATBELT_VIOLATION)
    yolo_covers_triple   = bool(violations & _YOLO_TRIPLE_VIOLATION)

    # no_helmet detections that survived the helmet-overlap filter but have
    # low YOLO confidence still need VLM confirmation.
    low_conf_no_helmets = [
        d for d in detections
        if d["class_name"] == "no_helmet"
        and d["confidence"] < NO_HELMET_CONF_THRESHOLD
    ]
    has_low_conf_no_helmet = bool(low_conf_no_helmets)

    need_vlm = VLM_AVAILABLE and (
        (has_rider_context   and not yolo_covers_helmet)    # rider seen but no helmet class
        or (has_vehicle_context and not yolo_covers_seatbelt) # vehicle seen but no seatbelt class
        or (not yolo_covers_triple)                           # always check triple via VLM
        or has_low_conf_no_helmet                             # low-confidence no_helmet → verify
    )

    if need_vlm:
        image = Image.open(image_path).convert("RGB")
        crop  = _crop_to_context(image, detections)
        caption = run_vlm_caption(crop)
        evidence.append(f"VLM caption (on {'crop' if crop.size != image.size else 'full frame'}): {caption[:120]}")

        # VLM fallback for helmet — only when YOLO missed it entirely
        if not yolo_covers_helmet and matches_any(caption, HELMET_NOT_WORN):
            violations.add("Riding without helmet")
            evidence.append("VLM fallback: no-helmet pattern in caption.")

        # VLM verification for low-confidence no_helmet detections
        if has_low_conf_no_helmet:
            min_conf = min(d["confidence"] for d in low_conf_no_helmets)
            if matches_any(caption, HELMET_NOT_WORN):
                violations.add("Riding without helmet")
                evidence.append(
                    f"VLM confirmed low-confidence no_helmet "
                    f"(min conf={min_conf:.2f}): no-helmet pattern in caption."
                )
            else:
                # VLM disagrees — remove violation if YOLO was the sole source
                if not yolo_covers_helmet:
                    violations.discard("Riding without helmet")
                evidence.append(
                    f"VLM did NOT confirm low-confidence no_helmet "
                    f"(min conf={min_conf:.2f}): violation removed."
                )

        # VLM fallback for seatbelt — only when YOLO missed it
        if not yolo_covers_seatbelt and has_vehicle_context and matches_any(caption, NO_SEATBELT):
            violations.add("Not wearing seatbelt")
            evidence.append("VLM fallback: no-seatbelt pattern in caption.")

        # Triple riding — VLM always supplements (hard to catch in single box)
        if not yolo_covers_triple and matches_any(caption, TRIPLE_RIDING):
            violations.add("Triple Riding")
            evidence.append("VLM fallback: triple-riding pattern in caption.")

    violations = sorted(violations)
    return {
        "verdict":          "VIOLATION" if violations else "CLEAN",
        "caption":          caption,
        "violations":       violations,
        "evidence":         evidence,
        "fine_total":       sum(FINE_MAP.get(v, 0) for v in violations),
        "detected_classes": sorted(detected_classes),
        "vlm_used":         need_vlm,
    }


# In[7]:


CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
INDIAN_STATE_CODES = {"AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TS", "TR", "UK", "UP", "WB"}
LETTER_FIX = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
STATE_FIX = {"BL": "DL", "JL": "KL", "M0": "MH", "T0": "TN"}

def normalize_plate_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())

def cleanup_plate_text(text):
    chars = list(normalize_plate_text(text))
    if len(chars) < 4:
        return "".join(chars)
    if len(chars) >= 2:
        prefix = "".join(chars[:2])
        if prefix in STATE_FIX:
            chars[0], chars[1] = STATE_FIX[prefix][0], STATE_FIX[prefix][1]
        elif prefix not in INDIAN_STATE_CODES:
            for idx, ch in enumerate(chars[:2]):
                if ch in LETTER_FIX:
                    trial = chars[:2].copy(); trial[idx] = LETTER_FIX[ch]
                    if "".join(trial) in INDIAN_STATE_CODES:
                        chars[0], chars[1] = trial[0], trial[1]; break
    for idx in [2, 3]:
        if idx < len(chars) and chars[idx] in DIGIT_FIX:
            chars[idx] = DIGIT_FIX[chars[idx]]
    for idx in range(max(0, len(chars) - 4), len(chars)):
        if chars[idx] in DIGIT_FIX:
            chars[idx] = DIGIT_FIX[chars[idx]]
    cleaned = "".join(chars)
    if len(cleaned) == 11 and cleaned[-1].isalpha() and cleaned[-5:-1].isdigit():
        cleaned = cleaned[:-1]
    return cleaned

def group_rows(chars):
    chars = sorted(chars, key=lambda c: c["y"])
    rows = []
    for ch in chars:
        placed = False
        for row in rows:
            if abs(ch["y"] - float(np.mean([c["y"] for c in row]))) <= ROW_THRESHOLD:
                row.append(ch); placed = True; break
        if not placed:
            rows.append([ch])
    return rows

def reconstruct_plate_text(chars):
    row_texts = []
    for row in group_rows(chars):
        row_texts.append("".join(c["char"] for c in sorted(row, key=lambda c: c["x"]) if c["char"]))
    return cleanup_plate_text("".join(row_texts))

def crop_plate(image, box, pad_frac=0.18):
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(w * pad_frac), int(h * pad_frac)
    return image.crop((max(0, x1 - px), max(0, y1 - py), min(image.width, x2 + px), min(image.height, y2 + py)))

def predict_chars_at_conf(crop_path, conf):
    start = time.perf_counter()
    result = anpr_model.predict(str(crop_path), imgsz=ANPR_IMGSZ, conf=conf, iou=ANPR_IOU, verbose=False, max_det=MAX_PLATE_CHARS, agnostic_nms=True)[0]
    chars = []
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        h, w = result.orig_shape
        for box, cls_id, score in zip(xyxy, cls_ids, confs):
            x1, y1, x2, y2 = box.tolist()
            model_name = anpr_model.names.get(int(cls_id), "") if hasattr(anpr_model, "names") else ""
            char = str(model_name).strip().upper()
            if len(char) != 1 or not char.isalnum():
                char = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else ""
            chars.append({"char": char, "conf": float(score), "x": ((x1 + x2) / 2) / w, "y": ((y1 + y2) / 2) / h, "bbox": [int(x1), int(y1), int(x2), int(y2)]})
    return {"plate_text": reconstruct_plate_text(chars), "ocr_confidence": float(np.mean([c["conf"] for c in chars])) if chars else np.nan, "char_count": len(chars), "latency_ms": round((time.perf_counter() - start) * 1000, 2), "characters": chars, "conf_threshold": conf}

def score_candidate(c):
    count = c["char_count"]
    avg = c["ocr_confidence"] if not np.isnan(c["ocr_confidence"]) else 0.0
    text = c.get("plate_text", "")
    length_score = 2.0 - abs(count - 9) * 0.12 if 6 <= count <= 10 else -abs(count - 9) * 0.35
    format_score = 0.0
    if len(text) >= 2 and text[:2] in INDIAN_STATE_CODES: format_score += 0.55
    if len(text) >= 4 and text[2:4].isdigit(): format_score += 0.35
    elif len(text) >= 3 and text[2].isdigit(): format_score += 0.15
    if len(text) >= 4 and text[-4:].isdigit(): format_score += 0.45
    return length_score + format_score + avg + c["conf_threshold"] * 0.5

def read_plate_crop(crop_path):
    """
    Sweep confidence thresholds, scoring each candidate.
    Early-exits as soon as a result with a valid Indian state-code prefix
    and reasonable char count is found — avoids unnecessary CPU inference.
    """
    best = None
    best_score = float('-inf')
    for conf in ANPR_CONF_CANDIDATES:
        c = predict_chars_at_conf(crop_path, conf)
        s = score_candidate(c)
        if s > best_score:
            best_score = s
            best = c
        # Early exit: valid state prefix + at least 6 chars → good enough
        text = c.get('plate_text', '')
        if (len(text) >= 6
                and text[:2] in INDIAN_STATE_CODES
                and s > 1.5):
            break
    best["threshold_candidates"] = [{"conf_threshold": best["conf_threshold"],
                                      "plate_text": best["plate_text"],
                                      "char_count": best["char_count"],
                                      "ocr_confidence": best["ocr_confidence"]}]
    return best

def annotate_plate_crop(crop_path, ocr_result, out_path):
    image = cv2.imread(str(crop_path))
    if image is None:
        raise ValueError(f"Could not read plate crop: {crop_path}")
    for ch in ocr_result.get("characters", []):
        x1, y1, x2, y2 = ch["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, ch["char"], (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    label = f"ANPR: {ocr_result.get('plate_text') or 'FAILED'}"
    cv2.rectangle(image, (6, 6), (min(image.shape[1] - 1, 18 + 12 * len(label)), 40), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)

def extract_plate_number(image_path, detections, challan_dir):
    image = Image.open(image_path).convert("RGB")
    plate_dets = [d for d in detections if d["class_name"] == "license_plate"]
    if not plate_dets:
        return {"plate_text": "[PLATE NOT DETECTED]", "ocr_confidence": np.nan, "char_count": 0, "plate_crop": None, "plate_annotated": None, "status": "plate_box_missing"}
    best_plate = max(plate_dets, key=lambda d: d["confidence"])
    crop = crop_plate(image, best_plate["box"])
    crop_path = challan_dir / "plate_crop.jpg"
    crop.save(crop_path, quality=95)
    ocr_result = read_plate_crop(crop_path)
    annotated_plate = challan_dir / "plate_ocr.jpg"
    annotate_plate_crop(crop_path, ocr_result, annotated_plate)
    ocr_result.update({"plate_crop": str(crop_path), "plate_annotated": str(annotated_plate), "status": "ok" if ocr_result.get("plate_text") else "ocr_failed"})
    if not ocr_result.get("plate_text"):
        ocr_result["plate_text"] = "[OCR FAILED]"
    return ocr_result


# In[8]:


def detections_table(detections):
    if not detections:
        return "| Class | Confidence | Bounding Box |\n|---|---:|---|\n| None | - | - |\n"
    rows = ["| Class | Confidence | Bounding Box |", "|---|---:|---|"]
    for d in detections:
        rows.append(f"| {d['class_name']} | {d['confidence']:.3f} | {d['box']} |")
    return "\n".join(rows) + "\n"

def write_challan(image_path, annotated_img, detections, vlm_result,
                  location=None, date=None, time_str=None):
    uid = uuid.uuid4().hex[:8].upper()
    challan_dir = OUTPUT_FOLDER / f"challan_{uid}_{vlm_result['verdict']}"
    challan_dir.mkdir(parents=True, exist_ok=False)

    original_dest = challan_dir / f"original{Path(image_path).suffix.lower()}"
    marked_dest = challan_dir / "yolo_marked.jpg"
    shutil.copy2(image_path, original_dest)
    annotated_img.save(marked_dest, quality=95)

    # ── Crop and save each confirmed no_helmet region ──────────────────
    orig_image = Image.open(image_path).convert("RGB")
    no_helmet_crops = crop_and_save_no_helmets(orig_image, detections, challan_dir)
    if no_helmet_crops:
        print(f"  Saved {len(no_helmet_crops)} no_helmet crop(s) to {challan_dir}")

    ocr_result = extract_plate_number(image_path, detections, challan_dir)
    plate_text = ocr_result.get("plate_text", "[OCR FAILED]")
    violations = vlm_result.get("violations", [])
    violations_md = "\n".join(f"- {v}" for v in violations) if violations else "_None detected_"
    evidence = vlm_result.get("evidence", [])
    evidence_md = "\n".join(f"- {v}" for v in evidence) if evidence else "_No extra evidence text._"
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Use caller-supplied metadata when available, otherwise derive from current time
    rec_date     = date     if date     else now.strftime("%Y-%m-%d")
    rec_time     = time_str if time_str else now.strftime("%H:%M")
    rec_location = location if location else "Hyderabad (default,no geo tag in image)"

    no_helmet_crops_md = (
        "\n".join(
            f"- ![no_helmet_crop_{i}]({Path(c['path']).name}) "
            f"conf={c['detection']['confidence']:.2f}"
            for i, c in enumerate(no_helmet_crops)
        ) or "_No confirmed no-helmet crops._"
    )

    md_text = f"""# Traffic Violation Challan

| Field | Value |
|---|---|
| Challan ID | {uid} |
| Date and Time | {now_str} |
| Location | {rec_location} |
| Source Image | {Path(image_path).name} |
| Verdict | {vlm_result['verdict']} |
| Registration Number | {plate_text} |
| Total Fine | INR {vlm_result.get('fine_total', 0)} |

## Violations

{violations_md}

## VLM Description

{vlm_result.get('caption', '')}

## VLM/YOLO Evidence

{evidence_md}

## YOLO Detections

{detections_table(detections)}

## Images

| Original | YOLO Marked | Plate OCR |
|---|---|---|
| ![original]({original_dest.name}) | ![marked]({marked_dest.name}) | ![plate](plate_ocr.jpg) |

## No-Helmet Crops

{no_helmet_crops_md}
"""
    md_path = challan_dir / "challan.md"
    md_path.write_text(md_text, encoding="utf-8")

    record = {
        "challan_id": uid, "challan_dir": str(challan_dir), "source_image": str(image_path),
        "verdict": vlm_result["verdict"], "violations": violations, "fine_total": vlm_result.get("fine_total", 0),
        "registration_number": plate_text, "vlm_description": vlm_result.get("caption", ""),
        "vlm_evidence": evidence, "detections": detections, "ocr": ocr_result,
        "date": rec_date, "time": rec_time, "location": rec_location,
        "no_helmet_crops": [
            {"path": c["path"], "confidence": c["detection"]["confidence"]}
            for c in no_helmet_crops
        ],
        "files": {"original": str(original_dest), "yolo_marked": str(marked_dest), "challan_md": str(md_path), "plate_crop": ocr_result.get("plate_crop"), "plate_ocr": ocr_result.get("plate_annotated")},
    }
    def _json_default(obj):
        import math
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return str(obj)

    (challan_dir / "challan.json").write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
    )
    return record
