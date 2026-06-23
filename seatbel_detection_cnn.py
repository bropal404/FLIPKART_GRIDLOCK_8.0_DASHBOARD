#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


# In[ ]:


import transformers
print(transformers.__version__)


# In[ ]:


get_ipython().system('pip install -q transformers==4.49.0')


# In[ ]:


import os
import json
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

# =====================================================
# CONFIG
# =====================================================

MODEL_ID = "microsoft/Florence-2-large"

# Make sure this dataset is attached via "Add Data" in the Kaggle sidebar,
# then copy the exact path Kaggle shows you (usually /kaggle/input/<dataset-slug>/...)
INPUT_FOLDER = "/kaggle/input/datasets/meliodassourav/traffic-violation-dataset-v3"
OUTPUT_FILE = "traffic_violations.xlsx"
OUTPUT_IMAGE_DIR = "annotated_images"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

# Rule thresholds - tune these on a few sample images before trusting results
MAX_LEGAL_RIDERS_2W = 2          # >2 people on a 2-wheeler = overloading
HEAD_REGION_FRACTION = 0.35      # top % of a person's box treated as "head" (for helmet check)
TORSO_REGION = (0.20, 0.75)      # vertical band of a person's box treated as "torso" (for seatbelt check)
RIDER_VEHICLE_OVERLAP = 0.25     # min overlap fraction to count a person as "on/in" a vehicle
GEAR_OVERLAP_THRESHOLD = 0.20    # min overlap fraction to count a helmet/seatbelt as "worn"

# =====================================================
# LOAD MODEL
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True
).to(DEVICE)

model.eval()

print(f"Florence-2 loaded on {DEVICE} ({DTYPE})")


# 

# In[ ]:


# =====================================================
# DETECT VIOLATIONS (Florence-2 open-vocab grounding)
# =====================================================
# Paste this as a new cell AFTER the "LOAD MODEL" cell and BEFORE the
# "EVALUATE AGAINST A LABELED DATASET" cell. It uses the same config
# constants you already defined: MAX_LEGAL_RIDERS_2W, HEAD_REGION_FRACTION,
# TORSO_REGION, RIDER_VEHICLE_OVERLAP, GEAR_OVERLAP_THRESHOLD, DEVICE,
# DTYPE, processor, model.

import numpy as np

VEHICLE_PROMPTS = ["motorcycle or scooter", "car"]
GEAR_PROMPTS = {
    "helmet": "a protective motorcycle helmet covering a person's head, not a headlight",
    "seatbelt": "a seatbelt strap worn across a person's chest",
}
PERSON_PROMPT = "a person riding or sitting on a vehicle"

# Boxes whose IoU exceeds this are considered "the same physical object"
NMS_IOU_THRESHOLD = 0.5


def _run_grounding(image, phrase):
    """Run Florence-2 <CAPTION_TO_PHRASE_GROUNDING> for one phrase.
    Returns a list of [x1, y1, x2, y2] boxes (pixel coords)."""
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task + phrase

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )

    result = parsed.get(task, {})
    boxes = result.get("bboxes", [])
    return boxes


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _overlap_fraction(small_box, big_box):
    """Fraction of small_box's area that falls inside big_box.
    Used for 'is this person on this vehicle' / 'is this helmet on this head'."""
    small_area = _box_area(small_box)
    if small_area == 0:
        return 0.0
    return _intersection_area(small_box, big_box) / small_area


def _sub_region(box, y_frac_start, y_frac_end):
    """Return the sub-box covering a vertical band of `box`,
    e.g. head region = top HEAD_REGION_FRACTION, torso = TORSO_REGION band."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    return [x1, y1 + h * y_frac_start, x2, y1 + h * y_frac_end]


def _is_near_head(gear_box, person_box, head_region, min_overlap):
    """Stricter than a plain overlap check: the gear box must (a) overlap
    the head_region by min_overlap AND (b) its vertical center must sit in
    the top portion of the FULL person box. This rejects boxes that are
    horizontally inside the person's box (e.g. a headlight behind a rider's
    torso) but are actually positioned well below the head."""
    if _overlap_fraction(gear_box, head_region) < min_overlap and \
       _overlap_fraction(head_region, gear_box) < min_overlap:
        return False

    px1, py1, px2, py2 = person_box
    person_height = py2 - py1
    if person_height <= 0:
        return False

    gx1, gy1, gx2, gy2 = gear_box
    gear_center_y = (gy1 + gy2) / 2.0
    relative_y = (gear_center_y - py1) / person_height  # 0 = top of person, 1 = bottom

    return relative_y <= HEAD_REGION_FRACTION


def _iou(box_a, box_b):
    """Standard intersection-over-union between two boxes."""
    inter = _intersection_area(box_a, box_b)
    if inter == 0:
        return 0.0
    union = _box_area(box_a) + _box_area(box_b) - inter
    if union == 0:
        return 0.0
    return inter / union


def _nms_dedupe(boxes, iou_threshold=NMS_IOU_THRESHOLD):
    """Collapse near-duplicate boxes (same physical object detected twice)
    down to one box each, keeping the largest box in each cluster.
    `boxes` can be a flat list of [x1,y1,x2,y2] or a list of (label, box) tuples."""
    if not boxes:
        return []

    has_labels = isinstance(boxes[0], tuple)
    items = list(boxes) if has_labels else [(None, b) for b in boxes]

    # Largest-area first so the "winning" box in a cluster is the most complete one
    items = sorted(items, key=lambda lb: _box_area(lb[1]), reverse=True)

    kept = []
    for label, box in items:
        is_duplicate = any(_iou(box, kept_box) >= iou_threshold for _, kept_box in kept)
        if not is_duplicate:
            kept.append((label, box))

    return kept if has_labels else [b for _, b in kept]


def _vehicle_nms(vehicle_boxes, iou_threshold=NMS_IOU_THRESHOLD):
    """Cross-class NMS for vehicles specifically: a 'car' box and a 'motorcycle'
    box that overlap heavily are the SAME physical vehicle (Florence-2 hedging
    between labels), so only the larger/more confident one should survive."""
    return _nms_dedupe(vehicle_boxes, iou_threshold)


def _person_nms(person_boxes, iou_threshold=NMS_IOU_THRESHOLD):
    """Dedup person boxes, but unlike generic NMS, prefer the box with the
    HIGHEST top edge (smallest y1) in each overlapping cluster rather than
    the largest area. A wider box that's missing the head can have more
    area than a tighter box that actually captures the head — area is not
    a reliable signal of completeness here, top-edge position is."""
    if not person_boxes:
        return []

    # Highest (smallest y1) first, so the most "complete" box wins ties
    items = sorted(person_boxes, key=lambda b: b[1])

    kept = []
    for box in items:
        is_duplicate = any(_iou(box, kept_box) >= iou_threshold for kept_box in kept)
        if not is_duplicate:
            kept.append(box)

    return kept


def detect_violations(image_path, save_annotated=False):
    """
    Detect traffic violations in a single image using Florence-2
    open-vocabulary grounding.

    Returns a dict:
        {
            "image_path": str,
            "vehicles": [
                {
                    "type": "motorcycle" | "car",
                    "bbox": [x1, y1, x2, y2],
                    "riders": int,
                    "rider_boxes": [[x1,y1,x2,y2], ...],
                    "violations": [str, ...],
                },
                ...
            ],
            "total_violations": int,
            "annotated_image": PIL.Image (only if save_annotated=True, else None),
        }
    """
    image = Image.open(image_path).convert("RGB")

    # 1. Detect vehicles, people, and gear independently
    raw_vehicle_boxes = []  # list of (vtype, box)
    for vtype in VEHICLE_PROMPTS:
        for box in _run_grounding(image, vtype):
            raw_vehicle_boxes.append((vtype, box))

    # Cross-class NMS: collapse a 'car' box and 'motorcycle' box that are
    # really the same physical vehicle down to a single detection.
    vehicle_boxes = _vehicle_nms(raw_vehicle_boxes)

    raw_person_boxes = _run_grounding(image, PERSON_PROMPT)
    person_boxes = _person_nms(raw_person_boxes)

    raw_helmet_boxes = _run_grounding(image, GEAR_PROMPTS["helmet"])
    helmet_boxes = _nms_dedupe(raw_helmet_boxes)

    raw_seatbelt_boxes = _run_grounding(image, GEAR_PROMPTS["seatbelt"])
    seatbelt_boxes = _nms_dedupe(raw_seatbelt_boxes)

    vehicles_report = []
    total_violations = 0

    # 2. Associate people with vehicles
    for vtype, vbox in vehicle_boxes:
        is_two_wheeler = vtype == VEHICLE_PROMPTS[0]  # "motorcycle or scooter"
        is_car = vtype == VEHICLE_PROMPTS[1]           # "car"

        riders = [
            pbox for pbox in person_boxes
            if _overlap_fraction(pbox, vbox) >= RIDER_VEHICLE_OVERLAP
        ]

        violations = []

        # --- Overloading (2-wheelers only) ---
        if is_two_wheeler and len(riders) > MAX_LEGAL_RIDERS_2W:
            violations.append(
                f"Overloading: {len(riders)} riders on motorcycle "
                f"(max legal: {MAX_LEGAL_RIDERS_2W})"
            )

        # --- Helmet check (motorcycle riders) ---
        if is_two_wheeler:
            for i, pbox in enumerate(riders):
                head_region = _sub_region(pbox, 0.0, HEAD_REGION_FRACTION)
                has_helmet = any(
                    _is_near_head(hbox, pbox, head_region, GEAR_OVERLAP_THRESHOLD)
                    for hbox in helmet_boxes
                )
                if not has_helmet:
                    violations.append(f"No helmet: rider {i + 1}")

        # --- Seatbelt check (car occupants) ---
        if is_car:
            for i, pbox in enumerate(riders):
                torso_region = _sub_region(pbox, TORSO_REGION[0], TORSO_REGION[1])
                has_seatbelt = any(
                    _overlap_fraction(sbox, torso_region) >= GEAR_OVERLAP_THRESHOLD
                    or _overlap_fraction(torso_region, sbox) >= GEAR_OVERLAP_THRESHOLD
                    for sbox in seatbelt_boxes
                )
                if not has_seatbelt:
                    violations.append(f"No seatbelt: occupant {i + 1}")

        total_violations += len(violations)

        vehicles_report.append({
            "type": "motorcycle" if is_two_wheeler else "car",
            "bbox": vbox,
            "riders": len(riders),
            "rider_boxes": riders,
            "violations": violations,
        })

    report = {
        "image_path": image_path,
        "vehicles": vehicles_report,
        "total_violations": total_violations,
        "annotated_image": None,
    }

    # 3. Draw annotated image if requested
    if save_annotated:
        report["annotated_image"] = _draw_annotations(image, vehicles_report, helmet_boxes, seatbelt_boxes)

    return report


def _draw_annotations(image, vehicles_report, helmet_boxes, seatbelt_boxes):
    """Draw vehicle/person boxes + violation labels on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    COLOR_OK = (0, 200, 0)
    COLOR_VIOLATION = (255, 0, 0)
    COLOR_PERSON = (255, 165, 0)
    COLOR_GEAR = (0, 150, 255)

    for vehicle in vehicles_report:
        color = COLOR_VIOLATION if vehicle["violations"] else COLOR_OK
        draw.rectangle(vehicle["bbox"], outline=color, width=3)
        label = vehicle["type"]
        draw.text((vehicle["bbox"][0], max(0, vehicle["bbox"][1] - 22)), label, fill=color, font=font)

        for pbox in vehicle["rider_boxes"]:
            draw.rectangle(pbox, outline=COLOR_PERSON, width=2)

        # Stack violation text below the vehicle box
        text_y = vehicle["bbox"][3] + 4
        for v in vehicle["violations"]:
            draw.text((vehicle["bbox"][0], text_y), v, fill=COLOR_VIOLATION, font=font)
            text_y += 20

    for hbox in helmet_boxes:
        draw.rectangle(hbox, outline=COLOR_GEAR, width=1)
    for sbox in seatbelt_boxes:
        draw.rectangle(sbox, outline=COLOR_GEAR, width=1)

    return annotated


print("detect_violations() ready")


# In[ ]:


# =====================================================
# HELPER FUNCTIONS: crop_with_padding & _run_caption
# =====================================================

def crop_with_padding(image, box, pad_frac=0.1):
    """Crop a region from `image` defined by `box` [x1, y1, x2, y2],
    expanding each side by pad_frac * box_dimension so the crop
    includes a little context around the detected object."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(image.width,  int(x2 + pad_x))
    cy2 = min(image.height, int(y2 + pad_y))
    return image.crop((cx1, cy1, cx2, cy2))


def _run_caption(image):
    """Run Florence-2 <DETAILED_CAPTION> on `image` and return the caption string."""
    task = "<DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    return parsed.get(task, "").strip()


print("crop_with_padding() and _run_caption() ready")


# In[ ]:


# =====================================================
# BUILD eval_image_paths FROM DATASET FOLDER
# =====================================================

eval_image_paths = []
categories = []

# Walk ALL subdirectories recursively
for dirpath, dirnames, filenames in os.walk(INPUT_FOLDER):
    folder_name = os.path.basename(dirpath)
    image_files = [f for f in filenames if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    
    if not image_files:
        continue
    
    # The category is the immediate parent folder name (e.g. "no_helmet", "helmet")
    category_name = folder_name
    if category_name not in categories:
        categories.append(category_name)
    
    for fname in sorted(image_files):
        eval_image_paths.append((category_name, os.path.join(dirpath, fname)))

print(f"Found {len(categories)} categories: {categories}")
print(f"Total images: {len(eval_image_paths)}")

from collections import Counter
counts = Counter(c for c, _ in eval_image_paths)
for cat, n in sorted(counts.items()):
    print(f"  {cat}: {n} images")


# In[ ]:


# =====================================================
# CAPTION-BASED HELMET CHECK -- ALL IMAGES
# =====================================================

import re
import matplotlib.pyplot as plt
import textwrap
import math

NOT_WORN_PATTERNS = [
    r"holding (?:a |the |his |her )?(?:black |white |red |blue |yellow )?helmet",
    r"carrying (?:a |the |his |her )?helmet",
    r"helmet (?:in|on) (?:his|her|their) (?:hand|hands|lap)",
    r"helmet (?:on|hanging from|attached to|resting on) the (?:motorcycle|bike|seat|handlebar)",
    r"without (?:a |any )?helmet",
    r"not wearing (?:a |any )?helmet",
    r"no helmet",
    r"not .{0,15}helmet",
]

WORN_PATTERNS = [
    r"wearing (?:a |the )?(?:black |white |red |blue |yellow |silver |grey |gray )?helmet",
    r"helmet on (?:his|her|their) head",
    r"with (?:a |the )?helmet on",
    r"wearing [^.]*?\bhelmet\b",
    # Non-standard helmet colors
    r"(?:colorful|multicolored|decorated|pink|orange|purple|bright|red|green|brown)\s+helmet",
    # Full riding gear implies helmet worn
    r"motorcycle suit",
    r"riding gear",
    r"biker gear",
    r"wearing .{0,50}helmet",
    # Fallback: any mention of helmet not caught by NOT_WORN_PATTERNS = worn
    r"\bhelmet\b",
]


def caption_indicates_helmet_worn(caption):
    text = caption.lower()
    # NOT_WORN checked first — if any match, definitely not worn
    for pattern in NOT_WORN_PATTERNS:
        if re.search(pattern, text):
            return False
    # WORN checked second
    for pattern in WORN_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


# --- Find positive category ---
POSITIVE_CATEGORY_HINTS = ["with_helmet", "helmet", "wearing_helmet", "helmet_worn", "compliant", "clean"]
positive_category = None
for hint in POSITIVE_CATEGORY_HINTS:
    matches = [c for c in categories if hint in c.lower() and "no_" not in c.lower() and "without" not in c.lower()]
    if matches:
        positive_category = matches[0]
        break

print("Categories found in dataset:", categories)
print("Auto-detected positive (worn-helmet) category:", positive_category)

# --- Use ALL images ---
test_set = [(c, p) for c, p in eval_image_paths if c in ("no_helmet", positive_category)][:200]
print(f"\nTesting on {len(test_set)} total images")
print(f"  no_helmet: {sum(1 for c, _ in test_set if c == 'no_helmet')}")
print(f"  {positive_category}: {sum(1 for c, _ in test_set if c == positive_category)}")

# --- Run captioning + verdict ---
results = []
for category, image_path in test_set:
    image = Image.open(image_path).convert("RGB")
    person_boxes = _person_nms(_run_grounding(image, PERSON_PROMPT))

    if not person_boxes:
        results.append({
            "category": category, "image_path": image_path,
            "crop": image, "caption": "(no person box found)", "worn": None,
        })
        continue

    rider_crop = crop_with_padding(image, person_boxes[0], pad_frac=0.1)
    rider_caption = _run_caption(rider_crop)
    worn = caption_indicates_helmet_worn(rider_caption)

    results.append({
        "category": category, "image_path": image_path,
        "crop": rider_crop, "caption": rider_caption, "worn": worn,
    })

    expected_worn = (category == positive_category)
    correct = "OK" if worn == expected_worn else "WRONG"
    print(f"[{correct}] {category} / {os.path.basename(image_path)} -> worn={worn}")

# --- Summary ---
total = len([r for r in results if r["worn"] is not None])
correct_count = sum(
    1 for r in results
    if r["worn"] is not None and r["worn"] == (r["category"] == positive_category)
)
print(f"\nAccuracy: {correct_count}/{total} = {correct_count/total:.1%}")

# --- Display WRONG predictions only ---
wrong_results = [
    r for r in results
    if r["worn"] is not None and r["worn"] != (r["category"] == positive_category)
]

print(f"Wrong predictions: {len(wrong_results)} / {len(results)}")

n = len(wrong_results)
if n == 0:
    print("No wrong predictions!")
else:
    cols = 5
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 5 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(wrong_results):
        expected = (r["category"] == positive_category)
        title = (
            f"[WRONG] {r['category']}\n"
            f"predicted worn={r['worn']} | expected={expected}\n"
            + textwrap.fill(r["caption"][:90], 35)
        )
        axes[i].imshow(r["crop"])
        axes[i].set_title(title, fontsize=7, color="red")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Wrong predictions: {len(wrong_results)} / {len(results)}", fontsize=10)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()


# In[ ]:


# =====================================================
# OVERLOADING CHECK -- 30 IMAGES
# =====================================================

import re

# Lower threshold: flag if more than 1 person detected
MAX_LEGAL_RIDERS_2W_EVAL = 1

def caption_indicates_overloading(caption):
    text = caption.lower()
    overload_patterns = [
        r"two (?:people|men|women|persons|individuals|riders|passengers)",
        r"three (?:people|men|women|persons|individuals|riders|passengers)",
        r"four (?:people|men|women|persons|individuals|riders|passengers)",
        r"(?:a man|woman|person) .{0,30} (?:behind|in front of|on the back)",
        r"passenger",
        r"pillion",
        r"riding together",
        r"multiple (?:people|riders|passengers)",
        r"family",
        r"child .{0,20} (?:front|between|sitting)",
        r"sitting behind",
        r"on the back of",
    ]
    for pattern in overload_patterns:
        if re.search(pattern, text):
            return True
    return False

# --- Find overloading categories ---
overloading_categories = [c for c in categories if "overload" in c.lower() or "triple" in c.lower() or "pillion" in c.lower()]
normal_categories = [c for c in categories if "normal" in c.lower() or "single" in c.lower() or "one" in c.lower()]

print("Overloading categories found:", overloading_categories)
print("Normal categories found:", normal_categories)

# Build test set - 15 from each
overload_paths = [(c, p) for c, p in eval_image_paths if c in overloading_categories]
normal_paths = [(c, p) for c, p in eval_image_paths if c in normal_categories]

test_set_overload = overload_paths[:100] + normal_paths[:100]
print(f"\nTesting on {len(test_set_overload)} images")
print(f"  overloading: {len(overload_paths[:15])}")
print(f"  normal: {len(normal_paths[:15])}")

# --- Run detection ---
results_overload = []
for category, image_path in test_set_overload:
    image = Image.open(image_path).convert("RGB")
    person_boxes = _person_nms(_run_grounding(image, PERSON_PROMPT))
    rider_count = len(person_boxes)

    # Signal 1: box count
    box_overload = rider_count > MAX_LEGAL_RIDERS_2W_EVAL

    # Signal 2: caption on full image
    full_caption = _run_caption(image)
    caption_overload = caption_indicates_overloading(full_caption)

    # Either signal triggers overloading
    predicted_overload = box_overload or caption_overload
    expected_overload = category in overloading_categories

    correct = "OK" if predicted_overload == expected_overload else "WRONG"
    print(f"[{correct}] {os.path.basename(image_path)} -> riders={rider_count}, box={box_overload}, caption={caption_overload} | {full_caption[:80]}")

    results_overload.append({
        "category": category,
        "image_path": image_path,
        "crop": image,
        "rider_count": rider_count,
        "predicted_overload": predicted_overload,
        "expected_overload": expected_overload,
        "caption": full_caption,
    })

# --- Summary ---
total = len(results_overload)
correct_count = sum(1 for r in results_overload if r["predicted_overload"] == r["expected_overload"])
print(f"\nAccuracy: {correct_count}/{total} = {correct_count/total:.1%}")

# --- Display WRONG only ---
wrong_results = [r for r in results_overload if r["predicted_overload"] != r["expected_overload"]]
print(f"Wrong predictions: {len(wrong_results)} / {total}")

n = len(wrong_results)
if n == 0:
    print("No wrong predictions!")
else:
    cols = 5
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 5 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(wrong_results):
        title = (
            f"[WRONG] {r['category']}\n"
            f"riders={r['rider_count']} | pred={r['predicted_overload']} | exp={r['expected_overload']}\n"
            + textwrap.fill(r["caption"][:80], 35)
        )
        axes[i].imshow(r["crop"])
        axes[i].set_title(title, fontsize=7, color="red")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Wrong predictions: {len(wrong_results)} / {total}", fontsize=10)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()


# In[ ]:


# =====================================================
# INSPECT YOLO-FORMAT SEATBELT DATASET
# =====================================================

import os, glob, re
from collections import Counter

SEATBELT_DATASET_ROOT = "/kaggle/input/datasets/manyaj123456/setabelt1"

# --- Locate and read data.yaml for class names ---
yaml_path = None
for root, dirs, files in os.walk(SEATBELT_DATASET_ROOT):
    for f in files:
        if f.lower() in ("data.yaml", "data.yml"):
            yaml_path = os.path.join(root, f)
            break
    if yaml_path:
        break

class_names = []
if yaml_path:
    print(f"Found data.yaml at: {yaml_path}\n")
    with open(yaml_path, "r") as f:
        yaml_content = f.read()
    print(yaml_content)

    m = re.search(r"names:\s*\[(.*?)\]", yaml_content, re.DOTALL)
    if m:
        class_names = [c.strip().strip("'\"") for c in m.group(1).split(",")]
    else:
        m2 = re.findall(r"^\s*\d+:\s*(.+)$", yaml_content, re.MULTILINE)
        if m2:
            class_names = [c.strip() for c in m2]
else:
    print("No data.yaml found in the dataset root -- will need class names manually.")

print("\nDetected class names:", class_names)

# --- Walk each split, pair images with their YOLO label files ---
eval_image_paths_sb = []  # (classes_present: set[int], image_path, split)
SPLITS = ["valid", "train", "test"]

for split in SPLITS:
    img_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "labels")
    if not os.path.isdir(img_dir):
        continue
    img_files = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.jpeg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        classes_present = set()
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as lf:
                for line in lf:
                    line = line.strip()
                    if line:
                        classes_present.add(int(line.split()[0]))
        eval_image_paths_sb.append((classes_present, img_path, split))

print(f"\nTotal images found: {len(eval_image_paths_sb)}")
print("By split:", dict(Counter(s for _, _, s in eval_image_paths_sb)))

with_labels = sum(1 for c, _, _ in eval_image_paths_sb if c)
without_labels = sum(1 for c, _, _ in eval_image_paths_sb if not c)
print(f"Images with >=1 bounding box: {with_labels}")
print(f"Images with NO bounding boxes: {without_labels}")

class_image_counts = Counter()
for classes_present, _, _ in eval_image_paths_sb:
    for c in classes_present:
        class_image_counts[c] += 1

print("\nPer-class image counts (class_id -> images containing it):")
for cid, cnt in sorted(class_image_counts.items()):
    name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
    print(f"  {cid} ({name}): {cnt}")


# In[ ]:


import re

NOT_WORN_PATTERNS_SEATBELT = [
    r"without (?:a |any )?seatbelt",
    r"not wearing (?:a |any )?seatbelt",
    r"no seatbelt",
    r"seatbelt (?:unbuckled|undone|off|removed)",
    r"seatbelt (?:hanging|dangling)",
    r"not .{0,15}seatbelt",
    r"not buckled",
    r"unbuckled",
]

WORN_PATTERNS_SEATBELT = [
    r"wearing (?:a |the )?seatbelt",
    r"seatbelt (?:on|across|over) (?:his|her|their|the) (?:chest|shoulder|body|torso)",
    r"with (?:a |the )?seatbelt on",
    r"buckled (?:up|in)",
    r"seatbelt strap",
    r"strap (?:across|over) (?:his|her|their) (?:chest|shoulder)",
    r"wearing .{0,50}seatbelt",
    r"\bseatbelt\b",
]

def caption_indicates_seatbelt_worn(caption):
    text = caption.lower()
    for pattern in NOT_WORN_PATTERNS_SEATBELT:
        if re.search(pattern, text):
            return False
    for pattern in WORN_PATTERNS_SEATBELT:
        if re.search(pattern, text):
            return True
    return False


# In[ ]:


# =====================================================
# CAPTION-BASED SEATBELT CHECK -- YOLO DATASET (per-person instances)
# =====================================================

import os, math
import textwrap
import matplotlib.pyplot as plt
from PIL import Image

SEATBELT_DATASET_ROOT = "/kaggle/input/datasets/manyaj123456/setabelt1"
SEATBELT_CLASS_NAMES = ['1', '2', 'person-noseatbelt', 'person-seatbelt', 'seatbelt']
PERSON_NOSEATBELT_ID = SEATBELT_CLASS_NAMES.index('person-noseatbelt')  # 2
PERSON_SEATBELT_ID = SEATBELT_CLASS_NAMES.index('person-seatbelt')      # 3

EVAL_SPLITS_SB = ["valid", "test"]   # held-out splits; add "train" for more samples
MAX_EVAL_INSTANCES = 200

def _yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

# --- Collect per-person instances with ground truth from YOLO labels ---
instances_sb = []  # (expected_worn, image_path, person_box)

for split in EVAL_SPLITS_SB:
    img_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "labels")
    if not os.path.isdir(img_dir):
        continue
    for fname in sorted(os.listdir(img_dir)):
        if os.path.splitext(fname)[1].lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        with open(lbl_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id not in (PERSON_NOSEATBELT_ID, PERSON_SEATBELT_ID):
                    continue
                x_center, y_center, w, h = map(float, parts[1:5])
                box = _yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h)
                expected_worn = (class_id == PERSON_SEATBELT_ID)
                instances_sb.append((expected_worn, img_path, box))

print(f"Total labeled person instances found: {len(instances_sb)}")
print(f"  person-seatbelt (worn=True):    {sum(1 for e, _, _ in instances_sb if e)}")
print(f"  person-noseatbelt (worn=False): {sum(1 for e, _, _ in instances_sb if not e)}")

# --- Balance + cap the eval set ---
worn_instances = [i for i in instances_sb if i[0]]
notworn_instances = [i for i in instances_sb if not i[0]]
half = MAX_EVAL_INSTANCES // 2
test_set_sb = worn_instances[:half] + notworn_instances[:half]
print(f"\nTesting on {len(test_set_sb)} person instances "
      f"({sum(1 for e,_,_ in test_set_sb if e)} worn / {sum(1 for e,_,_ in test_set_sb if not e)} not worn)")

# --- Run captioning + verdict per instance ---
results_sb = []
for expected_worn, image_path, person_box in test_set_sb:
    image = Image.open(image_path).convert("RGB")
    occupant_crop = crop_with_padding(image, person_box, pad_frac=0.15)
    occupant_caption = _run_caption(occupant_crop)
    predicted_worn = caption_indicates_seatbelt_worn(occupant_caption)

    correct = "OK" if predicted_worn == expected_worn else "WRONG"
    print(f"[{correct}] {os.path.basename(image_path)} -> pred={predicted_worn}, expected={expected_worn} | {occupant_caption[:80]}")

    results_sb.append({
        "image_path": image_path,
        "crop": occupant_crop,
        "caption": occupant_caption,
        "predicted_worn": predicted_worn,
        "expected_worn": expected_worn,
    })

# --- Summary ---
total_sb = len(results_sb)
correct_count_sb = sum(1 for r in results_sb if r["predicted_worn"] == r["expected_worn"])
print(f"\nAccuracy: {correct_count_sb}/{total_sb} = {correct_count_sb/total_sb:.1%}")

# --- Display WRONG predictions only ---
wrong_results_sb = [r for r in results_sb if r["predicted_worn"] != r["expected_worn"]]
print(f"Wrong predictions: {len(wrong_results_sb)} / {total_sb}")

n = len(wrong_results_sb)
if n == 0:
    print("No wrong predictions!")
else:
    cols = 5
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 5 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(wrong_results_sb):
        title = (
            f"[WRONG] pred={r['predicted_worn']} | exp={r['expected_worn']}\n"
            + textwrap.fill(r["caption"][:90], 35)
        )
        axes[i].imshow(r["crop"])
        axes[i].set_title(title, fontsize=7, color="red")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Wrong predictions: {len(wrong_results_sb)} / {total_sb}", fontsize=10)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.show()


# In[ ]:


# =====================================================
# HELPER: florence_ground_seatbelt -- phrase grounding for "seatbelt"
# =====================================================

def florence_ground_seatbelt(image, phrase="seatbelt"):
    """Run Florence-2 <CAPTION_TO_PHRASE_GROUNDING> on `image`, asking it to
    locate `phrase`. Returns a list of [x1, y1, x2, y2] boxes in image pixel
    coordinates (empty list if nothing was grounded)."""
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = processor(text=task + phrase, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )
    result = parsed.get(task, {})
    return result.get("bboxes", [])

print("florence_ground_seatbelt() ready")


# In[ ]:


# =====================================================
# DEBUG: inspect the 3 false-negative captions
# =====================================================

wrong_images = ["463.jpg", "494.jpg", "555.jpg"]

for category, image_path in eval_image_paths:
    if category == "helmet" and os.path.basename(image_path) in wrong_images:
        image = Image.open(image_path).convert("RGB")
        person_boxes = _person_nms(_run_grounding(image, PERSON_PROMPT))

        print(f"\n=== {os.path.basename(image_path)} ===")
        if not person_boxes:
            print("No person box found -- caption never ran.")
            continue

        rider_crop = crop_with_padding(image, person_boxes[0], pad_frac=0.1)
        caption = _run_caption(rider_crop)
        print("Raw caption:", caption)
        print("caption_indicates_helmet_worn:", caption_indicates_helmet_worn(caption))

        # Show which patterns (if any) matched
        text = caption.lower()
        matched_not_worn = [p for p in NOT_WORN_PATTERNS if re.search(p, text)]
        matched_worn = [p for p in WORN_PATTERNS if re.search(p, text)]
        print("Matched NOT_WORN patterns:", matched_not_worn)
        print("Matched WORN patterns:", matched_worn)


# In[ ]:


# =====================================================
# DEBUG: FULL IMAGE (no crop) -- grounding + marked-box captioning
# Testing Florence-2 with full scene context instead of isolated crops
# =====================================================
import matplotlib.pyplot as plt
import textwrap
import os
from PIL import Image, ImageDraw

def draw_box_marker(image, box, color="red", width=4):
    """Draw a rectangle on a COPY of the full image to mark the person region."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle(box, outline=color, width=width)
    return marked

debug_samples = test_set_sb[:10]

fig, axes = plt.subplots(2, 10, figsize=(28, 7))
for i, (expected_worn, image_path, person_box) in enumerate(debug_samples):
    image = Image.open(image_path).convert("RGB")

    # ---- Strategy 1: full-image grounding for "seatbelt" ----
    seatbelt_boxes_full = florence_ground_seatbelt(image)
    grounding_found_full = len(seatbelt_boxes_full) > 0

    # Check overlap: does any grounded seatbelt box fall inside/near the person box?
    def boxes_overlap(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        return x2 > x1 and y2 > y1

    near_person = any(boxes_overlap(b, person_box) for b in seatbelt_boxes_full) if seatbelt_boxes_full else False

    # ---- Strategy 2: full image with person region marked, then caption ----
    marked_image = draw_box_marker(image, person_box)
    marked_caption = _run_caption(marked_image)

    print(f"--- {os.path.basename(image_path)} (expected_worn={expected_worn}) ---")
    print(f"  GROUNDING (full img) -> boxes_found={len(seatbelt_boxes_full)}, near_person_box={near_person}")
    print(f"  MARKED-BOX CAPTION   -> {marked_caption}")
    print()

    axes[0][i].imshow(image)
    for b in seatbelt_boxes_full:
        bx = plt.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], fill=False, edgecolor="lime", linewidth=1)
        axes[0][i].add_patch(bx)
    px = plt.Rectangle((person_box[0], person_box[1]), person_box[2]-person_box[0], person_box[3]-person_box[1],
                        fill=False, edgecolor="red", linewidth=1)
    axes[0][i].add_patch(px)
    axes[0][i].set_title(textwrap.fill(f"GROUND exp={expected_worn}\nnear_person={near_person}", 28), fontsize=6)
    axes[0][i].axis("off")

    axes[1][i].imshow(marked_image)
    axes[1][i].set_title(textwrap.fill(f"MARKED exp={expected_worn}\n{marked_caption[:60]}", 28), fontsize=6)
    axes[1][i].axis("off")

plt.tight_layout()
plt.show()


# In[ ]:


# =====================================================
# DEBUG: try Florence-2 VQA-style direct question on seatbelt status
# Testing whether Florence-2 can answer a targeted yes/no question
# instead of free-form captioning or object grounding
# =====================================================
import textwrap

def _run_vqa(image, question):
    """Attempt Florence-2 VQA task. If the checkpoint doesn't support a
    generic VQA task token, this will likely return garbage or echo the
    question -- that result itself is informative."""
    task = "<VQA>"
    prompt = task + question
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=50,
            num_beams=3,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    try:
        parsed = processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        return parsed.get(task, generated_text).strip()
    except Exception as e:
        # task token not recognized by this checkpoint's post-processor
        return f"[RAW/UNPARSED]: {generated_text}"

debug_samples = test_set_sb[:10]
question = "Is the person inside the car wearing a seatbelt?"

for expected_worn, image_path, person_box in debug_samples:
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=0.3)

    answer = _run_vqa(crop, question)
    print(f"--- {os.path.basename(image_path)} (expected_worn={expected_worn}) ---")
    print(f"  VQA ANSWER: {answer}")
    print()


# In[ ]:


print(len(image_paths), len(rows))
display_captions(image_paths, [r["caption"] for r in rows], cols=3)


# In[ ]:


# =====================================================
# RUN GROUNDING + CLAHE ON THE SAME 5 DEBUG IMAGES (no Qwen2-VL)
# =====================================================
import matplotlib.pyplot as plt
import textwrap, os

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

for i, (expected_worn, image_path, person_box) in enumerate(debug_samples):
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=0.15)
    clahe_crop = clahe_enhance(crop)

    seatbelt_boxes = florence_ground_seatbelt(crop)
    grounding_found = len(seatbelt_boxes) > 0

    clahe_caption = _run_caption(clahe_crop)

    print(f"--- {os.path.basename(image_path)} (expected_worn={expected_worn}) ---")
    print(f"  1) Florence-2 grounding -> boxes_found={len(seatbelt_boxes)}, boxes={seatbelt_boxes}")
    print(f"  2) CLAHE + caption      -> {clahe_caption}")
    print()

    axes[i].imshow(clahe_crop)
    axes[i].set_title(textwrap.fill(f"exp={expected_worn} | ground={grounding_found}", 28), fontsize=8)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


# In[ ]:


# =====================================================
# CELL 2a: INSTALL bitsandbytes (run this, then RESTART KERNEL before next cell)
# =====================================================
get_ipython().system('pip install -q -U bitsandbytes accelerate qwen-vl-utils')
print("Installed. NOW RESTART THE KERNEL (Run > Restart & Clear Output), then run the next cell.")


# ### Restart kernel now
# 
# Run the cell above, wait for install to finish, then **Run > Restart & Clear Output** before running the cell below. `bitsandbytes` will not pick up the upgraded version in the same kernel session.

# In[ ]:


# =====================================================
# CELL 2b: LOAD Qwen2-VL-7B (4-bit) FOR YES/NO SEATBELT CHECK
# Run this fresh AFTER restarting the kernel (post bitsandbytes install).
# =====================================================
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor as QwenProcessor, BitsAndBytesConfig

QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

qwen_processor = QwenProcessor.from_pretrained(QWEN_MODEL_ID)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
qwen_model.eval()
print("Qwen2-VL-7B loaded (4-bit)")


# In[ ]:


def qwen_seatbelt_yesno(pil_image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer with only one word: yes or no."},
        ],
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=10, do_sample=False)

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    return qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# In[6]:


# =====================================================
# REBUILD test_set_sb (standalone, no Florence-2 needed)
# =====================================================
import os
from PIL import Image

SEATBELT_DATASET_ROOT = "/kaggle/input/datasets/manyaj123456/setabelt1"
SEATBELT_CLASS_NAMES = ['1', '2', 'person-noseatbelt', 'person-seatbelt', 'seatbelt']
PERSON_NOSEATBELT_ID = SEATBELT_CLASS_NAMES.index('person-noseatbelt')
PERSON_SEATBELT_ID = SEATBELT_CLASS_NAMES.index('person-seatbelt')

EVAL_SPLITS_SB = ["valid", "test"]
MAX_EVAL_INSTANCES = 200

def _yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h):
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]

def crop_with_padding(image, box, pad_frac=0.1):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(image.width,  int(x2 + pad_x))
    cy2 = min(image.height, int(y2 + pad_y))
    return image.crop((cx1, cy1, cx2, cy2))

instances_sb = []

for split in EVAL_SPLITS_SB:
    img_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(SEATBELT_DATASET_ROOT, split, "labels")
    if not os.path.isdir(img_dir):
        continue
    for fname in sorted(os.listdir(img_dir)):
        if os.path.splitext(fname)[1].lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        with open(lbl_path, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id not in (PERSON_NOSEATBELT_ID, PERSON_SEATBELT_ID):
                    continue
                x_center, y_center, w, h = map(float, parts[1:5])
                box = _yolo_to_pixel_box(x_center, y_center, w, h, img_w, img_h)
                expected_worn = (class_id == PERSON_SEATBELT_ID)
                instances_sb.append((expected_worn, img_path, box))

worn_instances = [i for i in instances_sb if i[0]]
notworn_instances = [i for i in instances_sb if not i[0]]
half = MAX_EVAL_INSTANCES // 2
test_set_sb = worn_instances[:half] + notworn_instances[:half]

print(f"Total labeled person instances found: {len(instances_sb)}")
print(f"  person-seatbelt (worn=True):    {sum(1 for e, _, _ in instances_sb if e)}")
print(f"  person-noseatbelt (worn=False): {sum(1 for e, _, _ in instances_sb if not e)}")
print(f"\ntest_set_sb ready: {len(test_set_sb)} instances "
      f"({sum(1 for e,_,_ in test_set_sb if e)} worn / {sum(1 for e,_,_ in test_set_sb if not e)} not worn)")


# 

# In[ ]:


# =====================================================
# EVAL (40-sample test): Qwen2-VL-7B with geometric strap prompt
# + explicit cache clearing between calls to fix loop-degradation
# =====================================================
import math
import gc
import matplotlib.pyplot as plt
import textwrap

def qwen_seatbelt_yesno(pil_image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer with only one word: yes or no."},
        ],
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    result = qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # explicit cleanup to prevent state/memory drift across loop iterations
    del inputs, generated_ids, trimmed
    gc.collect()
    torch.cuda.empty_cache()

    return result

def qwen_answer_to_bool(answer):
    text = answer.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None

worn_only = [t for t in test_set_sb if t[0] == True]
notworn_only = [t for t in test_set_sb if t[0] == False]
eval_sample_40 = worn_only[:20] + notworn_only[:20]

results_qwen_40 = []
unparsed_count = 0

for expected_worn, image_path, person_box in eval_sample_40:
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=0.4)

    answer_raw = qwen_seatbelt_yesno(crop)
    predicted_worn = qwen_answer_to_bool(answer_raw)

    if predicted_worn is None:
        unparsed_count += 1

    correct = "OK" if predicted_worn == expected_worn else "WRONG"
    print(f"[{correct}] {os.path.basename(image_path)} -> pred={predicted_worn}, expected={expected_worn} | raw='{answer_raw}'")

    results_qwen_40.append({
        "image_path": image_path,
        "crop": crop,
        "raw_answer": answer_raw,
        "predicted_worn": predicted_worn,
        "expected_worn": expected_worn,
    })

total_40 = len(results_qwen_40)
correct_count_40 = sum(1 for r in results_qwen_40 if r["predicted_worn"] == r["expected_worn"])
print(f"\nAccuracy: {correct_count_40}/{total_40} = {correct_count_40/total_40:.1%}")
print(f"Unparsed answers: {unparsed_count}/{total_40}")

tp = sum(1 for r in results_qwen_40 if r["predicted_worn"] == True and r["expected_worn"] == True)
fp = sum(1 for r in results_qwen_40 if r["predicted_worn"] == True and r["expected_worn"] == False)
tn = sum(1 for r in results_qwen_40 if r["predicted_worn"] == False and r["expected_worn"] == False)
fn = sum(1 for r in results_qwen_40 if r["predicted_worn"] == False and r["expected_worn"] == True)
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
if (tp + fn) > 0:
    print(f"Recall: {tp/(tp+fn):.1%}")
if (tp + fp) > 0:
    print(f"Precision: {tp/(tp+fp):.1%}")


# In[ ]:


# =====================================================
# DIAGNOSTIC on the specific failing image: 1625227980294
# =====================================================

target_filename = "1625227980294_jpg.rf.ce95df9439a6bbc98a6f95a54d23c370.jpg"

target = None
for expected_worn, image_path, person_box in test_set_sb:
    if target_filename in image_path:
        target = (expected_worn, image_path, person_box)
        break

expected_worn, image_path, person_box = target
image = Image.open(image_path).convert("RGB")
crop = crop_with_padding(image, person_box, pad_frac=0.4)

plt.figure(figsize=(6, 7))
plt.imshow(crop)
plt.axis("off")
plt.title(f"{os.path.basename(image_path)}\nexpected_worn={expected_worn}, crop_size={crop.size}")
plt.show()

def ask_qwen(pil_image, question, max_tokens=80):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": question},
        ],
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    return qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

questions = [
    "Is the person wearing a seatbelt? Answer with only one word: yes or no.",
    "Describe everything visible across the person's chest and shoulder area.",
    "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer yes or no.",
    "Describe the lighting and visibility of this image in one sentence.",
]

for q in questions:
    answer = ask_qwen(crop, q)
    print(f"Q: {q}")
    print(f"A: {answer}\n")


# In[ ]:


def qwen_seatbelt_yesno(pil_image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer with only one word: yes or no."},
        ],
    }]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=30, do_sample=False)
    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    return qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# In[ ]:


answer = qwen_seatbelt_yesno(crop)
print(f"answer='{answer}'")


# In[ ]:


# =====================================================
# DIRECT DIFF: compare the two functions' actual inputs
# =====================================================

question = "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer with only one word: yes or no."

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": crop},
        {"type": "text", "text": question},
    ],
}]
text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== FULL TEMPLATED PROMPT TEXT ===")
print(repr(text))
print()

inputs = qwen_processor(text=[text], images=[crop], return_tensors="pt").to(qwen_model.device)
print("=== input_ids shape ===", inputs.input_ids.shape)
print("=== pixel_values shape ===", inputs.pixel_values.shape if hasattr(inputs, "pixel_values") else "N/A")

# Now run generation with explicit max_new_tokens and print raw token ids before decoding
with torch.no_grad():
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=30, do_sample=False)

trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
print("\n=== raw generated token ids ===", trimmed)
print("=== decoded (skip_special_tokens=True) ===", repr(qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0]))
print("=== decoded (skip_special_tokens=False) ===", repr(qwen_processor.batch_decode(trimmed, skip_special_tokens=False)[0]))


# In[ ]:


# =====================================================
# SIDE-BY-SIDE: run ask_qwen and qwen_seatbelt_yesno back-to-back,
# same image, same cell, to catch the actual divergence
# =====================================================

question = "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer with only one word: yes or no."

# --- via ask_qwen (max_tokens=80) ---
answer_ask_qwen = ask_qwen(crop, question, max_tokens=80)
print(f"ask_qwen()            -> '{answer_ask_qwen}'")

# --- via qwen_seatbelt_yesno (uses its own hardcoded prompt text) ---
answer_eval_fn = qwen_seatbelt_yesno(crop)
print(f"qwen_seatbelt_yesno()  -> '{answer_eval_fn}'")

# --- print both functions' source to compare exact prompt strings ---
import inspect
print("\n--- ask_qwen source ---")
print(inspect.getsource(ask_qwen))
print("\n--- qwen_seatbelt_yesno source ---")
print(inspect.getsource(qwen_seatbelt_yesno))


# In[ ]:


import inspect
print(inspect.getsource(qwen_seatbelt_yesno))


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (expected_worn, image_path, person_box) in enumerate(eval_sample_40[:5]):
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=0.4)
    axes[i].imshow(crop)
    axes[i].set_title(f"{os.path.basename(image_path)[:20]}\nexp={expected_worn}\nsize={crop.size}", fontsize=8)
    axes[i].axis("off")
plt.tight_layout()
plt.show()


# In[ ]:


# =====================================================
# VISUAL CHECK: 20 large crops where Qwen said "No" but expected=True
# =====================================================
import matplotlib.pyplot as plt
import textwrap

false_negatives = [r for r in results_qwen if r["expected_worn"] == True and r["predicted_worn"] == False]
sample = false_negatives[:20]

cols = 5
rows_n = (len(sample) + cols - 1) // cols
fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 6 * rows_n))
axes = axes.flatten()

for i, r in enumerate(sample):
    axes[i].imshow(r["crop"])
    axes[i].set_title(
        textwrap.fill(f"{os.path.basename(r['image_path'])}\nraw='{r['raw_answer']}' | crop_size={r['crop'].size}", 40),
        fontsize=8
    )
    axes[i].axis("off")

for j in range(len(sample), len(axes)):
    axes[j].axis("off")

plt.suptitle(f"False negatives (expected worn, Qwen said No): {len(sample)} shown of {len(false_negatives)} total", fontsize=12)
plt.tight_layout()
plt.show()


# In[ ]:


# =====================================================
# SINGLE-IMAGE DIAGNOSTIC: is Qwen seeing the belt at all,
# or just failing the yes/no judgment?
# =====================================================

# Pick a known clean example with an obvious diagonal strap
target_filename = "1625227980423_jpg.rf.6736873b5e0014fb32de53e58c9b8e88.jpg"

target = None
for expected_worn, image_path, person_box in test_set_sb:
    if target_filename in image_path:
        target = (expected_worn, image_path, person_box)
        break

if target is None:
    print("Target not found in test_set_sb -- pick another filename from the printed results")
else:
    expected_worn, image_path, person_box = target
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=0.4)

    print(f"Testing: {os.path.basename(image_path)} (expected_worn={expected_worn}, crop_size={crop.size})\n")

    def ask_qwen(pil_image, question, max_tokens=60):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": question},
            ],
        }]
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(text=[text], images=[pil_image], return_tensors="pt").to(qwen_model.device)
        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        return qwen_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    questions = [
        "Is the person wearing a seatbelt? Answer with only one word: yes or no.",
        "Describe everything visible across the person's chest and shoulder area.",
        "Is there a diagonal strap crossing the person's chest from shoulder to hip? Answer yes or no.",
        "What objects or markings do you see on the driver's body and clothing?",
        "Look carefully at the driver's torso. Do you see a seatbelt strap? Think step by step, then answer yes or no.",
    ]

    plt.figure(figsize=(6, 7))
    plt.imshow(crop)
    plt.axis("off")
    plt.title(f"{os.path.basename(image_path)}\nexpected_worn={expected_worn}")
    plt.show()

    for q in questions:
        answer = ask_qwen(crop, q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


# In[2]:


get_ipython().run_cell_magic('writefile', 'seatbelt_crop_classifier.py', '#!/usr/bin/env python3\n"""\nTrain/evaluate a binary seatbelt classifier from a YOLOv8 dataset.\n\nThe dataset labels are expected to contain:\n  2: person-noseatbelt -> class 0\n  3: person-seatbelt   -> class 1\n\nUsage:\n  python seatbelt_crop_classifier.py train --data "/path/to/seat belt.v1i.yolov8"\n  python seatbelt_crop_classifier.py predict --weights runs/seatbelt_crop/best.pt --image crop.jpg\n"""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport random\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\nfrom PIL import Image, ImageOps\nfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix\nfrom torch.utils.data import DataLoader, Dataset, WeightedRandomSampler\nfrom torchvision import models, transforms\n\n\nPERSON_NO_SEATBELT = 2\nPERSON_SEATBELT = 3\nLABEL_MAP = {PERSON_NO_SEATBELT: 0, PERSON_SEATBELT: 1}\nCLASS_NAMES = ["not_worn", "worn"]\n\n\ndef seed_everything(seed: int) -> None:\n    random.seed(seed)\n    np.random.seed(seed)\n    torch.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)\n    torch.backends.cudnn.benchmark = False\n\n\n@dataclass(frozen=True)\nclass CropSample:\n    image_path: Path\n    label: int\n    box_xywhn: tuple[float, float, float, float]\n\n\ndef yolo_to_xyxy(\n    box_xywhn: tuple[float, float, float, float],\n    width: int,\n    height: int,\n    pad: float,\n) -> tuple[int, int, int, int]:\n    cx, cy, bw, bh = box_xywhn\n    x1 = (cx - bw / 2.0) * width\n    y1 = (cy - bh / 2.0) * height\n    x2 = (cx + bw / 2.0) * width\n    y2 = (cy + bh / 2.0) * height\n\n    px = (x2 - x1) * pad\n    py = (y2 - y1) * pad\n    x1 = max(0, int(round(x1 - px)))\n    y1 = max(0, int(round(y1 - py)))\n    x2 = min(width, int(round(x2 + px)))\n    y2 = min(height, int(round(y2 + py)))\n    return x1, y1, x2, y2\n\n\ndef matching_image(label_path: Path, images_dir: Path) -> Path | None:\n    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):\n        candidate = images_dir / f"{label_path.stem}{ext}"\n        if candidate.exists():\n            return candidate\n    return None\n\n\ndef collect_samples(data_root: Path, split: str) -> list[CropSample]:\n    images_dir = data_root / split / "images"\n    labels_dir = data_root / split / "labels"\n    samples: list[CropSample] = []\n\n    for label_path in sorted(labels_dir.glob("*.txt")):\n        image_path = matching_image(label_path, images_dir)\n        if image_path is None:\n            continue\n\n        for line in label_path.read_text().splitlines():\n            parts = line.strip().split()\n            if len(parts) < 5:\n                continue\n            cls_id = int(float(parts[0]))\n            if cls_id not in LABEL_MAP:\n                continue\n            box = tuple(float(x) for x in parts[1:5])\n            samples.append(CropSample(image_path, LABEL_MAP[cls_id], box))  # type: ignore[arg-type]\n\n    return samples\n\n\ndef limit_balanced(samples: list[CropSample], limit: int | None, seed: int) -> list[CropSample]:\n    if limit is None or limit <= 0 or len(samples) <= limit:\n        return samples\n    rng = random.Random(seed)\n    by_label = {0: [], 1: []}\n    for sample in samples:\n        by_label[sample.label].append(sample)\n    half = max(1, limit // 2)\n    picked: list[CropSample] = []\n    for label in (0, 1):\n        bucket = by_label[label]\n        rng.shuffle(bucket)\n        picked.extend(bucket[: min(half, len(bucket))])\n    if len(picked) < limit:\n        rest = [s for s in samples if s not in set(picked)]\n        rng.shuffle(rest)\n        picked.extend(rest[: limit - len(picked)])\n    rng.shuffle(picked)\n    return picked\n\n\nclass SeatbeltCropDataset(Dataset):\n    def __init__(self, samples: list[CropSample], train: bool, crop_pad: float, image_size: int):\n        self.samples = samples\n        self.crop_pad = crop_pad\n        if train:\n            self.tf = transforms.Compose(\n                [\n                    transforms.Resize((image_size, image_size)),\n                    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.2, 0.05)], p=0.8),\n                    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),\n                    transforms.ToTensor(),\n                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n                ]\n            )\n        else:\n            self.tf = transforms.Compose(\n                [\n                    transforms.Resize((image_size, image_size)),\n                    transforms.ToTensor(),\n                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n                ]\n            )\n\n    def __len__(self) -> int:\n        return len(self.samples)\n\n    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:\n        sample = self.samples[idx]\n        with Image.open(sample.image_path) as img:\n            img = ImageOps.exif_transpose(img).convert("RGB")\n            box = yolo_to_xyxy(sample.box_xywhn, img.width, img.height, self.crop_pad)\n            crop = img.crop(box)\n        return self.tf(crop), torch.tensor(sample.label, dtype=torch.long)\n\n\ndef build_model() -> nn.Module:\n    model = models.mobilenet_v3_small(weights=None)\n    in_features = model.classifier[-1].in_features\n    model.classifier[-1] = nn.Linear(in_features, 2)\n    return model\n\n\ndef make_loader(\n    samples: list[CropSample],\n    train: bool,\n    batch_size: int,\n    image_size: int,\n    crop_pad: float,\n    workers: int,\n    pin_memory: bool,\n) -> DataLoader:\n    dataset = SeatbeltCropDataset(samples, train=train, crop_pad=crop_pad, image_size=image_size)\n    if not train:\n        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)\n\n    labels = np.array([s.label for s in samples])\n    counts = np.bincount(labels, minlength=2)\n    weights = 1.0 / np.maximum(counts, 1)\n    sample_weights = torch.DoubleTensor([weights[s.label] for s in samples])\n    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)\n    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=pin_memory)\n\n\n@torch.inference_mode()\ndef evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:\n    model.eval()\n    y_true: list[int] = []\n    y_pred: list[int] = []\n    y_prob: list[float] = []\n\n    for images, labels in loader:\n        images = images.to(device)\n        logits = model(images)\n        probs = torch.softmax(logits, dim=1)[:, 1]\n        preds = (probs >= 0.5).long().cpu().numpy().tolist()\n        y_pred.extend(preds)\n        y_prob.extend(probs.cpu().numpy().tolist())\n        y_true.extend(labels.numpy().tolist())\n\n    return {\n        "accuracy": accuracy_score(y_true, y_pred),\n        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),\n        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0),\n        "n": len(y_true),\n        "positive_rate": float(np.mean(y_pred)) if y_pred else 0.0,\n        "mean_worn_probability": float(np.mean(y_prob)) if y_prob else 0.0,\n    }\n\n\ndef train(args: argparse.Namespace) -> None:\n    seed_everything(args.seed)\n    data_root = Path(args.data).expanduser().resolve()\n    out_dir = Path(args.out).expanduser().resolve()\n    out_dir.mkdir(parents=True, exist_ok=True)\n\n    train_samples = collect_samples(data_root, "train")\n    valid_samples = collect_samples(data_root, "valid")\n    test_samples = collect_samples(data_root, "test")\n    train_samples = limit_balanced(train_samples, args.limit_samples, args.seed)\n    valid_samples = limit_balanced(valid_samples, args.limit_valid, args.seed)\n    test_samples = limit_balanced(test_samples, args.limit_valid, args.seed)\n\n    print(f"samples: train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples)}", flush=True)\n    print(\n        "train labels:",\n        {name: sum(s.label == i for s in train_samples) for i, name in enumerate(CLASS_NAMES)},\n        flush=True,\n    )\n    print(\n        "valid labels:",\n        {name: sum(s.label == i for s in valid_samples) for i, name in enumerate(CLASS_NAMES)},\n        flush=True,\n    )\n    print(\n        "test labels:",\n        {name: sum(s.label == i for s in test_samples) for i, name in enumerate(CLASS_NAMES)},\n        flush=True,\n    )\n\n    if not train_samples or not valid_samples:\n        raise SystemExit("Missing train/valid samples. Check the dataset path and YOLO label classes.")\n\n    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")\n    print("device:", device, flush=True)\n\n    pin_memory = device.type == "cuda"\n    train_loader = make_loader(\n        train_samples, True, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory\n    )\n    valid_loader = make_loader(\n        valid_samples, False, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory\n    )\n    test_loader = (\n        make_loader(test_samples, False, args.batch_size, args.image_size, args.crop_pad, args.workers, pin_memory)\n        if test_samples\n        else None\n    )\n\n    model = build_model().to(device)\n    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)\n    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)\n    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)\n\n    best_acc = -1.0\n    best_path = out_dir / "best.pt"\n\n    for epoch in range(1, args.epochs + 1):\n        model.train()\n        running_loss = 0.0\n        seen = 0\n        for images, labels in train_loader:\n            images = images.to(device, non_blocking=True)\n            labels = labels.to(device, non_blocking=True)\n            optimizer.zero_grad(set_to_none=True)\n            logits = model(images)\n            loss = criterion(logits, labels)\n            loss.backward()\n            optimizer.step()\n            running_loss += float(loss.item()) * images.size(0)\n            seen += images.size(0)\n        scheduler.step()\n\n        valid_metrics = evaluate(model, valid_loader, device)\n        print(\n            f"epoch {epoch:02d}/{args.epochs} "\n            f"loss={running_loss / max(seen, 1):.4f} "\n            f"valid_acc={valid_metrics[\'accuracy\']:.4f} "\n            f"valid_pos_rate={valid_metrics[\'positive_rate\']:.3f}",\n            flush=True,\n        )\n\n        if valid_metrics["accuracy"] > best_acc:\n            best_acc = valid_metrics["accuracy"]\n            torch.save(\n                {\n                    "model": model.state_dict(),\n                    "class_names": CLASS_NAMES,\n                    "image_size": args.image_size,\n                    "crop_pad": args.crop_pad,\n                    "valid_metrics": valid_metrics,\n                },\n                best_path,\n            )\n\n    checkpoint = torch.load(best_path, map_location=device)\n    model.load_state_dict(checkpoint["model"])\n    final = {"valid": evaluate(model, valid_loader, device)}\n    if test_loader is not None:\n        final["test"] = evaluate(model, test_loader, device)\n\n    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))\n    print("\\nBEST WEIGHTS:", best_path, flush=True)\n    print("\\nVALID REPORT\\n", final["valid"]["report"], flush=True)\n    if "test" in final:\n        print("\\nTEST REPORT\\n", final["test"]["report"], flush=True)\n    print("metrics saved:", out_dir / "metrics.json", flush=True)\n\n\ndef predict(args: argparse.Namespace) -> None:\n    weights = Path(args.weights).expanduser().resolve()\n    checkpoint = torch.load(weights, map_location="cpu")\n    image_size = int(checkpoint.get("image_size", 224))\n\n    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")\n    model = build_model().to(device)\n    model.load_state_dict(checkpoint["model"])\n    model.eval()\n\n    tf = transforms.Compose(\n        [\n            transforms.Resize((image_size, image_size)),\n            transforms.ToTensor(),\n            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n        ]\n    )\n\n    with Image.open(Path(args.image).expanduser()) as img:\n        img = ImageOps.exif_transpose(img).convert("RGB")\n        tensor = tf(img).unsqueeze(0).to(device)\n\n    with torch.inference_mode():\n        prob_worn = torch.softmax(model(tensor), dim=1)[0, 1].item()\n    label = "worn" if prob_worn >= args.threshold else "not_worn"\n    print(json.dumps({"label": label, "prob_worn": prob_worn, "threshold": args.threshold}, indent=2))\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser()\n    sub = parser.add_subparsers(dest="command", required=True)\n\n    train_p = sub.add_parser("train")\n    train_p.add_argument("--data", default="/kaggle/input/datasets/manyaj123456/setabelt1")\n    train_p.add_argument("--out", default="runs/seatbelt_crop")\n    train_p.add_argument("--epochs", type=int, default=30)\n    train_p.add_argument("--batch-size", type=int, default=48)\n    train_p.add_argument("--image-size", type=int, default=224)\n    train_p.add_argument("--crop-pad", type=float, default=0.18)\n    train_p.add_argument("--lr", type=float, default=3e-4)\n    train_p.add_argument("--weight-decay", type=float, default=1e-4)\n    train_p.add_argument("--label-smoothing", type=float, default=0.03)\n    train_p.add_argument("--seed", type=int, default=7)\n    train_p.add_argument("--workers", type=int, default=0)\n    train_p.add_argument("--limit-samples", type=int, default=0)\n    train_p.add_argument("--limit-valid", type=int, default=0)\n    train_p.add_argument("--cpu", action="store_true")\n    train_p.set_defaults(func=train)\n\n    pred_p = sub.add_parser("predict")\n    pred_p.add_argument("--weights", required=True)\n    pred_p.add_argument("--image", required=True)\n    pred_p.add_argument("--threshold", type=float, default=0.5)\n    pred_p.add_argument("--cpu", action="store_true")\n    pred_p.set_defaults(func=predict)\n\n    return parser.parse_args()\n\n\nif __name__ == "__main__":\n    parsed = parse_args()\n    parsed.func(parsed)\n')


# In[3]:


get_ipython().system('python seatbelt_crop_classifier.py train    --out runs/seatbelt_crop    --epochs 30    --batch-size 48    --workers 2')


# In[7]:


from PIL import Image

target_filename = "1625227980451_jpg.rf.f01eef38846c1dc90ac460dfd8d6debf.jpg"

for expected_worn, image_path, person_box in test_set_sb:
    if target_filename in image_path:
        image = Image.open(image_path).convert("RGB")
        crop = crop_with_padding(image, person_box, pad_frac=0.18)
        crop.save("/kaggle/working/test_crop.jpg")
        print(f"Saved crop. expected_worn={expected_worn}")
        break


# In[ ]:


get_ipython().system('python seatbelt_crop_classifier.py predict    --weights runs/seatbelt_crop/best.pt    --image /kaggle/working/test_crop.jpg')


# In[8]:


# =====================================================
# FULL EVAL: trained MobileNetV3 seatbelt classifier on all labeled instances
# =====================================================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import math
import matplotlib.pyplot as plt
import textwrap

# --- Load the trained model once ---
WEIGHTS_PATH = "runs/seatbelt_crop/best.pt"

def build_model():
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
image_size = checkpoint.get("image_size", 224)
crop_pad_trained = checkpoint.get("crop_pad", 0.18)

clf_model = build_model().to(device)
clf_model.load_state_dict(checkpoint["model"])
clf_model.eval()
print(f"Loaded classifier. image_size={image_size}, crop_pad={crop_pad_trained}")

clf_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.inference_mode()
def classify_seatbelt(pil_crop):
    tensor = clf_transform(pil_crop).unsqueeze(0).to(device)
    prob_worn = torch.softmax(clf_model(tensor), dim=1)[0, 1].item()
    return prob_worn >= 0.5, prob_worn

# --- Run over ALL labeled instances (use instances_sb for the full set, not just test_set_sb) ---
eval_full = instances_sb  # full labeled set; swap to test_set_sb if you only want the capped 200

results_clf = []
for expected_worn, image_path, person_box in eval_full:
    image = Image.open(image_path).convert("RGB")
    crop = crop_with_padding(image, person_box, pad_frac=crop_pad_trained)
    predicted_worn, prob_worn = classify_seatbelt(crop)

    results_clf.append({
        "image_path": image_path,
        "crop": crop,
        "predicted_worn": predicted_worn,
        "prob_worn": prob_worn,
        "expected_worn": expected_worn,
    })

# --- Summary ---
total = len(results_clf)
correct = sum(1 for r in results_clf if r["predicted_worn"] == r["expected_worn"])
print(f"\nTotal instances evaluated: {total}")
print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")

tp = sum(1 for r in results_clf if r["predicted_worn"] and r["expected_worn"])
fp = sum(1 for r in results_clf if r["predicted_worn"] and not r["expected_worn"])
tn = sum(1 for r in results_clf if not r["predicted_worn"] and not r["expected_worn"])
fn = sum(1 for r in results_clf if not r["predicted_worn"] and r["expected_worn"])
print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
if (tp + fn) > 0:
    print(f"Recall (worn correctly caught): {tp/(tp+fn):.1%}")
if (tp + fp) > 0:
    print(f"Precision (worn predictions correct): {tp/(tp+fp):.1%}")
if (tn + fp) > 0:
    print(f"Specificity (not_worn correctly caught): {tn/(tn+fp):.1%}")

# --- Show wrong predictions ---
wrong = [r for r in results_clf if r["predicted_worn"] != r["expected_worn"]]
print(f"\nWrong predictions: {len(wrong)} / {total}")

if wrong:
    n = len(wrong)
    cols = 5
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 5 * rows_n))
    axes = axes.flatten() if n > 1 else [axes]
    for i, r in enumerate(wrong):
        title = f"[WRONG] pred={r['predicted_worn']} (p={r['prob_worn']:.2f})\nexp={r['expected_worn']}"
        axes[i].imshow(r["crop"])
        axes[i].set_title(title, fontsize=7, color="red")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("No wrong predictions!")


# In[ ]:




