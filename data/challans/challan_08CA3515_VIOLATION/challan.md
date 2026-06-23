# Traffic Violation Challan

| Field | Value |
|---|---|
| Challan ID | 08CA3515 |
| Date and Time | 2026-06-23 18:02:57 |
| Source Image | extracted_1782217958_2.jpg |
| Verdict | VIOLATION |
| Registration Number | DL8CY9567 |
| Total Fine | INR 1000 |

## Violations

- Riding without helmet

## VLM Description

The image shows a man riding a motor scooter down a street, wearing a white t-shirt and black pants.

## VLM/YOLO Evidence

- YOLO detected: Riding without helmet
- VLM caption (on crop): The image shows a man riding a motor scooter down a street, wearing a white t-shirt and black pants.

## YOLO Detections

| Class | Confidence | Bounding Box |
|---|---:|---|
| no_helmet | 0.891 | [38, 0, 202, 367] |
| license_plate | 0.722 | [86, 269, 137, 301] |


## Images

| Original | YOLO Marked | Plate OCR |
|---|---|---|
| ![original](original.jpg) | ![marked](yolo_marked.jpg) | ![plate](plate_ocr.jpg) |

## No-Helmet Crops

- ![no_helmet_crop_0](no_helmet_crop_00_conf0.89.jpg) conf=0.89
