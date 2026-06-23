# Traffic Violation Challan

| Field | Value |
|---|---|
| Challan ID | 927D1CE9 |
| Date and Time | 2026-06-23 00:37:05 |
| Source Image | extracted_1782155204_3.jpg |
| Verdict | VIOLATION |
| Registration Number | 0L8CY9667 |
| Total Fine | INR 1000 |

## Violations

- Riding without helmet

## VLM Description

The image shows a man riding a motor scooter down a street, with a wall in the background. He is wearing a white t-shirt and black pants.

## VLM/YOLO Evidence

- YOLO detected: Riding without helmet
- VLM caption (on crop): The image shows a man riding a motor scooter down a street, with a wall in the background. He is wearing a white t-shirt

## YOLO Detections

| Class | Confidence | Bounding Box |
|---|---:|---|
| no_helmet | 0.891 | [38, 5, 192, 374] |
| license_plate | 0.752 | [86, 276, 136, 310] |


## Images

| Original | YOLO Marked | Plate OCR |
|---|---|---|
| ![original](original.jpg) | ![marked](yolo_marked.jpg) | ![plate](plate_ocr.jpg) |

## No-Helmet Crops

- ![no_helmet_crop_0](no_helmet_crop_00_conf0.89.jpg) conf=0.89
