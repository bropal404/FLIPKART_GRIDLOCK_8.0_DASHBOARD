# Traffic Violation Challan

| Field | Value |
|---|---|
| Challan ID | 2AA31B91 |
| Date and Time | 2026-06-23 00:57:42 |
| Source Image | extracted_1782156426_4.jpg |
| Verdict | VIOLATION |
| Registration Number | UP14FP8624 |
| Total Fine | INR 3000 |

## Violations

- Riding without helmet
- Triple Riding

## VLM Description

The image shows three men riding on the back of a motor scooter down a street, with a wall on the right side of the road.

## VLM/YOLO Evidence

- YOLO detected: Riding without helmet
- VLM caption (on crop): The image shows three men riding on the back of a motor scooter down a street, with a wall on the right side of the road
- VLM fallback: triple-riding pattern in caption.

## YOLO Detections

| Class | Confidence | Bounding Box |
|---|---:|---|
| no_helmet | 0.845 | [19, 0, 246, 390] |
| license_plate | 0.531 | [120, 290, 180, 328] |


## Images

| Original | YOLO Marked | Plate OCR |
|---|---|---|
| ![original](original.jpg) | ![marked](yolo_marked.jpg) | ![plate](plate_ocr.jpg) |

## No-Helmet Crops

- ![no_helmet_crop_0](no_helmet_crop_00_conf0.84.jpg) conf=0.84
