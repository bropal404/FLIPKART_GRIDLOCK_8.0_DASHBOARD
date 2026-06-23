# Traffic Violation Challan

| Field | Value |
|---|---|
| Challan ID | C5016E29 |
| Date and Time | 2026-06-23 18:03:05 |
| Source Image | extracted_1782217958_3.jpg |
| Verdict | VIOLATION |
| Registration Number | UP14FP8624 |
| Total Fine | INR 3000 |

## Violations

- Riding without helmet
- Triple Riding

## VLM Description

The image shows three men riding on the back of a motor scooter down a street, with a footpath on the right side of the road. The men are wearing casual clothing and appear to be enjoying the ride.

## VLM/YOLO Evidence

- YOLO detected: Riding without helmet
- VLM caption (on crop): The image shows three men riding on the back of a motor scooter down a street, with a footpath on the right side of the 
- VLM fallback: triple-riding pattern in caption.

## YOLO Detections

| Class | Confidence | Bounding Box |
|---|---:|---|
| no_helmet | 0.846 | [33, 8, 250, 452] |
| license_plate | 0.626 | [133, 332, 190, 370] |
| rider | 0.309 | [37, 8, 176, 357] |


## Images

| Original | YOLO Marked | Plate OCR |
|---|---|---|
| ![original](original.jpg) | ![marked](yolo_marked.jpg) | ![plate](plate_ocr.jpg) |

## No-Helmet Crops

- ![no_helmet_crop_0](no_helmet_crop_00_conf0.85.jpg) conf=0.85
