import os
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file

app = Flask(__name__, static_folder='data')

# Ensure folders
UPLOAD_FOLDER = Path('data/uploads')
CHALLAN_FOLDER = Path('data/challans')
EXTRACTED_FOLDER = Path('data/extracted')
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CHALLAN_FOLDER.mkdir(parents=True, exist_ok=True)
EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

processing_tasks = {}


# ── GPS / Location Extraction ─────────────────────────────────────────────────

def _dms_to_decimal(dms_values):
    """Convert EXIF DMS tuple [(deg,1),(min,1),(sec,100)] to decimal degrees."""
    try:
        def ratio(v):
            """Handle both IFDRational objects and plain (num, denom) tuples."""
            if hasattr(v, 'numerator') and hasattr(v, 'denominator'):
                return v.numerator / v.denominator if v.denominator else 0.0
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return v[0] / v[1] if v[1] else 0.0
            return float(v)
        d = ratio(dms_values[0])
        m = ratio(dms_values[1])
        s = ratio(dms_values[2])
        return d + m / 60.0 + s / 3600.0
    except Exception:
        return None


def extract_gps_location(file_path):
    """
    Attempt to extract GPS coordinates from image EXIF data.
    Returns a human-readable location string, or "Hyderabad (default,no geo tag in image)" if unavailable.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img = Image.open(str(file_path))
        exif_data = img._getexif()
        if not exif_data:
            return "Hyderabad (default,no geo tag in image)"

        # Build tag → value map
        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
        gps_info_raw = exif.get('GPSInfo')
        if not gps_info_raw:
            return "Hyderabad (default,no geo tag in image)"

        gps = {GPSTAGS.get(k, k): v for k, v in gps_info_raw.items()}

        lat_dms  = gps.get('GPSLatitude')
        lat_ref  = gps.get('GPSLatitudeRef', 'N')
        lon_dms  = gps.get('GPSLongitude')
        lon_ref  = gps.get('GPSLongitudeRef', 'E')

        if not lat_dms or not lon_dms:
            return "Hyderabad (default,no geo tag in image)"

        lat = _dms_to_decimal(lat_dms)
        lon = _dms_to_decimal(lon_dms)
        if lat is None or lon is None:
            return "Hyderabad (default,no geo tag in image)"

        if lat_ref == 'S':
            lat = -lat
        if lon_ref == 'W':
            lon = -lon

        # Attempt reverse-geocoding via Nominatim (no API key needed)
        try:
            import urllib.request, urllib.parse, json as _json
            params = urllib.parse.urlencode({
                'lat': f'{lat:.6f}',
                'lon': f'{lon:.6f}',
                'format': 'json',
                'zoom': 14,
                'addressdetails': 0,
            })
            req = urllib.request.Request(
                f'https://nominatim.openstreetmap.org/reverse?{params}',
                headers={'User-Agent': 'GRIDLOCK-Traffic-Engine/1.0'}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                geo = _json.loads(resp.read().decode())
            display_name = geo.get('display_name', '')
            if display_name:
                # Trim to first 3 address components for brevity
                parts = [p.strip() for p in display_name.split(',') if p.strip()]
                return ', '.join(parts[:3])
        except Exception:
            pass

        # Fallback: return raw coordinates
        return f"{lat:.5f}, {lon:.5f}"

    except Exception:
        return "Hyderabad (default,no geo tag in image)"

def log_task(task_id, msg):
    """Log a message to the console and to the specific task's log array."""
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(f"TASK {task_id} | {formatted_msg}")
    if task_id in processing_tasks:
        processing_tasks[task_id]['logs'].append(formatted_msg)
        processing_tasks[task_id]['status'] = msg

def process_image_task(task_id, file_path, pipeline_core, vehicle_model):
    try:
        log_task(task_id, 'Starting image processing pipeline...')
        processing_tasks[task_id]['progress'] = 20

        # Extract location from original image EXIF before any cropping
        log_task(task_id, 'Extracting GPS location from image metadata...')
        location = extract_gps_location(file_path)
        log_task(task_id, f'Location: {location}')

        # Capture timestamp from file metadata
        import datetime as _dt
        capture_date = _dt.datetime.now().strftime('%Y-%m-%d')
        capture_time = _dt.datetime.now().strftime('%H:%M')
        try:
            from PIL import Image as _PILImg
            from PIL.ExifTags import TAGS as _TAGS
            _exif_raw = _PILImg.open(str(file_path))._getexif()
            if _exif_raw:
                _exif = {_TAGS.get(k, k): v for k, v in _exif_raw.items()}
                _dt_str = _exif.get('DateTimeOriginal') or _exif.get('DateTime', '')
                if _dt_str:
                    _parsed = _dt.datetime.strptime(_dt_str.strip(), '%Y:%m:%d %H:%M:%S')
                    capture_date = _parsed.strftime('%Y-%m-%d')
                    capture_time = _parsed.strftime('%H:%M')
        except Exception:
            pass

        # Initial scan to find vehicles
        log_task(task_id, 'Running vehicle YOLO to locate vehicles...')
        
        results = vehicle_model.predict(str(file_path), conf=0.25, verbose=False)[0]
        boxes = results.boxes
        
        clusters = []
        if boxes is None or len(boxes) == 0:
            clusters.append(file_path)
            log_task(task_id, 'No vehicles detected. Will process full image.')
            
            annotated_img, detections, yolo_violations = pipeline_core.run_traffic_yolo(file_path)
            if detections:
                vlm_result = pipeline_core.get_vlm_verdict(file_path, detections, yolo_violations)
                record = pipeline_core.write_challan(
                    file_path, annotated_img, detections, vlm_result,
                    location=location, date=capture_date, time_str=capture_time
                )
                ext_name = f"extracted_{task_id}_full.jpg"
                ext_path = EXTRACTED_FOLDER / ext_name
                annotated_img.save(ext_path)
                import json
                with open(EXTRACTED_FOLDER / f"{ext_name}.json", "w") as f:
                    json.dump({
                        "vlm_caption": vlm_result.get("caption", ""),
                        "verdict": record.get("verdict", ""),
                        "violations": record.get("violations", []),
                        "location": location,
                        "date": capture_date,
                        "time": capture_time,
                    }, f)
        else:
            import PIL.Image
            orig_img = PIL.Image.open(file_path).convert("RGB")
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(orig_img.width, x2 + pad_x)
                cy2 = min(orig_img.height, y2 + pad_y)
                
                crop_img = orig_img.crop((cx1, cy1, cx2, cy2))
                ext_name = f"extracted_{task_id}_{idx}.jpg"
                ext_path = EXTRACTED_FOLDER / ext_name
                crop_img.save(ext_path)
                
                log_task(task_id, f"Running VLM/YOLO on vehicle {idx + 1}...")
                
                annotated_img, detections, yolo_violations = pipeline_core.run_traffic_yolo(ext_path)
                if detections:
                    vlm_result = pipeline_core.get_vlm_verdict(ext_path, detections, yolo_violations)
                    record = pipeline_core.write_challan(
                        ext_path, annotated_img, detections, vlm_result,
                        location=location, date=capture_date, time_str=capture_time
                    )
                    if record['verdict'] == 'VIOLATION':
                        log_task(task_id, f'VIOLATION ISSUED: {record.get("violations", [])}')
                        
                    import json
                    with open(EXTRACTED_FOLDER / f"{ext_name}.json", "w") as f:
                        ocr_data = record.get("ocr", {})
                        plate_text = ocr_data.get("plate_text", "")
                        plate_crop_path = ocr_data.get("plate_crop")
                        frontend_plate_crop = ""
                        
                        if plate_crop_path:
                            import shutil, os
                            if os.path.exists(plate_crop_path):
                                dest_plate = EXTRACTED_FOLDER / f"{ext_name}_plate.jpg"
                                shutil.copy2(plate_crop_path, dest_plate)
                                frontend_plate_crop = f"data/extracted/{dest_plate.name}"
                        
                        if not plate_text or plate_text == "[PLATE NOT DETECTED]" or plate_text == "[OCR FAILED]":
                            plate_text = "number not visible"

                        json.dump({
                            "vlm_caption": vlm_result.get("caption", ""),
                            "verdict": record.get("verdict", ""),
                            "violations": record.get("violations", []),
                            "ocr_plate": plate_text,
                            "ocr_crop": frontend_plate_crop,
                            "location": location,
                            "date": capture_date,
                            "time": capture_time,
                        }, f)
                
                clusters.append(ext_path)
        
        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['result'] = {'extracted_count': len(clusters)}
        
        log_task(task_id, 'Task finished successfully.')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_task(task_id, f'ERROR during processing: {str(e)}')
        processing_tasks[task_id]['progress'] = 0

def process_video_task(task_id, file_path, pipeline_core, video_core, vehicle_model):
    try:
        log_task(task_id, f'Opening video stream: {file_path}')
        processing_tasks[task_id]['progress'] = 5

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise FileNotFoundError('Cannot open video file. Ensure it is a valid MP4/AVI/MOV.')

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        frame_gap    = max(1, int(fps * video_core.MIN_INTERVAL_SEC))
        max_frames   = video_core.MAX_FRAMES
        blur_thresh  = video_core.HARD_DROP_BLUR_THRESHOLD

        log_task(task_id,
            f'Video loaded. FPS: {fps:.1f} | Frames: {total_frames} | '
            f'Duration: {duration_sec:.1f}s | Sampling every {video_core.MIN_INTERVAL_SEC}s.')

        idx = 0
        saved = 0
        skipped = 0
        prev_frame = None
        extracted_frames = []

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            score = video_core.laplacian_variance(frame)
            if score < blur_thresh:
                skipped += 1
                idx += frame_gap
                continue

            frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            enhanced, prev_frame, meta = video_core.auto_enhance(
                frame_resized, blur_score=score, prev_accumulated=prev_frame
            )

            frame_path = UPLOAD_FOLDER / f"frame_{task_id}_{saved}.jpg"
            cv2.imwrite(str(frame_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_frames.append(frame_path)

            saved += 1
            idx += frame_gap

            progress = min(40, 10 + int(30 * (idx / max(1, total_frames))))
            processing_tasks[task_id]['progress'] = progress

            if saved % 3 == 0:
                log_task(task_id, f'Extracted {saved} clean frames so far ({skipped} blurry skipped)...')

            if max_frames and saved >= max_frames:
                log_task(task_id, f'Reached frame cap ({max_frames}). Stopping extraction.')
                break

        cap.release()
        log_task(task_id,
            f'Extraction complete. Clean frames: {len(extracted_frames)} | Skipped (blurry): {skipped}.')

        if not extracted_frames:
            log_task(task_id, 'ERROR: No usable frames extracted from video. Try a different file.')
            processing_tasks[task_id]['progress'] = 0
            return

        log_task(task_id, f'Beginning ML detection on {len(extracted_frames)} frames...')

        all_records = []
        for i, frame_path in enumerate(extracted_frames):
            try:
                log_task(task_id, f'[Frame {i+1}/{len(extracted_frames)}] Running vehicle YOLO...')
                results = vehicle_model.predict(str(frame_path), conf=0.25, verbose=False)[0]
                
                boxes = results.boxes
                if boxes is None or len(boxes) == 0:
                    log_task(task_id, f'[Frame {i+1}] No vehicles detected, skipping.')
                    continue
                    
                import PIL.Image
                orig_img = PIL.Image.open(frame_path).convert("RGB")
                
                for v_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(orig_img.width, x2 + pad_x)
                    cy2 = min(orig_img.height, y2 + pad_y)
                    
                    crop_img = orig_img.crop((cx1, cy1, cx2, cy2))
                    ext_name = f"extracted_{task_id}_frame{i}_veh{v_idx}.jpg"
                    ext_path = UPLOAD_FOLDER / ext_name
                    crop_img.save(ext_path)
                    
                    log_task(task_id, f'[Frame {i+1} Veh {v_idx+1}] Running YOLO...')
                    annotated_img, detections, yolo_violations = pipeline_core.run_traffic_yolo(ext_path)

                    if not detections:
                        log_task(task_id, f'[Frame {i+1} Veh {v_idx+1}] No objects detected, skipping.')
                    else:
                        log_task(task_id,
                            f'[Frame {i+1} Veh {v_idx+1}] Found: {[d["class_name"] for d in detections]}. Running VLM/OCR...')
                        vlm_result = pipeline_core.get_vlm_verdict(ext_path, detections, yolo_violations)
                        record = pipeline_core.write_challan(ext_path, annotated_img, detections, vlm_result)

                        if record['verdict'] == 'VIOLATION':
                            log_task(task_id,
                                f'[Frame {i+1} Veh {v_idx+1}] VIOLATION ISSUED: {record.get("violations", [])}')
                            all_records.append(record)
                        else:
                            log_task(task_id, f'[Frame {i+1} Veh {v_idx+1}] CLEAN — no violations.')

            except Exception as ex:
                import traceback
                traceback.print_exc()
                log_task(task_id, f'[Frame {i+1}] Error: {ex}')

            prog = 40 + int(60 * ((i + 1) / len(extracted_frames)))
            processing_tasks[task_id]['progress'] = prog

        processing_tasks[task_id]['progress'] = 100
        processing_tasks[task_id]['records'] = all_records
        log_task(task_id,
            f'Task finished successfully. Generated {len(all_records)} challan(s) from video.')

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_task(task_id, f'ERROR during processing: {str(e)}')
        processing_tasks[task_id]['progress'] = 0

# ML worker disabled — app runs in challan-display-only mode

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = file.filename
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    task_id = str(int(time.time()))
    save_path = UPLOAD_FOLDER / f"{task_id}_{filename}"
    file.save(save_path)
    
    print(f"--- NEW UPLOAD: {filename} (Task ID: {task_id}) ---")

    if file_ext in ['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'webp']:
        processing_tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'status': 'Queued for processing.',
            'progress': 0,
            'logs': [f"[{time.strftime('%H:%M:%S')}] Received file {filename}. Queued for processing."]
        }
        task_queue.put((task_id, save_path, file_ext))
    else:
        processing_tasks[task_id] = {
            'id': task_id,
            'filename': filename,
            'status': f'ERROR: Unsupported file type: .{file_ext}',
            'progress': 0,
            'logs': [f"[{time.strftime('%H:%M:%S')}] ERROR: Unsupported file type '.{file_ext}'. Accepted: JPG, PNG, WEBP, MP4, AVI, MOV."]
        }

    return jsonify({'task_id': task_id, 'status': 'Started'})

@app.route('/api/status', methods=['GET'])
def get_status():
    from flask import Response
    import math
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        return obj
    import json
    safe = sanitize(list(processing_tasks.values()))
    return Response(json.dumps(safe), mimetype='application/json')

@app.route('/api/extracted', methods=['GET'])
def get_extracted():
    from flask import jsonify
    files = []
    if EXTRACTED_FOLDER.exists():
        for d in sorted(EXTRACTED_FOLDER.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_file() and d.suffix.lower() in ['.jpg', '.png']:
                if d.name.endswith('_plate.jpg'):
                    continue
                
                parts = d.stem.split('_')
                label = parts[-1] if len(parts) > 3 else "UNKNOWN"
                label = label.replace('-', ' ').upper()
                
                vlm_caption = "No VLM data"
                verdict = ""
                violations = []
                ocr_plate = "number not visible"
                ocr_crop = ""
                location = "Hyderabad (default,no geo tag in image)"
                date = ""
                time_str = ""
                json_path = EXTRACTED_FOLDER / f"{d.name}.json"
                if json_path.exists():
                    import json
                    try:
                        with open(json_path, "r") as f:
                            vlm_data = json.load(f)
                            vlm_caption = vlm_data.get("vlm_caption", "")
                            verdict = vlm_data.get("verdict", "")
                            violations = vlm_data.get("violations", [])
                            ocr_plate = vlm_data.get("ocr_plate", "number not visible")
                            ocr_crop = vlm_data.get("ocr_crop", "")
                            
                            loc_val = vlm_data.get("location", "")
                            if not loc_val or loc_val == "Not Found":
                                loc_val = "Hyderabad (default,no geo tag in image)"
                            location = loc_val
                            
                            date = vlm_data.get("date", "")
                            time_str = vlm_data.get("time", "")
                    except:
                        pass
                
                if violations:
                    label = " | ".join(violations).upper()
                
                files.append({
                    "path": f"data/extracted/{d.name}",
                    "label": label,
                    "vlm_caption": vlm_caption,
                    "verdict": verdict,
                    "violations": violations,
                    "ocr_plate": ocr_plate,
                    "ocr_crop": ocr_crop,
                    "location": location,
                    "date": date,
                    "time": time_str,
                })
    return jsonify(files)

@app.route('/api/seatbelt')
def get_seatbelt_cnn():
    files = []
    if EXTRACTED_FOLDER.exists():
        import seatbelt_cnn_core
        from PIL import Image
        for d in sorted(EXTRACTED_FOLDER.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_file() and d.suffix.lower() in ['.jpg', '.png']:
                if d.name.endswith('_plate.jpg'):
                    continue
                
                parts = d.stem.split('_')
                label = parts[-1] if len(parts) > 3 else "Vehicle"

                try:
                    import pipeline_core as _pc
                    pil_img = Image.open(d).convert('RGB')
                    annotated_img, detections, yolo_violations = _pc.run_traffic_yolo(str(d))
                    
                    # Find occupant box if YOLO found one
                    occupant_dets = [det for det in detections if det['class_name'] in ['seatbelt', 'no_seatbelt', 'rider']]
                    if occupant_dets:
                        best_occ = max(occupant_dets, key=lambda x: x['confidence'])
                        occ_crop = _pc.crop_plate(pil_img, best_occ['box']) # crop_plate works for any box
                    else:
                        occ_crop = pil_img # fallback to full vehicle crop
                        
                    worn, prob = seatbelt_cnn_core.classify_seatbelt(occ_crop)
                    
                    # ALWAYS Pass to OCR pipeline to see results
                    ocr_result = _pc.extract_plate_number(str(d), detections, EXTRACTED_FOLDER)
                    plate_text = ocr_result.get("plate_text", "number not visible")
                    plate_crop_path = ocr_result.get("plate_crop", "")
                    
                    ocr_crop_url = ""
                    if plate_crop_path:
                        import shutil
                        dest_plate = EXTRACTED_FOLDER / f"cnn_plate_{d.name}"
                        try:
                            shutil.copy2(plate_crop_path, dest_plate)
                            ocr_crop_url = f"data/extracted/{dest_plate.name}"
                        except:
                            pass
                        
                    verdict_str = "VIOLATION" if not worn else "CLEAN"
                    violation_list = ["No Seatbelt (CNN Detected)"] if not worn else ["Wearing Seatbelt (CNN Confirmed)"]
                        
                    files.append({
                        "path": f"data/extracted/{d.name}",
                        "label": label,
                        "prob_worn": prob,
                        "verdict": verdict_str,
                        "violations": violation_list,
                        "ocr_plate": plate_text,
                        "ocr_crop": ocr_crop_url
                    })
                except Exception as e:
                    print(f"Error processing {d.name} for seatbelt: {e}")
                    pass
    return jsonify(files)

@app.route('/api/challans', methods=['GET'])
def get_challans():
    import json, math
    from datetime import datetime
    
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        return obj
        
    challans = []
    if CHALLAN_FOLDER.exists():
        for d in sorted(CHALLAN_FOLDER.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if d.is_dir() and (d / 'challan.json').exists():
                try:
                    with open(d / 'challan.json', 'r') as f:
                        raw = f.read().replace('NaN', 'null').replace('Infinity', 'null')
                        data = json.loads(raw)
                        data = sanitize(data)
                        
                    # Use real date/time/location; show "Not Found" for date/time and Hyderabad for location if absent
                    if "date" not in data:
                        data["date"] = "Not Found"
                    if "time" not in data:
                        data["time"] = "Not Found"
                    if "location" not in data or not data["location"] or data["location"] == "Not Found":
                        data["location"] = "Hyderabad (default,no geo tag in image)"
                        
                    # Map true challan schema to what UI expects
                    data["filename"] = d.name  # Use folder name as the ID
                    data["image_url"] = data.get("files", {}).get("original", data.get("source_image", ""))
                    data["ocr_plate"] = data.get("registration_number", "")
                    data["vlm_caption"] = data.get("vlm_description", "")
                    data["ocr_crop"] = data.get("files", {}).get("plate_crop", "")
                    
                    challans.append(data)
                except Exception as e:
                    print(f"Error loading {d}: {e}")
    return jsonify(challans)

@app.route('/api/challans/<folder_name>', methods=['PUT', 'DELETE'])
def manage_challan(folder_name):
    import json, shutil
    challan_dir = CHALLAN_FOLDER / folder_name
    json_path = challan_dir / 'challan.json'

    if not challan_dir.exists():
        return jsonify({"error": "Challan not found"}), 404

    if request.method == 'DELETE':
        try:
            shutil.rmtree(challan_dir)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if request.method == 'PUT':
        try:
            updates = request.json
            with open(json_path, 'r') as f:
                raw = f.read().replace('NaN', 'null').replace('Infinity', 'null')
                data = json.loads(raw)

            if "ocr_plate" in updates:
                data["registration_number"] = updates["ocr_plate"]
            if "violations" in updates:
                data["violations"] = updates["violations"]
            if "verdict" in updates:
                data["verdict"] = updates["verdict"]

            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            return jsonify({"status": "success", "data": data})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/api/challans/<folder_name>/pdf')
def generate_challan_pdf(folder_name):
    """Generate a clean, official challan PDF using pandoc + xelatex."""
    import json, subprocess, tempfile, math, os
    from datetime import datetime

    challan_dir = CHALLAN_FOLDER / folder_name
    json_path = challan_dir / 'challan.json'

    if not json_path.exists():
        return jsonify({"error": "Challan not found"}), 404

    try:
        with open(json_path, 'r') as f:
            raw = f.read().replace('NaN', 'null').replace('Infinity', 'null')
            data = json.loads(raw)

        challan_id   = data.get("challan_id", folder_name)
        verdict      = data.get("verdict", "UNKNOWN")
        violations   = data.get("violations", [])
        fine_total   = data.get("fine_total", 0)
        plate        = data.get("registration_number", "Not Detected") or "Not Detected"
        if plate in ("[PLATE NOT DETECTED]", "[OCR FAILED]", ""):
            plate = "Not Detected"
        vlm_desc     = data.get("vlm_description", "") or ""
        evidence     = data.get("vlm_evidence", [])
        location     = data.get("location", "")
        if not location or location == "Not Found":
            location = "Hyderabad (default,no geo tag in image)"
        
        vehicle_type = data.get("vehicle_type", "Unknown") or "Unknown"
        if vehicle_type == "Unknown":
            if any(v in ["Riding without helmet", "Improperly worn helmet", "Triple Riding"] for v in violations):
                vehicle_type = "Two-Wheeler"
            elif any(v in ["Not wearing seatbelt"] for v in violations):
                vehicle_type = "Four-Wheeler"
                
        rec_date     = data.get("date", "Not Found") or "Not Found"
        rec_time     = data.get("time", "Not Found") or "Not Found"
        now_str      = datetime.now().strftime("%d %B %Y, %I:%M %p")

        def tex_escape(s):
            """Escape LaTeX special characters in dynamic string values."""
            if not s:
                return ""
            for ch, repl in [
                ('\\', r'\textbackslash{}'),
                ('&',  r'\&'),
                ('%',  r'\%'),
                ('$',  r'\$'),
                ('#',  r'\#'),
                ('^',  r'\textasciicircum{}'),
                ('_',  r'\_'),
                ('{',  r'\{'),
                ('}',  r'\}'),
                ('~',  r'\textasciitilde{}'),
            ]:
                s = s.replace(ch, repl)
            return s

        # Build violations table rows
        if violations:
            fine_map = {
                'Riding without helmet': 1000,
                'Improperly worn helmet': 500,
                'Triple Riding': 2000,
                'Not wearing seatbelt': 1000,
            }
            violation_rows = "\n".join(
                f"| {i+1} | {tex_escape(v)} | {fine_map.get(v, 500)} |"
                for i, v in enumerate(violations)
            )
        else:
            violation_rows = "| 1 | No violations detected | 0 |"

        # Build evidence list
        evidence_items = "\n".join(f"- {tex_escape(str(e))}" for e in evidence) if evidence else "- No additional evidence available"

        # Resolve original image
        original_path = None
        for name in ["original.jpg", "original.jpeg", "original.png"]:
            p = challan_dir / name
            if p.exists():
                original_path = p
                break
        if not original_path:
            candidates = sorted(challan_dir.glob("original*"))
            if candidates:
                original_path = candidates[0]

        img_line = ""
        if original_path and original_path.exists():
            img_line = f"![Evidence]({original_path.resolve()})" + "\n\n"

        # Plate crop
        plate_img_line = ""
        plate_crop_path = challan_dir / "plate_crop.jpg"
        if plate_crop_path.exists():
            plate_img_line = f"![Plate Crop]({plate_crop_path.resolve()})" + "\n\n"

        # Write a LaTeX header file (avoids YAML multi-line indentation issues)
        header_tex = r"""\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{colortbl}
\usepackage{mdframed}

\definecolor{govgray}{HTML}{EFEFEF}
\definecolor{accentblue}{HTML}{0B3D91}
\definecolor{violationred}{HTML}{CC0000}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{2pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{accentblue}\leaders\hrule height \headrulewidth\hfill}}
\fancyhead[L]{\textbf{\color{accentblue}GRIDLOCK} \small\ Traffic Enforcement System}
\fancyhead[R]{\small Challan: \texttt{""" + challan_id + r"""}}
\fancyfoot[L]{\small Government Traffic Enforcement System}
\fancyfoot[C]{\small Page \thepage\ of \pageref{LastPage}}
\fancyfoot[R]{\small System Generated}
"""

        tmp_header_fd, tmp_header_path = tempfile.mkstemp(suffix='.tex')
        with os.fdopen(tmp_header_fd, 'w') as f:
            f.write(header_tex)

        vlm_section = tex_escape(vlm_desc) if vlm_desc else "_No AI observations available._"

        markdown = f"""---
geometry: "margin=1.8cm"
fontsize: "8pt"
mainfont: "DejaVu Sans"
colorlinks: true
---

\\begin{{center}}
{{\\Large\\textbf{{\\color{{accentblue}}GOVERNMENT TRAFFIC ENFORCEMENT}}}}

{{\\large Electronic Challan — Gridlock AI System}}

\\vspace{{3pt}}
{{\\small Generated on {now_str}}}
\\end{{center}}

\\vspace{{4mm}}
\\noindent\\rule{{\\textwidth}}{{0.4pt}}
\\vspace{{2mm}}

\\rowcolors{{2}}{{white}}{{govgray}}

| **CHALLAN DETAILS** | |
|:--------------------|:---------------------------|
| Challan Number | `{challan_id}` |
| Date | {tex_escape(rec_date)} |
| Time | {tex_escape(rec_time)} |
| Generated On | {now_str} |
| Verdict | **{verdict}** |
| Total Fine | **INR {fine_total}** |
| Registration Number | **{tex_escape(plate)}** |
| Vehicle Type | {tex_escape(vehicle_type)} |
| Incident Location | {tex_escape(location)} |

\\vspace{{5mm}}

## Violations Detected

| Sl. | Offence | Fine (INR) |
|:---:|:------|----------:|
{violation_rows}

\\vspace{{4mm}}

{img_line}

## AI Observations

{vlm_section}

\\vspace{{3mm}}

## Evidence Chain

{evidence_items}

{plate_img_line}

\\vspace{{8mm}}

\\begin{{mdframed}}
\\textbf{{Declaration:}} This electronic challan has been generated automatically by the Gridlock AI
Traffic Enforcement System using image and video evidence captured by certified surveillance equipment.
The listed violations have been detected using automated analysis and are subject to verification under
the applicable provisions of the Motor Vehicles Act.
\\end{{mdframed}}

\\vfill

\\begin{{center}}
\\rule{{\\linewidth}}{{0.4pt}}

\\small This is a computer-generated document and does not require a physical signature. \\\\
\\textbf{{Gridlock AI Traffic Analyzer}} — Team GOPAKART
\\end{{center}}
"""

        tmp_fd, tmp_md_path = tempfile.mkstemp(suffix='.md')
        with os.fdopen(tmp_fd, 'w') as f:
            f.write(markdown)

        pdf_path = challan_dir.resolve() / f"challan_{challan_id}.pdf"

        result = subprocess.run(
            [
                'pandoc', os.path.abspath(tmp_md_path),
                '-o', str(pdf_path),
                '--pdf-engine=xelatex',
                '--standalone',
                f'--include-in-header={tmp_header_path}',
                f'--resource-path={challan_dir.resolve()}',
            ],
            capture_output=True, text=True, timeout=120,
            cwd=str(challan_dir.resolve()),
        )

        # Clean up temp files
        try:
            os.unlink(tmp_md_path)
            os.unlink(tmp_header_path)
        except Exception:
            pass

        if result.returncode != 0:
            print(f"Pandoc stderr:\n{result.stderr}")
            return jsonify({"error": "PDF generation failed", "details": result.stderr}), 500

        return send_file(
            str(pdf_path),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Challan_{challan_id}.pdf",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/add_challan')
def add_challan_page():
    return send_from_directory('.', 'add_challan.html')


@app.route('/api/add_challan', methods=['POST'])
def api_add_challan():
    """Manually create a challan and save it to the challans folder."""
    import json, uuid, shutil
    from datetime import datetime

    FINE_MAP = {
        'Illegal Parking (Overtime)': 500,
        'Red Line Crossing': 1000,
        'Stop Line Violation': 500,
        'No Helmet': 1000,
        'Triple Riding': 2000,
        'No Seatbelt': 1000,
    }

    try:
        violations_raw = request.form.get('violations', '[]')
        violations = json.loads(violations_raw)
        if not violations:
            return jsonify({'error': 'No violations provided'}), 400

        fine_total = sum(FINE_MAP.get(v, 500) for v in violations)
        registration_number = request.form.get('registration_number', 'Not Provided').strip()
        vehicle_type = request.form.get('vehicle_type', 'Unknown')
        date = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        time_str = request.form.get('time', datetime.now().strftime('%H:%M'))
        location = request.form.get('location', 'Unknown Location').strip()
        notes = request.form.get('notes', '').strip()

        # Generate unique challan ID and folder
        challan_id = uuid.uuid4().hex[:8].upper()
        folder_name = f"challan_{challan_id}_MANUAL"
        challan_dir = CHALLAN_FOLDER / folder_name
        challan_dir.mkdir(parents=True, exist_ok=True)

        # Handle optional image upload
        original_path = None
        image_file = request.files.get('image')
        if image_file and image_file.filename:
            ext = Path(image_file.filename).suffix.lower() or '.jpg'
            original_path = challan_dir / f"original{ext}"
            image_file.save(str(original_path))

        files = {
            'original': str(original_path.relative_to(Path('.'))) if original_path else None,
            'yolo_marked': None,
            'challan_md': None,
            'plate_crop': None,
            'plate_ocr': None,
        }

        vlm_evidence = [f"Manually issued by enforcement officer"]
        if notes:
            vlm_evidence.append(f"Officer notes: {notes}")

        challan_data = {
            'challan_id': challan_id,
            'challan_dir': str(challan_dir),
            'source_image': files['original'],
            'verdict': 'VIOLATION',
            'entry_type': 'MANUAL',
            'violations': violations,
            'fine_total': fine_total,
            'registration_number': registration_number,
            'vehicle_type': vehicle_type,
            'date': date,
            'time': time_str,
            'location': location,
            'vlm_description': notes or '',
            'vlm_evidence': vlm_evidence,
            'detections': [],
            'ocr': {
                'plate_text': registration_number,
                'ocr_confidence': None,
                'char_count': len(registration_number),
                'plate_crop': None,
                'plate_annotated': None,
                'status': 'manual_entry',
            },
            'no_helmet_crops': [],
            'files': files,
        }

        json_path = challan_dir / 'challan.json'
        with open(json_path, 'w') as f:
            json.dump(challan_data, f, indent=2)

        return jsonify({'status': 'success', 'challan_id': challan_id, 'folder': folder_name})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/<path:filename>')
def serve_root(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("Starting GRIDLOCK Engine Flask Server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
