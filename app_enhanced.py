"""
Enhanced Object Detection System
- Upload image → Detect ALL objects → List them
- Enter keyword → Filter images containing that keyword → Display results
"""

import os
import cv2
import json
import time
import glob
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Import detectors
from yolo_detector import YOLODetector
from weapon_detector import EnhancedWeaponDetector

# Try to import enhanced CLIP
try:
    from enhanced_clip_detector import EnhancedCLIPDetector
    ENHANCED_CLIP_AVAILABLE = True
except:
    ENHANCED_CLIP_AVAILABLE = False
    print("⚠️ Enhanced CLIP not available - using YOLO only")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize detectors
print("🔍 Initializing Detection Systems...")
yolo_detector = EnhancedWeaponDetector(conf_threshold=0.15, nms_threshold=0.4)
print("✅ Enhanced Weapon Detector ready!")

clip_detector = None
CLIP_AVAILABLE = False

if ENHANCED_CLIP_AVAILABLE:
    try:
        clip_detector = EnhancedCLIPDetector(model_name="ViT-B/32")
        print("✅ Enhanced CLIP ready!")
        CLIP_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ CLIP initialization failed: {e}")
        CLIP_AVAILABLE = False
else:
    print("⚠️ Enhanced CLIP not available - install: pip install git+https://github.com/openai/CLIP.git")

# Storage for detected objects in each image
detected_objects_cache = {}

def detect_all_objects_in_image(image_path):
    """
    Detect ALL objects in an image and return them as a list
    """
    try:
        # Use enhanced weapon detector
        detections = yolo_detector.detect_enhanced(image_path)
        
        # Extract unique object labels
        object_types = list(set([det['label'] for det in detections]))
        
        # Add weapon-related tags if weapons detected
        has_weapon = any(d.get('type') == 'gun_like_object' for d in detections)
        if has_weapon and 'weapon' not in object_types:
            object_types.append('weapon')
        
        return {
            'objects': object_types,
            'detections': detections,
            'count': len(detections),
            'has_weapons': has_weapon
        }
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return {'objects': [], 'detections': [], 'count': 0}

def search_images_by_keyword(keyword):
    """
    Search all uploaded images for a keyword and return matching images
    """
    results = []
    
    # Scan all images in uploads folder
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        image_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], ext)))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        # Get detected objects for this image (from cache or detect)
        if filename in detected_objects_cache:
            detected_objects = detected_objects_cache[filename]['objects']
        else:
            result = detect_all_objects_in_image(img_path)
            detected_objects = result['objects']
            detected_objects_cache[filename] = result
        
        # Check if keyword matches
        keyword_lower = keyword.lower().strip()
        
        # Direct match
        if any(keyword_lower in obj.lower() for obj in detected_objects):
            results.append({
                'filename': filename,
                'objects': detected_objects,
                'matched': True,
                'match_type': 'direct'
            })
        # Use CLIP for semantic matching if available
        elif CLIP_AVAILABLE:
            try:
                # Get CLIP similarity for the keyword
                clip_results = clip_detector.detect_all_objects_in_image(img_path, [keyword])
                similarity = clip_results.get(keyword, 0)
                
                if similarity > 0.25:  # Threshold for semantic match
                    results.append({
                        'filename': filename,
                        'objects': detected_objects,
                        'matched': True,
                        'match_type': 'semantic',
                        'similarity': similarity
                    })
            except:
                pass
    
    return results

@app.route('/')
def index():
    return render_template('enhanced_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Detect all objects in the uploaded image
    result = detect_all_objects_in_image(filepath)
    detected_objects_cache[filename] = result
    
    return jsonify({
        'message': 'File uploaded successfully',
        'filename': filename,
        'objects_detected': result['objects'],
        'object_count': result['count']
    })

@app.route('/list_all_objects', methods=['GET'])
def list_all_objects():
    """
    List all objects detected in all uploaded images
    """
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
        image_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], ext)))
    
    all_objects = {}
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        if filename in detected_objects_cache:
            result = detected_objects_cache[filename]
        else:
            result = detect_all_objects_in_image(img_path)
            detected_objects_cache[filename] = result
        
        all_objects[filename] = {
            'objects': result['objects'],
            'count': result['count']
        }
    
    # Get unique list of all objects
    unique_objects = set()
    for data in all_objects.values():
        unique_objects.update(data['objects'])
    
    return jsonify({
        'images': all_objects,
        'all_objects': sorted(list(unique_objects)),
        'total_images': len(all_objects)
    })

@app.route('/search_by_keyword', methods=['POST'])
def search_by_keyword():
    """
    Search for images containing a specific keyword
    """
    data = request.get_json() or {}
    keyword = data.get('keyword', '').strip()
    
    if not keyword:
        return jsonify({'error': 'Keyword required'}), 400
    
    results = search_images_by_keyword(keyword)
    
    return jsonify({
        'keyword': keyword,
        'matches': len(results),
        'images': results
    })

@app.route('/get_image_info/<filename>', methods=['GET'])
def get_image_info(filename):
    """
    Get detailed info about an image including all detected objects
    """
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # Get detection info
    if filename in detected_objects_cache:
        result = detected_objects_cache[filename]
    else:
        result = detect_all_objects_in_image(image_path)
        detected_objects_cache[filename] = result
    
    return jsonify({
        'filename': filename,
        'objects': result['objects'],
        'detections': result['detections'],
        'count': result['count']
    })

@app.route('/enhanced_detect/<filename>', methods=['POST'])
def enhanced_detect(filename):
    """
    Enhanced detection for a specific image using CLIP
    """
    data = request.get_json() or {}
    keywords = data.get('keywords', [])
    
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(',')]
    
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    all_detections = []
    
    # Use enhanced CLIP for semantic detection
    if CLIP_AVAILABLE:
        for keyword in keywords:
            try:
                detections = clip_detector.detect_text_guided(
                    image_path, 
                    keyword,
                    score_threshold=0.22
                )
                
                for det in detections:
                    det['keyword'] = keyword
                    all_detections.append(det)
                
                # Annotate image
                if detections:
                    output_path = os.path.join(
                        app.config['OUTPUT_FOLDER'], 
                        f"enhanced_{keyword}_{filename}"
                    )
                    clip_detector.annotate_image(image_path, detections, output_path, keyword)
                    
            except Exception as e:
                print(f"Error in enhanced detect: {e}")
    
    return jsonify({
        'filename': filename,
        'keywords': keywords,
        'detections': all_detections,
        'count': len(all_detections)
    })

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Enhanced Object Detection System")
    print("="*60)
    print("Features:")
    print("  ✅ Upload images → Detect ALL objects")
    print("  ✅ List all detected objects")
    print("  ✅ Search by keyword → Filter matching images")
    print("  ✅ Enhanced CLIP semantic detection")
    print("="*60)
    print("📱 Open: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

