#!/usr/bin/env python3
"""
Simplified Flask app for testing - starts faster without heavy AI models
"""

import os
import cv2
import json
import time
import glob
import hashlib
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# DB initialization
def init_db():
    conn = sqlite3.connect('ocr_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS processed_images
                 (filename TEXT PRIMARY KEY,
                  file_hash TEXT,
                  text_data TEXT,
                  boxes TEXT,
                  words TEXT,
                  processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
init_db()

# Simple file hash function
def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Simple image processing
def process_image_simple(filepath):
    """Simple image processing without heavy AI models"""
    try:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Cannot read image")
        
        # Basic image info
        height, width = img.shape[:2]
        
        # Simple text extraction simulation (replace with actual OCR later)
        text_data = f"Image processed: {os.path.basename(filepath)}"
        words = ["Sample", "text", "extraction"]
        boxes = [[[0, 0], [100, 0], [100, 50], [0, 50]]]
        
        return {
            'text': text_data,
            'boxes': boxes,
            'words': words,
            'image_info': {'width': width, 'height': height}
        }
    except Exception as e:
        return {
            'text': f"Error processing image: {str(e)}",
            'boxes': [],
            'words': [],
            'image_info': {'width': 0, 'height': 0}
        }

# Simple PDF generation
def generate_simple_pdf(case_name, results, output_pdf_path):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    story.append(Paragraph(f"<b>Simple Case Report</b>", styles['Title']))
    story.append(Paragraph(f"Case: {case_name}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    for entry in results:
        filename = entry['filename']
        story.append(Paragraph(f"<b>File:</b> {filename}", styles['Heading4']))
        story.append(Paragraph(f"<b>Status:</b> Processed", styles['Normal']))
        story.append(Spacer(1, 10))
    
    doc.build(story)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

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
    
    try:
        result = process_image_simple(filepath)
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'filename': filename,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_case', methods=['POST'])
def api_process_case():
    data = request.get_json() or {}
    case_name = data.get('case_name', f"case_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    keywords = data.get('keywords', [])
    generate_pdf = bool(data.get('generate_pdf', True))
    
    try:
        # Get all image files
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
            image_files.extend(glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], ext)))
        
        results = []
        for img_path in image_files:
            entry = {'filename': os.path.basename(img_path)}
            result = process_image_simple(img_path)
            entry.update(result)
            results.append(entry)
        
        # Generate PDF if requested
        pdf_path = None
        if generate_pdf and results:
            pdf_name = f"{case_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_name)
            generate_simple_pdf(case_name, results, pdf_path)
        
        return jsonify({
            'message': 'Case processed successfully',
            'pdf': os.path.basename(pdf_path) if pdf_path else None,
            'results_count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'message': 'Simple OCR app is running',
        'uploads_folder': app.config['UPLOAD_FOLDER'],
        'outputs_folder': app.config['OUTPUT_FOLDER']
    })

if __name__ == '__main__':
    print("🚀 Starting Simple OCR & Object Detection System...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
