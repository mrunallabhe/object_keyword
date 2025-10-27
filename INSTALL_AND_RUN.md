# 📋 Install and Run Guide

## Simple Installation (3 Commands)

```bash
# 1. Install basic dependencies
pip install -r requirements.txt

# 2. Install CLIP (for enhanced features - optional but recommended)
pip install git+https://github.com/openai/CLIP.git

# 3. Run the app
python app_enhanced.py
```

## What Works Without CLIP

The system works with **YOLO only** (no CLIP needed):

```bash
python app_enhanced.py
```

**Features available:**
- ✅ Upload images
- ✅ Auto-detect ALL objects  
- ✅ List all objects
- ✅ Search by keyword (basic matching)
- ✅ Display matching images

**What you'll miss without CLIP:**
- Semantic similarity matching
- Multi-scale CLIP detection
- Context-aware search (e.g., "person wearing red")

## Quick Start

### Step 1: Install Dependencies
```bash
pip install Flask opencv-python-headless Pillow numpy ultralytics easyocr reportlab
```

### Step 2: Run
```bash
python app_enhanced.py
```

### Step 3: Open Browser
```
http://localhost:5000
```

## Troubleshooting

### Option A: Install CLIP (Full Features)
```bash
pip install git+https://github.com/openai/CLIP.git
python app_enhanced.py
```

### Option B: Use Without CLIP (Basic Features)
```bash
# Just run it - works without CLIP!
python app_enhanced.py
```

The system will detect that CLIP is not available and work with YOLO only.

### Option C: Use Simple YOLO System
```bash
python app_yolo.py
```

This is the original system with fewer features but guaranteed to work.

## What You Get

### With app_enhanced.py (YOLO + Optional CLIP)

**3 Tabs:**

1. **Upload Tab**
   - Upload images
   - See detected objects automatically

2. **List Tab**
   - View all images and their objects
   - See unique object list

3. **Search Tab**  
   - Enter keyword
   - See matching images

### With app_yolo.py (Original)

**Simple Interface:**
- Upload images
- Enter keywords to detect
- See results

## Recommended Setup

```bash
# Install everything
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Run enhanced system
python app_enhanced.py
```

This gives you:
- ✅ Full object detection
- ✅ Semantic search
- ✅ Multi-scale detection
- ✅ Best accuracy

## No Git? No CLIP?

If you can't install CLIP from git, the system still works!

Just run:
```bash
python app_enhanced.py
```

It will use YOLO only and work perfectly for basic object detection and search.

## Summary

**Simplest way to start:**
```bash
pip install -r requirements.txt
python app_enhanced.py
```

**Then go to:** http://localhost:5000

The system works with or without CLIP! 🎉

