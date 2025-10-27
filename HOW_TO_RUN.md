# 🚀 How to Run YOLO Object Detection System

## Step-by-Step Instructions

### Step 1: Install Python Dependencies

Open Command Prompt (Windows) or Terminal (Linux/Mac) in the project folder:

```bash
cd C:\Users\HP\OneDrive\Desktop\VII_Sem

# Install all required packages
pip install -r requirements.txt
```

**What this installs:**
- Flask (web framework)
- OpenCV (image processing)
- PyTorch (deep learning)
- Ultralytics (YOLO)
- EasyOCR (text extraction)

### Step 2: Run the Application

**Option A: Windows (Easiest)**
```bash
# Double-click the batch file
start_yolo.bat

# OR run from command prompt
python app_yolo.py
```

**Option B: Command Line**
```bash
python app_yolo.py
```

**Option C: Use the simple version (faster startup)**
```bash
python app_simple.py
```

### Step 3: Open Your Browser

Once the server starts, you'll see:
```
🔍 YOLO Object Detection System
===============================================
📱 Open your browser and go to: http://localhost:5000
⏹️  Press Ctrl+C to stop the server
```

**Open your web browser and go to:** `http://localhost:5000`

## Using the Web Interface

### 1️⃣ Upload Images

- **Click** on the upload area or **drag & drop** images
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- You can upload multiple images

### 2️⃣ Fill Out Case Information

Enter the following details:
- **Case Name**: e.g., "Crime Scene 001" or "Case_ABC"
- **Investigator ID** (optional): Your investigator ID
- **Description** (optional): Brief case description
- **Keywords**: What objects to detect (separate by commas)
  - Examples: `person, gun, knife, vehicle, bag`
- **☑ Generate PDF**: Check this to create a report

### 3️⃣ Process the Case

Click **"🚀 Process Case"** button

The system will:
- ✅ Detect objects using YOLO (80 object classes)
- ✅ Extract text using OCR
- ✅ Generate annotated images
- ✅ Create PDF report (if selected)

### 4️⃣ Download Results

Once processing is complete:
- Download the PDF report
- View annotated images
- See detection results

## Example Usage

### Example 1: Detect Weapons
```
Keywords: person, gun, knife, weapon, bag
```

### Example 2: Detect Vehicles and People
```
Keywords: person, vehicle, car, motorcycle, bicycle
```

### Example 3: Detect Evidence Items
```
Keywords: bottle, bag, backpack, phone, laptop, document
```

## Command Line Alternative

You can also test via command line:

```bash
# Test YOLO detection
python yolo_detector.py

# Start Flask server
python app_yolo.py
```

## What Objects Can YOLO Detect?

**80 Object Classes** including:

👤 **People & Animals**: person, bird, cat, dog, horse, cow, sheep, elephant

🚗 **Vehicles**: car, truck, motorcycle, bicycle, bus, train, boat, airplane

🔪 **Weapons & Tools**: knife, scissors, baseball bat, tennis racket

💻 **Electronics**: tv, laptop, mouse, keyboard, cell phone, remote

🪑 **Furniture**: chair, couch, bed, dining table, toilet

🎒 **Containers**: bottle, wine glass, cup, bowl, backpack, handbag, suitcase

**And 50+ more classes!**

## Troubleshooting

### ❌ Error: "Module 'ultralytics' not found"

**Fix:**
```bash
pip install ultralytics
```

### ❌ Error: "Out of memory"

**Fix:**
Edit `app_yolo.py` line 40, change to:
```python
DEVICE = "cpu"  # Use CPU instead of GPU
```

### ❌ Slow Processing

**Solutions:**
1. Reduce image sizes (resize before uploading)
2. Process fewer images at once
3. Use `app_simple.py` for faster startup

### ❌ Port 5000 Already in Use

**Fix:**
Change port in `app_yolo.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change to 5001
```

## Quick Test

To quickly test if everything works:

```bash
# 1. Run the application
python app_yolo.py

# 2. In another terminal, test the API
curl http://localhost:5000/status

# Should return: {"status": "running"}
```

## Performance

- **On GPU**: 50-60 FPS (very fast!)
- **On CPU**: 5-10 FPS (still good)
- **Memory**: 2-4 GB RAM needed

## Next Steps

1. ✅ Run: `python app_yolo.py`
2. ✅ Open: http://localhost:5000
3. ✅ Upload test images
4. ✅ Enter keywords
5. ✅ Process case
6. ✅ Download report

## Need Help?

- Read: `QUICK_START_YOLO.md` for quick start
- Read: `YOLO_IMPLEMENTATION_GUIDE.md` for technical details
- Read: `INSTALL_YOLO.md` for detailed installation

---

**That's it! Your YOLO object detection system is ready to analyze crime scene evidence!** 🎉

