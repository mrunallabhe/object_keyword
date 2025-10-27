# 🚀 Quick Start Guide

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Installation
```bash
python test_installation.py
```

### 3. Start the Application

**Option A: Using the startup script (Recommended)**
```bash
python run_app.py
```

**Option B: Direct execution**
```bash
python app_integrated.py
```

**Option C: Windows batch file**
```bash
start_app.bat
```

### 4. Access the Web Interface
Open your browser and go to: **http://localhost:5000**

## 🎯 Quick Usage

### Upload Images
1. Click the upload area or drag & drop images
2. Supported formats: JPG, PNG, BMP, TIFF
3. Images are automatically processed

### Process a Case
1. Enter case name (e.g., "Case_001")
2. Add keywords (e.g., "gun, knife, license plate")
3. Click "Process Case"
4. Download the generated PDF report

## 📁 Project Files

- `app_integrated.py` - Main Flask application
- `templates/index.html` - Web interface
- `requirements.txt` - Python dependencies
- `run_app.py` - Startup script
- `test_installation.py` - Installation test
- `config.py` - Configuration settings
- `start_app.bat` - Windows batch file
- `README.md` - Full documentation

## 🔧 Troubleshooting

### Common Issues
1. **Missing packages**: Run `pip install -r requirements.txt`
2. **CUDA errors**: The app will automatically fall back to CPU
3. **Permission errors**: Ensure write access to uploads/ and outputs/ folders

### Test Your Installation
```bash
python test_installation.py
```

## 📞 Support
- Check `README.md` for detailed documentation
- Review error messages in the console
- Ensure all dependencies are installed

---
**Ready to go!** 🎉
