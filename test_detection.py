"""
Quick test script to verify YOLO detection is working
"""

import os
from yolo_detector import YOLODetector

def test_yolo():
    print("🔍 Testing YOLO Detection")
    print("="*50)
    
    # Initialize detector
    print("\n1️⃣ Initializing YOLO detector...")
    detector = YOLODetector(conf_threshold=0.15)  # Lower threshold for testing
    print("✅ YOLO initialized")
    
    # Check for test images
    print("\n2️⃣ Looking for test images...")
    test_files = []
    if os.path.exists("uploads"):
        for f in os.listdir("uploads"):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_files.append(f"uploads/{f}")
    
    if not test_files:
        print("⚠️ No images found in 'uploads' folder")
        print("💡 Please upload some images first")
        return
    
    print(f"✅ Found {len(test_files)} images")
    
    # Test detection
    for img_path in test_files[:3]:  # Test first 3 images
        print(f"\n3️⃣ Testing: {img_path}")
        try:
            detections = detector.detect(img_path)
            
            if detections:
                print(f"✅ Success! Found {len(detections)} objects:")
                for i, det in enumerate(detections[:5], 1):  # Show first 5
                    print(f"   {i}. {det['label']} (confidence: {det['score']:.3f})")
            else:
                print("⚠️ No objects detected")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("✅ Test completed")

if __name__ == "__main__":
    test_yolo()

