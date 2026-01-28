#this is the script for detection to run on opencv mac camera

from ultralytics import YOLO
import cv2
import os

# Get absolute path to model
script_dir = os.path.dirname(os.path.abspath(__file__))  # /Users/.../src/
project_dir = os.path.dirname(script_dir)                # /Users/.../PPE-Compliance-Gate-System/
model_path = os.path.join(project_dir, 'models', 'best.pt')

print(f"📁 Project directory: {project_dir}")
print(f"🔍 Looking for model at: {model_path}")

# Check if model exists
if not os.path.exists(model_path):
    print(f"❌ ERROR: Model not found at {model_path}")
    exit(1)

print(f"✅ Loading model...")
model = YOLO(model_path)

# Classes you care about
PPE_CLASSES = {
    'Hardhat': 3,
    'Safety Vest': 13,
    'Gloves': 1,
    'Goggles': 2,
    'NO-Hardhat': 8,
    'NO-Safety Vest': 10,
    'NO-Gloves': 6,
    'NO-Goggles': 7,
}

# Open webcam
print("🎥 Starting PPE Detection...")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ ERROR: Could not open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    # Run detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Display frame
    cv2.imshow('PPE Detection - Press Q to quit', annotated_frame)
    
    # Check for violations (optional)
    detections = results[0].boxes
    for box in detections:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Alert on violations
        if cls in [6, 7, 8, 10]:  # NO-Gloves, NO-Goggles, NO-Hardhat, NO-Safety Vest
            print(f"⚠️ VIOLATION: {model.names[cls]} detected (confidence: {conf:.2f})")
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection stopped")