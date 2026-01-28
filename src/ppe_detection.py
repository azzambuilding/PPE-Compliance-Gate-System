#this is the script for detection to run on opencv mac camera

from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('best.pt')  # Make sure best.pt is in same folder

# Classes you care about (adjust based on your needs)
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
cap = cv2.VideoCapture(0)  # 0 = default camera

print("🎥 Starting PPE Detection...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
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