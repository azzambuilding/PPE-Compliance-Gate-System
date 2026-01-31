from ultralytics import YOLO
import cv2
import os

# Get absolute path to model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_path = os.path.join(project_dir, 'models', 'best.pt')

print(f"Loading model...")
model = YOLO(model_path)

# PPE classes only (no mask, ladder, cone, fall)
PPE_CLASS_IDS = [1, 2, 3, 6, 7, 8, 10, 11, 13]
VIOLATION_IDS = [6, 7, 8, 10]

print(f"\n📋 Detecting these classes:")
for class_id in PPE_CLASS_IDS:
    print(f"  {class_id}: {model.names[class_id]}")

print("\nStarting PPE Detection...")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect all classes
    results = model.predict(
        frame, 
        conf=0.1,           # Lower confidence threshold
        iou=0.9,            # Very high IOU = allows lots of overlap
        max_det=10,         # Allow more detections per image
        agnostic_nms=False, # Allow multiple classes at same location
        verbose=False,
        #visualize=True #debugging/visualizes model
    )
    
    # Manually filter to only PPE classes
    all_boxes = results[0].boxes
    filtered_boxes = []
    
    for box in all_boxes:
        cls = int(box.cls[0])
        if cls in PPE_CLASS_IDS:  # Only keep PPE classes
            filtered_boxes.append(box)
    
    # Replace results with filtered boxes
    results[0].boxes = filtered_boxes
    
    # Draw only filtered boxes
    annotated_frame = results[0].plot()
    
    cv2.imshow('PPE Detection - Press Q to quit', annotated_frame)
    
    # Check for violations
    for box in filtered_boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls in VIOLATION_IDS:
            print(f"VIOLATION: {model.names[cls]} detected (confidence: {conf:.2f})")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n Detection stopped")