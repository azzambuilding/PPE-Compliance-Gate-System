from ultralytics import YOLO
import cv2
import os
from confidence_helper import check_hardhat_violation, check_safety_vest_violation
import time
import statistics

#next impliment person tracker with ID

# Get absolute path to model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_path = os.path.join(project_dir, 'models', 'best.pt')

print(f"Loading model...")
model = YOLO(model_path)

# PPE classes only (no mask, ladder, cone, fall)
"""List of classes
'Fall-Detected',    #0
    'Gloves',           # 1
    'Goggles',          # 2
    'Hardhat',          # 3
    'Ladder',           # 4
    'Mask',             # 5
    'NO-Gloves',        # 6
    'NO-Goggles',       # 7
    'NO-Hardhat',       # 8
    'NO-Mask',          # 9
    'NO-Safety Vest',   # 10
    'Person',           # 11
    'Safety Cone',      # 12
    'Safety Vest'      #13
 """
PPE_CLASS_IDS = [1, 2, 3, 6, 7, 8, 10, 11, 13] 
#PPE_CLASS_IDS = [3,8]
VIOLATION_IDS = [6, 7, 8, 10]
#VIOLATION_IDS = [8]

print(f"\nDetecting these classes:")
for class_id in PPE_CLASS_IDS:
    print(f"  {class_id}: {model.names[class_id]}")

print("\nStarting PPE Detection...")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

#conf_dict, last_print_time, PRINT_INTERVAL are used for block below
conf_dict = {} #creates a dictionary of conf scores per class
last_print_time = time.time()
PRINT_INTERVAL = 2 #measured in seconds
hardhat_conf_scores = []
vest_conf_scores = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect all classes
    results = model.predict(
        frame, 
        conf=0.03,           # Lower confidence threshold
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

    current_time = time.time()  # compute once, reused by all throttle checks below

     # Check for hardhat violations using the imported function
    hardhat_result = check_hardhat_violation(filtered_boxes)

     # If hardhat violation detected via decision tree, accumulate confidence scores
    if hardhat_result['violation']:
        hardhat_conf_scores.append(hardhat_result['confidence'])

    # Check for safety-vest violations using the imported function
    safety_vest_result = check_safety_vest_violation(filtered_boxes)

     # If safety vest violation detected via decision tree, accumulate confidence scores
    if safety_vest_result['violation']:
        vest_conf_scores.append(safety_vest_result['confidence'])

    # Check for violations
    for box in filtered_boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 8 or cls == 10:  # NO-Hardhat class
            continue

        if cls in VIOLATION_IDS:
            # Accumulate conf scores per class
            if cls not in conf_dict:
                conf_dict[cls] = []
            conf_dict[cls].append(conf)

    # Non-blocking timed print
    if current_time - last_print_time >= PRINT_INTERVAL:
        if hardhat_conf_scores:
            median = statistics.median(hardhat_conf_scores)
            print(f"HARDHAT VIOLATION: {hardhat_result['source']} (median confidence: {median:.2f}) - {hardhat_result['reason']}")
            hardhat_conf_scores.clear()

        if vest_conf_scores:
            median = statistics.median(vest_conf_scores)
            print(f"SAFETY VEST VIOLATION: {safety_vest_result['source']} (median confidence: {median:.2f}) - {safety_vest_result['reason']}")
            vest_conf_scores.clear()

        for cls, scores in conf_dict.items():
            if scores:
                median = statistics.median(scores)
                print(f"VIOLATION: {model.names[cls]} detected (median confidence: {median:.2f})")
        
        conf_dict.clear()  # reset for next interval
        last_print_time = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n Detection stopped")