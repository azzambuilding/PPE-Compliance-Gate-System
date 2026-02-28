# PPE-Compliance-Gate-System

Overview: A computer vision checkpoint system that verifies workers wear required safety equipment before entering construction sites. Uses YOLOv8 object detection to identify hard hats, safety vests, gloves, and safety glasses in real-time video feeds at site entrances. When a worker approaches the gate, the system scans for all required PPE items within 2 seconds, displays pass/fail status on a screen, and logs entry attempts with timestamped photos. Non-compliant workers receive immediate visual/audio feedback and cannot proceed until equipped properly. Dashboard shows live compliance rates, frequent violators, and generates daily reports for safety managers with photo evidence for OSHA documentation.

## Model Training

The YOLOv8 model was trained on Google Colab over 30 epochs on a custom PPE dataset.
You can view, run, or retrain the model using the notebook below:

https://colab.research.google.com/drive/1DQsrHNaIBA2Bs_ZDgDnLB8p96xrKXAay?usp=sharing

> **Note:** Make sure to save a copy to your own Drive before making changes (`File -> Save a copy in Drive`).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/azzambuilding/PPE-Compliance-Gate-System.git
cd PPE-Compliance-Gate-System
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt

### 4. Change directory
```bash
cd src
#
then run python3 ppe_detection.py


###file formatting

/models - trained YOLOv8 weights
    -best.pt - best model achieved from 30 epochs of training

/src - main detection scripts
    -ppe_detection.py YOLO
    -database.py PostgreSQL
/utils - helper functions
-requirements.txt
-README.md
-.env this is where your postgreSQL password/username/port ect private information go
