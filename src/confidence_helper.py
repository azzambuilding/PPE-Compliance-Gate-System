"""training overview: 
Dataset: 8,814 images across 15 classes
Epochs: 30
Full parameters and training configuration documented in the .ipynb notebook.
"""

"""data = [
    # Class                Images  Instances  Precision  Recall   mAP50   mAP50-95
    ['all',                 8814,   22077,     0.706,     0.828,   0.781,  0.514],
    ['Fall-Detected',       899,    899,       0.731,     0.858,   0.862,  0.577],
    ['Gloves',              395,    858,       0.827,     0.934,   0.95,   0.506],
    ['Goggles',             746,    827,       0.833,     0.982,   0.963,  0.604],
    ['Hardhat',             3191,   8952,      0.812,     0.906,   0.905,  0.538],
    ['Ladder',              193,    202,       0.878,     0.931,   0.947,  0.812],
    ['Mask',                292,    554,       0.48,      0.92,    0.514,  0.413],
    ['NO-Gloves',           571,    1258,      0.791,     0.88,    0.906,  0.446],
    ['NO-Goggles',          679,    859,       0.839,     0.942,   0.953,  0.576],
    ['NO-Hardhat',          865,    2222,      0.595,     0.881,   0.746,  0.502],
    ['NO-Mask',             327,    505,       0.569,     0.857,   0.653,  0.462],
    ['NO-Safety Vest',      189,    361,       0.287,     0.23,    0.22,   0.112],
    ['Person',              193,    277,       0.917,     0.91,    0.93,   0.77],
    ['Safety Cone',         338,    3016,      0.716,     0.687,   0.702,  0.386],
    ['Safety Vest',         609,    1287,      0.61,      0.672,   0.682,  0.491]
]
"""
"""CRITICAL ISSUE: NO-HARDHAT DETECTION ACCURACY
----------------------------------------------
I identified the problem while real world testing. During validation with live camera feed, 
a significant accuracy discrepancy was discovered between the Hardhat detector and the NO-Hardhat detector:

  Hardhat Detection:    mAP@50 = 0.905 (90.5% accurate)
  NO-Hardhat Detection: mAP@50 = 0.746 (74.6% accurate)

Key Finding:
  The inverse of the Hardhat confidence score (1 - hardhat_confidence) provides MORE 
  accurate NO-Hardhat predictions than the direct NO-Hardhat detector.
  
  However, completely ignoring the NO-Hardhat detector would discard valuable information
  when it shows high confidence scores.


Solution: Create reliability weighted decision tree logic

Use the NO-Hardhat detector only when it's confident enough (after 21% boost) 
to beat the inverse Hardhat calculation; otherwise, trust the more reliable inverse logic.
"""


#Hardhat mAP@50 = 0.905 : No-Hardhat mAP@50 = 0.746
#0.905 / 0.746 = 1.213 hardcoded a reliability multiplier to equal out the reliability of both when comparing
reliability_multiplier = 1.213 
#Safety vest mAP@50 mAP@50 = 0.682 : No-safety vest mAP@50 = 0.22
reliability_multiplier_vest = 3.07
high_conf = 0.75
low_conf = 0.25

def check_hardhat_violation(detections): 

    hard_hat_conf = 0.0 #class 3
    no_hard_hat_conf = 0.0 #class 8

    for box in detections:
        if box.cls == 3: #cls = class
            hard_hat_conf = float(box.conf[0])

        elif box.cls == 8:
            no_hard_hat_conf = float(box.conf[0])

    inverse_hardhat = 1 - hard_hat_conf
    adjusted_inverse = reliability_multiplier * inverse_hardhat

    #branch 1: Very confient NO hardhat > 0.75
    if inverse_hardhat > high_conf:
        return {
            'violation' : True,
            'confidence' : inverse_hardhat,
            'source' : 'inverse_hardhat',
            'reason' : 'primary detector very confident - no hardhat'
        }
    
    #branch 2: very confident hardhat IS present (inverse < 0.25)
    elif inverse_hardhat < low_conf:
        return {
            'violation' : False,
            'confidence' : hard_hat_conf,
            'source' : 'hardhat_high_confidence',
            'reason' : 'primary detector very confident - hardhat present'
        }
    
    #branch 3: uncertain zone, uses secondary opinion
    else: 
        # 3a: NO-Hardhat detector very confident AND beats adjusted inverse
        if no_hard_hat_conf > high_conf and no_hard_hat_conf > adjusted_inverse:
            return {
                'violation': True,
                'confidence': no_hard_hat_conf,
                'source': 'no_hardhat_override',
                'reason': 'Secondary detector very confident and overcomes reliability gap'
            }
        #3b: Uses the weighted comparison
        elif adjusted_inverse > no_hard_hat_conf:
            return {
                'violation': inverse_hardhat >= 0.5,
                'confidence': inverse_hardhat,
                'source': 'adjusted inverse',
                'reason': 'adjusted inverse opinion ovveride - '
            }
        #3c: no other branch is true result relies on 'no hard hat detector'.
        else:
            return {
                'violation' : no_hard_hat_conf >= 0.5,
                'confidence' : no_hard_hat_conf,
                'source' : 'no_hardhat_detector',
                'reason' : 'NO-Hardhat detector wins weighted comparison'
            }

high_conf1 = 0.5
low_conf1 =  0.2

def check_safety_vest_violation(detections): 

    safety_vest_conf = 0.0 #class 13
    no_safety_vest_conf = 0.0 #class 10

    for box in detections:
        if box.cls == 13: #cls = class
            safety_vest_conf = float(box.conf[0])

        elif box.cls == 10:
            no_safety_vest_conf = float(box.conf[0])

    inverse_safety_vest = 1 - safety_vest_conf
    adjusted_inverse = reliability_multiplier_vest * inverse_safety_vest

    #branch 1: Very confient NO safety vest > 0.75
    if inverse_safety_vest > high_conf1:
        return {
            'violation' : True,
            'confidence' : inverse_safety_vest,
            'source' : 'inverse_Safety_vest',
            'reason' : 'primary detector very confident - no safety vest'
        }
    
    #branch 2: very confident safety vest IS present (inverse < 0.25)
    elif inverse_safety_vest < low_conf1:
        return {
            'violation' : False,
            'confidence' : safety_vest_conf,
            'source' : 'Safety_Vest_high_confidence',
            'reason' : 'primary detector very confident - safety vest present'
        }
    
    #branch 3: uncertain zone, uses secondary opinion
    else: 
        # 3a: NO-safety vest detector very confident AND beats adjusted inverse
        if no_safety_vest_conf > high_conf1 and no_safety_vest_conf > adjusted_inverse:
            return {
                'violation': True,
                'confidence': no_safety_vest_conf,
                'source': 'NO_Safety_vest_override',
                'reason': 'Secondary detector very confident and overcomes reliability gap'
            }
        #3b: Uses the weighted comparison
        elif adjusted_inverse > no_safety_vest_conf:
            return {
                'violation': inverse_safety_vest >= 0.5,
                'confidence': inverse_safety_vest,
                'source': 'adjusted inverse',
                'reason': 'adjusted inverse opinion ovveride '
            }
        #3c: no other branch is true result relies on 'no safety vest detector'.
        else:
            return {
                'violation' : no_safety_vest_conf >= 0.5,
                'confidence' : no_safety_vest_conf,
                'source' : 'no_safety_vest_detector',
                'reason' : 'NO-Safety Vest detector wins weighted comparison'
            }
    

