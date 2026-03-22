import os
import json
import numpy as np
import cv2
import re
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from sklearn.utils import class_weight
from tensorflow.keras.utils import Sequence, to_categorical


# MAPPINGS & CONFIGURATION
OFFENCE_MAP = {"No offence": 0, "": 0, "Between": 1, "Offence": 1}

SEVERITY_CLASS_MAP = {"NO_CARD": 0, "YELLOW_CARD": 1, "RED_CARD": 2}

MACRO_ACTION_MAP = {
    # Group 0: Neutral / No Impact
    "": 0, "Dont know": 0, "Dive": 0,

    # Group 1: Standard Tackles
    "Challenge": 1, "Tackling": 1, "Standing tackling": 1,

    # Group 2: Upper Body / Arm Fouls
    "Holding": 2, "Pushing": 2, "Elbowing": 2,

    # Group 3: Dangerous Play
    "High leg": 3,
}

BODYPART_MAP = {"": 0, "Under body": 1, "Upper body": 2}
CONTACT_MAP = {"Without contact": 0, "": 0, "With contact": 1}
TOUCH_BALL_MAP = {"No": 0.0, "": 0.0, "Maybe": 0.5, "Yes": 1.0}
TRY_TO_PLAY_MAP = {"No": 0.0, "": 0.5, "Yes": 1.0}
HANDBALL_MAP = {"No handball": 0, "": 0, "Handball": 1}


def get_severity_class_raw(row):
    try:
        raw_sev = float(row.get('Severity', 0))
    except (ValueError, TypeError):
        raw_sev = 0.0
    if raw_sev >= 5.0:
        return SEVERITY_CLASS_MAP["RED_CARD"]
    elif raw_sev >= 3.0:
        return SEVERITY_CLASS_MAP["YELLOW_CARD"]
    else:
        return SEVERITY_CLASS_MAP["NO_CARD"]