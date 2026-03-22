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


# DATA GENERATOR CLASS
class ArbItroDataGenerator(Sequence):

    def __init__(self, json_path, base_video_path, batch_size=16, max_clips=4, dim=(224, 398),
                 n_frames=16, shuffle=True, use_auxiliary_features=False,
                 augment=True):
        super().__init__()
        self.dim = dim
        self.max_clips = max_clips
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.base_video_path = base_video_path
        self.shuffle = shuffle
        self.use_auxiliary_features = use_auxiliary_features
        self.augment = augment

        try:
            with open(json_path, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: JSON file not found: {json_path}")

        # Samples extraction
        temp_samples = []
        actions = raw_data.get("Actions", raw_data)
        if isinstance(actions, dict):
            for k, v in actions.items():
                if "Clips" in v and len(v["Clips"]) > 0: temp_samples.append(v)
        elif isinstance(actions, list):
            for v in actions:
                if "Clips" in v and len(v["Clips"]) > 0: temp_samples.append(v)

        self.samples = []

        count_org = {0: 0, 1: 0, 2: 0}
        stats = {'aug_red': 0, 'aug_yellow': 0, 'aug_no_offence': 0, 'aug_rare_action': 0}

        # Rare actions targeted for augmentation
        RARE_ACTIONS = ["High leg", "Elbowing", "Pushing", "Dive", "Holding"]

        for s in temp_samples:
            severity_class = get_severity_class_raw(s)
            offence_val = OFFENCE_MAP.get(s.get('Offence', ''), 0)
            action_str = s.get('Action class', '')

            count_org[severity_class] += 1

            # Always add the original sample
            s_original = s.copy()
            s_original['augment_type'] = 'original'
            self.samples.append(s_original)

            # Strategic Augmentation
            if self.augment:
                # Red Cards (x10)
                if severity_class == 2:
                    augmentation_types = [
                        'flip', 'rotation_small', 'flip_rotation', 'zoom_in',
                        'brightness', 'contrast', 'rotation_contrast', 'zoom_brightness', 'zoom_out'
                    ]
                    for aug_type in augmentation_types:
                        s_aug = s.copy()
                        s_aug['augment_type'] = aug_type
                        self.samples.append(s_aug)
                        stats['aug_red'] += 1

                # Yellow Cards (x2)
                elif severity_class == 1:
                    aug_types = ['flip']
                    for aug in aug_types:
                        s_aug = s.copy()
                        s_aug['augment_type'] = aug
                        self.samples.append(s_aug)
                        stats['aug_no_offence'] += 1

                # Rare Actions (x2)
                elif action_str in RARE_ACTIONS:
                    aug_types = ['flip']
                    for aug in aug_types:
                        s_aug = s.copy()
                        s_aug['augment_type'] = aug
                        self.samples.append(s_aug)
                        stats['aug_rare_action'] += 1

        self.n_samples = len(self.samples)
        self.indexes = np.arange(self.n_samples)

        self.on_epoch_end()