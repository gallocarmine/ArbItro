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


    def get_class_weights(self):
        # Extract labels
        all_sev = []
        all_off = []
        all_act = []

        for s in self.samples:
            all_sev.append(get_severity_class_raw(s))
            all_off.append(OFFENCE_MAP.get(s.get('Offence', ''), 0))
            all_act.append(MACRO_ACTION_MAP.get(s.get('Action class', ''), 0))

        # Compute class weights
        # Severity
        classes_sev = np.unique(all_sev)
        weights_sev = class_weight.compute_class_weight(
            class_weight='balanced', classes=classes_sev, y=all_sev
        )
        dict_sev = dict(zip(classes_sev, weights_sev))

        # Offence
        classes_off = np.unique(all_off)
        weights_off = class_weight.compute_class_weight(
            class_weight='balanced', classes=classes_off, y=all_off
        )
        dict_off = dict(zip(classes_off, weights_off))

        # Action
        classes_act = np.unique(all_act)
        weights_act = class_weight.compute_class_weight(
            class_weight='balanced', classes=classes_act, y=all_act
        )
        dict_act = dict(zip(classes_act, weights_act))

        return {
            "head_severity": dict_sev,
            "head_offence": dict_off,
            "head_action": dict_act,
        }

    def __len__(self):
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_samples_temp = [self.samples[k] for k in indexes]
        return self.__data_generation(list_samples_temp)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_video_frames_native(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frame_count = 0

        if frame_count <= 0:
            cap.release()
            return np.zeros((self.n_frames, *self.dim, 3), dtype='float32')

        if frame_count >= self.n_frames:
            indices = np.linspace(0, frame_count - 1, self.n_frames, dtype=int)
        else:
            indices = np.array([i % frame_count for i in range(self.n_frames)])

        for target_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((*self.dim, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        frames = np.array(frames, dtype='float32')
        return frames

    def _parse_replay_speed(self, x, default=1.0):
        # Accepts float, string, or mixed formats
        try:
            if isinstance(x, (int, float)):
                return float(x)
            if x is None:
                return float(default)
            s = str(x)
            m = re.search(r"[-+]?\d*\.?\d+", s)
            return float(m.group(0)) if m else float(default)
        except:
            return float(default)

    def _apply_augmentation(self, video_data, sample):
        aug_type = sample.get('augment_type', 'original')

        # Skip augmentation for original samples
        if aug_type == 'original':
            return video_data

        augmented = video_data.copy()
        n_frames, h, w, c = augmented.shape

        # Parse compound ops: "flip_zoom" -> ["flip", "zoom"]
        ops = aug_type.split('_')

        # Flip
        if 'flip' in ops:
            # axis 2 is width (Frames, H, W, C)
            augmented = np.flip(augmented, axis=2)

        # Rotation
        if 'rotation' in ops:
            # large = stronger rotation
            limit = 10 if 'large' in ops else 5
            angle = np.random.uniform(-limit, limit)

            # rotation matrix (centered)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

            for i in range(n_frames):
                augmented[i] = cv2.warpAffine(
                    augmented[i], M, (w, h),
                    borderMode=cv2.BORDER_REFLECT  # reflect borders to avoid black edges
                )

        # Zoom
        if 'zoom' in ops:
            # out = shrinks (adds padding), in = enlarges (crops center)
            if 'out' in ops:
                scale = np.random.uniform(0.85, 0.95)
            elif 'in' in ops:
                scale = np.random.uniform(1.1, 1.25)
            else:
                scale = np.random.uniform(0.9, 1.1)

            for i in range(n_frames):
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(augmented[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                if scale > 1.0:
                    # Zoom in: center crop
                    sh, sw = (new_h - h) // 2, (new_w - w) // 2
                    augmented[i] = resized[sh:sh + h, sw:sw + w]
                else:
                    # Zoom out: black padding (centered)
                    ph, pw = (h - new_h) // 2, (w - new_w) // 2

                    # Black canvas
                    canvas = np.zeros((h, w, c), dtype=augmented[i].dtype)

                    # Safety bounds to avoid off-by-one rounding errors
                    end_h = min(ph + new_h, h)
                    end_w = min(pw + new_w, w)

                    # Paste resized image at center
                    canvas[ph:end_h, pw:end_w] = resized[:end_h - ph, :end_w - pw]
                    augmented[i] = canvas

        # Brightness
        if 'brightness' in ops:
            factor = np.random.uniform(0.8, 1.2)
            # Clip to valid float32 range
            augmented = np.clip(augmented * factor, 0.0, 1.0)

        # Contrast
        if 'contrast' in ops:
            factor = np.random.uniform(0.7, 1.3)
            mean = augmented.mean()
            # Standard contrast formula
            augmented = np.clip((augmented - mean) * factor + mean, 0.0, 1.0)

        return augmented

    def __data_generation(self, list_samples_temp):
        # BATCH ALLOCATION
        X_video = np.zeros((self.batch_size, self.max_clips, self.n_frames, *self.dim, 3), dtype='float32')
        X_clip_mask = np.zeros((self.batch_size, self.max_clips), dtype='float32')
        X_speed = np.zeros((self.batch_size, 1), dtype='float32')

        y_sev, y_off, y_act = [], [], []
        if self.use_auxiliary_features:
            y_bodypart, y_contact, y_touch_ball, y_try_play, y_handball = [], [], [], [], []

        for i, sample in enumerate(list_samples_temp):
            clips = sample.get("Clips", [])
            n = min(len(clips), self.max_clips)

            speed_sum = 0.0
            speed_cnt = 0.0

            for j in range(n):
                clip_info = clips[j]

                parts = clip_info["Url"].replace('\\', '/').split('/')
                try:
                    start_idx = next(idx for idx, p in enumerate(parts) if "action_" in p)
                except:
                    start_idx = -2

                clean_path = os.path.join(*parts[start_idx:])
                full_path = os.path.join(self.base_video_path, clean_path)

                if not os.path.exists(full_path):
                    for ext in ['.mp4', '.avi', '.mkv', '.mov']:
                        if os.path.exists(full_path + ext):
                            full_path += ext
                            break

                video_data = self._load_video_frames_native(full_path).astype(np.float32)
                video_data = video_data / 255.0
                video_data = self._apply_augmentation(video_data, sample)
                video_data = (video_data * 255.0).astype(np.float32)
                video_data = preprocess_input(video_data)

                X_video[i, j] = video_data
                X_clip_mask[i, j] = 1.0

                # Replay speed
                rs = self._parse_replay_speed(clip_info.get("Replay speed", None), default=1.0)
                speed_sum += rs
                speed_cnt += 1.0

            # Masked mean of replay speeds (0 if no clips)
            X_speed[i, 0] = (speed_sum / speed_cnt) if speed_cnt > 0 else 0.0

            # Labels
            y_sev.append(get_severity_class_raw(sample))
            y_off.append(OFFENCE_MAP.get(sample.get('Offence', ''), 0))
            y_act.append(MACRO_ACTION_MAP.get(sample.get('Action class', ''), 0))

            if self.use_auxiliary_features:
                y_bodypart.append(BODYPART_MAP.get(sample.get('Bodypart', ''), 0))
                y_contact.append(CONTACT_MAP.get(sample.get('Contact', ''), 0))
                y_touch_ball.append(TOUCH_BALL_MAP.get(sample.get('Touch ball', ''), 0.0))
                y_try_play.append(TRY_TO_PLAY_MAP.get(sample.get('Try to play', ''), 0.5))
                y_handball.append(HANDBALL_MAP.get(sample.get('Handball', ''), 0))

        inputs = {
            "video_input": X_video,
            "clip_mask": X_clip_mask,
            "speed_input": X_speed,
        }

        outputs = {
            "head_severity": to_categorical(y_sev, num_classes=3),
            "head_offence": np.array(y_off, dtype='float32').reshape(-1, 1),
            "head_action": to_categorical(y_act, num_classes=4),
        }

        if self.use_auxiliary_features:
            outputs.update({
                "aux_contact": np.array(y_contact, dtype='float32').reshape(-1, 1),
                "aux_bodypart": to_categorical(y_bodypart, num_classes=3),
                "aux_touch_ball": np.array(y_touch_ball, dtype='float32').reshape(-1, 1),
                "aux_handball": np.array(y_handball, dtype='float32').reshape(-1, 1),
                "aux_try_play": np.array(y_try_play, dtype='float32').reshape(-1, 1),
            })

        return inputs, outputs
