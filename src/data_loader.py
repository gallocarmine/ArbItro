import os
import numpy as np
import json
import cv2
from tensorflow.keras.utils import Sequence, to_categorical


# 1. MAPPINGS & CONFIGURATION
OFFENCE_MAP = {"No offence": 0, "": 0, "Between": 0, "Offence": 1}

SEVERITY_CLASS_MAP = {"NO_CARD": 0, "YELLOW_CARD": 1, "RED_CARD": 2}

ACTION_CLASS_MAP = {
    "": 0, "Dont know": 0, "Challenge": 1, "Tackling": 2, "Standing tackling": 3,
    "High leg": 4, "Holding": 5, "Pushing": 6, "Elbowing": 7, "Dive": 8
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

    def __init__(self, json_path, base_video_path, batch_size=4, dim=(224, 398),
                 n_frames=16, shuffle=True, use_auxiliary_features=False,
                 augment=True):
        super().__init__()
        self.dim = dim
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
        count_aug = {0: 0, 1: 0, 2: 0}

        for s in temp_samples:
            severity_class = get_severity_class_raw(s)
            count_org[severity_class] += 1

            # Original sample
            s_original = s.copy()
            s_original['augment_type'] = 'original'
            self.samples.append(s_original)

            if self.augment:

                # YELLOW CARD: 2x (original + flip)
                if severity_class == 1:
                    s_flip = s.copy()
                    s_flip['augment_type'] = 'flip'
                    self.samples.append(s_flip)
                    count_aug[1] += 1

                # RED CARD: 10x (original + 9 augmented)
                elif severity_class == 2:
                    augmentation_types = [
                        'flip', 'rotation_small', 'flip_rotation', 'zoom_in',
                        'brightness', 'contrast', 'rotation_contrast', 'zoom_brightness', 'zoom_out'
                    ]
                    for aug_type in augmentation_types:
                        s_augmented = s.copy()
                        s_augmented['augment_type'] = aug_type
                        self.samples.append(s_augmented)
                        count_aug[2] += 1

        self.n_samples = len(self.samples)
        self.indexes = np.arange(self.n_samples)
        self.on_epoch_end()

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
        frames = np.array(frames, dtype='float32') / 255.0
        return frames

    def _apply_augmentation(self, video_data, sample):
        augment_type = sample.get('augment_type', 'original')
        if augment_type == 'original': return video_data

        augmented = video_data.copy()
        n_frames, h, w, c = augmented.shape

        if 'flip' in augment_type:
            augmented = np.flip(augmented, axis=2)

        if 'rotation' in augment_type:
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            for i in range(n_frames):
                augmented[i] = cv2.warpAffine(augmented[i], M, (w, h), borderMode=cv2.BORDER_REFLECT)

        if 'zoom' in augment_type:
            if 'in' in augment_type:
                scale = np.random.uniform(1.1, 1.2)
            elif 'out' in augment_type:
                scale = np.random.uniform(0.85, 0.95)
            else:
                scale = np.random.uniform(0.9, 1.15)

            for i in range(n_frames):
                new_h, new_w = int(h * scale), int(w * scale)
                resized = cv2.resize(augmented[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if scale > 1.0:
                    sh, sw = (new_h - h) // 2, (new_w - w) // 2
                    augmented[i] = resized[sh:sh + h, sw:sw + w]
                else:
                    ph, pw = (h - new_h) // 2, (w - new_w) // 2
                    cropped = np.zeros((h, w, c), dtype=augmented[i].dtype)
                    cropped[ph:ph + new_h, pw:pw + new_w] = resized
                    augmented[i] = cropped

        if 'brightness' in augment_type:
            augmented = np.clip(augmented * np.random.uniform(0.8, 1.2), 0.0, 1.0)

        if 'contrast' in augment_type:
            mean = augmented.mean()
            augmented = np.clip((augmented - mean) * np.random.uniform(0.7, 1.3) + mean, 0.0, 1.0)

        return augmented

    def __data_generation(self, list_samples_temp):
        X_video = np.empty((self.batch_size, self.n_frames, *self.dim, 3), dtype='float32')
        X_speed = np.empty((self.batch_size, 1), dtype='float32')
        y_sev, y_off, y_act = [], [], []

        if self.use_auxiliary_features:
            y_bodypart, y_contact, y_touch_ball, y_try_play, y_handball = [], [], [], [], []

        for i, sample in enumerate(list_samples_temp):
            clip_info = sample["Clips"][0]
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

            video_data = self._load_video_frames_native(full_path)
            X_video[i, :] = self._apply_augmentation(video_data, sample)
            X_speed[i, 0] = float(clip_info.get('Replay speed', 1.0))

            y_sev.append(get_severity_class_raw(sample))
            y_off.append(OFFENCE_MAP.get(sample.get('Offence', ''), 0))
            y_act.append(ACTION_CLASS_MAP.get(sample.get('Action class', ''), 0))

            if self.use_auxiliary_features:
                y_bodypart.append(BODYPART_MAP.get(sample.get('Bodypart', ''), 0))
                y_contact.append(CONTACT_MAP.get(sample.get('Contact', ''), 0))
                y_touch_ball.append(TOUCH_BALL_MAP.get(sample.get('Touch ball', ''), 0.0))
                y_try_play.append(TRY_TO_PLAY_MAP.get(sample.get('Try to play', ''), 0.5))
                y_handball.append(HANDBALL_MAP.get(sample.get('Handball', ''), 0))

        inputs = {"video_input": X_video, "speed_input": X_speed}
        outputs = {
            "head_severity": to_categorical(y_sev, num_classes=3),
            "head_offence": np.array(y_off, dtype='float32').reshape(-1, 1),
            "head_action": to_categorical(y_act, num_classes=9)
        }
        if self.use_auxiliary_features:
            outputs.update({
                "aux_contact": np.array(y_contact, dtype='float32').reshape(-1, 1),
                "aux_bodypart": to_categorical(y_bodypart, num_classes=3),
                "aux_touch_ball": np.array(y_touch_ball, dtype='float32').reshape(-1, 1),
                "aux_handball": np.array(y_handball, dtype='float32').reshape(-1, 1),
                "aux_try_play": np.array(y_try_play, dtype='float32').reshape(-1, 1)
            })

        return inputs, outputs