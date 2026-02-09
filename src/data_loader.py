import os
import numpy as np
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