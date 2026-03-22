import os
import gc
import time
import sys
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_1_PATH = os.path.join(BASE_DIR, 'ArbItro_Training', 'models', 'pipeline1.keras')
MODEL_2_PATH = os.path.join(BASE_DIR, 'ArbItro_Training', 'models', 'pipeline2.keras')
ACTIONS      = ["NEUTRAL/DIVE", "STANDARD TACKLE", "UPPER BODY FOUL", "HIGH LEG"]

N_FRAMES  = 16
DIM_H     = 224
DIM_W     = 398
MAX_CLIPS = 4

model_pipeline1 = None
model_pipeline2 = None


def log(msg):
    sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


def masked_mean(inputs):
    x, mask = inputs[0], inputs[1]
    mask = tf.expand_dims(mask, axis=-1)
    return tf.reduce_sum(x * mask, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-7)


class BinaryBalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='bin_bal_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.tn.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        return (tf.math.divide_no_nan(self.tp, self.tp + self.fn) +
                tf.math.divide_no_nan(self.tn, self.tn + self.fp)) / 2.0

    def reset_state(self):
        for v in [self.tp, self.tn, self.fp, self.fn]:
            v.assign(0.0)


class MulticlassBalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name='multi_bal_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(name='cm', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.cm.assign_add(
            tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32)
        )

    def result(self):
        diag = tf.linalg.tensor_diag_part(self.cm)
        return tf.reduce_mean(tf.math.divide_no_nan(diag, tf.reduce_sum(self.cm, axis=1)))

    def reset_state(self):
        self.cm.assign(tf.zeros((self.num_classes, self.num_classes)))


def load_models():
    global model_pipeline1, model_pipeline2

    custom_objs = {
        'masked_mean': masked_mean,
        'BinaryBalancedAccuracy': BinaryBalancedAccuracy,
        'MulticlassBalancedAccuracy': MulticlassBalancedAccuracy,
    }

    if os.path.exists(MODEL_1_PATH):
        model_pipeline1 = load_model(MODEL_1_PATH, custom_objects=custom_objs, compile=False)
        log(f"Pipeline 1 loaded: {MODEL_1_PATH}")
    else:
        log(f"Pipeline 1 not found: {MODEL_1_PATH}")

    if os.path.exists(MODEL_2_PATH):
        model_pipeline2 = load_model(MODEL_2_PATH, custom_objects=custom_objs, compile=False)
        log(f"Pipeline 2 loaded: {MODEL_2_PATH}")
    else:
        log(f"Pipeline 2 not found: {MODEL_2_PATH}")


def extract_frames(video_path):
    if not video_path or not os.path.exists(video_path):
        return np.zeros((N_FRAMES, DIM_H, DIM_W, 3), dtype=np.float32)

    cap    = cv2.VideoCapture(video_path)
    frames = []

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return np.zeros((N_FRAMES, DIM_H, DIM_W, 3), dtype=np.float32)

        frame_indices = np.linspace(0, total_frames - 1, N_FRAMES).astype(int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (DIM_W, DIM_H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame.astype(np.float32))
            else:
                frames.append(np.zeros((DIM_H, DIM_W, 3), dtype=np.float32))
    finally:
        cap.release()

    while len(frames) < N_FRAMES:
        frames.append(np.zeros((DIM_H, DIM_W, 3), dtype=np.float32))

    return np.array(frames[:N_FRAMES], dtype=np.float32)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data   = request.json
        paths  = data.get('video_paths', [])
        speeds = data.get('speeds', [])

        valid_speeds = []
        for i in range(len(paths)):
            try:
                s = float(speeds[i]) if i < len(speeds) else 1.0
                s = s if s > 0 else 1.0
            except Exception:
                s = 1.0
            valid_speeds.append(s)

        avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 1.0

        raw_clips = []
        for i in range(MAX_CLIPS):
            if i < len(paths):
                raw_clips.append(extract_frames(paths[i]))
            else:
                raw_clips.append(None)

        raw_mask  = np.array([1.0 if i < len(paths) else 0.0 for i in range(MAX_CLIPS)], dtype=np.float32)
        raw_speed = np.array([[avg_speed]], dtype=np.float32)

        processed_clips = np.zeros((MAX_CLIPS, N_FRAMES, DIM_H, DIM_W, 3), dtype=np.float32)
        for i in range(MAX_CLIPS):
            if raw_clips[i] is not None:
                processed_clips[i] = preprocess_input(raw_clips[i].copy())

        # PIPELINE 2 — multi-clip → decide RED / non-RED
        if not model_pipeline2:
            raise Exception("Pipeline 2 not loaded")

        p_new_raw = model_pipeline2.predict([
            np.expand_dims(processed_clips, axis=0),
            np.expand_dims(raw_mask, axis=0),
            raw_speed,
        ], verbose=0)

        p_dict_new     = (p_new_raw if isinstance(p_new_raw, dict)
                          else {n: v for n, v in zip(model_pipeline2.output_names, p_new_raw)})
        prob_sev_new   = p_dict_new['head_severity'][0]
        prob_off_new   = float(p_dict_new['head_offence'][0][0])
        prob_act_new   = p_dict_new['head_action'][0]
        y_pred_new_sev = int(np.argmax(prob_sev_new))

        # ENSEMBLE
        if y_pred_new_sev == 2:
            final_sev  = 2
            final_conf = float(prob_sev_new[2])

        else:
            if not model_pipeline1:
                raise Exception("Pipeline 1 not loaded")

            old_inputs = []
            for inp in model_pipeline1.inputs:
                name = inp.name.lower()
                if 'speed' in name:
                    old_inputs.append(raw_speed)
                elif 'mask' in name:
                    old_inputs.append(np.array([[1.0]], dtype=np.float32))
                else:
                    old_inputs.append(np.expand_dims(processed_clips[0], axis=0))

            p_old_raw = model_pipeline1.predict(
                old_inputs[0] if len(old_inputs) == 1 else old_inputs, verbose=0
            )

            p_dict_old   = (p_old_raw if isinstance(p_old_raw, dict)
                            else {n: v for n, v in zip(model_pipeline1.output_names, p_old_raw)})
            prob_sev_old  = p_dict_old['head_severity'][0]

            probs_01      = prob_sev_old[:2]
            probs_01_norm = probs_01 / (probs_01.sum() + 1e-7)

            final_sev  = int(np.argmax(probs_01_norm))
            final_conf = float(probs_01_norm[final_sev])

        is_foul = bool(prob_off_new > 0.5)
        act_idx = int(np.argmax(prob_act_new))

        gc.collect()
        return jsonify({
            "severity":      final_sev,
            "severity_conf": final_conf,
            "is_foul":       is_foul,
            "offence_conf":  float(prob_off_new if is_foul else 1.0 - prob_off_new),
            "action_class":  ACTIONS[act_idx] if act_idx < len(ACTIONS) else "UNKNOWN",
            "action_conf":   float(np.max(prob_act_new)),
        })

    except Exception as e:
        log(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_models()
    app.run(port=5000, debug=False)