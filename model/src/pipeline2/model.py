import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.optimizers import Adam


# METRICS DEFINITION
class BinaryBalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='bin_bal_acc', **kwargs):
        super(BinaryBalancedAccuracy, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.tn.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        sens = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        spec = tf.math.divide_no_nan(self.tn, self.tn + self.fp)
        return (sens + spec) / 2.0

    def reset_state(self):
        for v in [self.tp, self.tn, self.fp, self.fn]: v.assign(0.0)


class MulticlassBalancedAccuracy(tf.keras.metrics.Metric):

    def __init__(self, num_classes=3, name='multi_bal_acc', **kwargs):
        super(MulticlassBalancedAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(name='cm', shape=(num_classes, num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.cm.assign_add(tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32))

    def result(self):
        diag = tf.linalg.tensor_diag_part(self.cm)
        row_sums = tf.reduce_sum(self.cm, axis=1)
        # recall means for class
        return tf.reduce_mean(tf.math.divide_no_nan(diag, row_sums))

    def reset_state(self):
        self.cm.assign(tf.zeros((self.num_classes, self.num_classes)))


# MODEL ARCHITECTURE
def build_arbitro_model_speed_aware_lstm_multiclip(
    frame_input_shape=(16, 224, 398, 3),
    max_clips=4,
    l2_shared=0.001,
    l2_head=0.005,
    dropout_video=0.40,
    dropout_shared=0.45,
    dropout_head=0.40,
    freeze_ratio=0.70,
    lstm_units=128,
    bidirectional=True,
):
    n_frames, frame_height, frame_width, channels = frame_input_shape

    # INPUT LAYERS
    video_input = layers.Input(shape=(max_clips, *frame_input_shape), name="video_input")
    clip_mask = layers.Input(shape=(max_clips,), name="clip_mask")
    speed_input = layers.Input(shape=(1,), name="speed_input")

    # VIDEO BRANCH (InceptionResNetV2 + BiLSTM)
    base_cnn = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(frame_height, frame_width, channels),
    )

    # Freeze freeze_ratio layers (Transfer Learning)
    num_layers = len(base_cnn.layers)
    freeze_until = int(num_layers * freeze_ratio)
    for layer in base_cnn.layers[:freeze_until]:
        layer.trainable = False

    x = layers.TimeDistributed(
        layers.TimeDistributed(base_cnn, name="td_cnn_frames"),
        name="td_cnn_clips"
    )(video_input)

    x = layers.TimeDistributed(
        layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="td_gap_frames"),
        name="td_gap_clips"
    )(x)

    x = layers.Dropout(dropout_video, name="dropout_video")(x)

    lstm = layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.0, name="lstm")
    if bidirectional:
        clip_encoder = layers.Bidirectional(lstm, name="bilstm")
    else:
        clip_encoder = lstm

    x_clip = layers.TimeDistributed(clip_encoder, name="td_lstm_per_clip")(x)

    def masked_mean(tensors):
        feats, mask = tensors
        mask = tf.cast(mask, feats.dtype)
        mask = tf.expand_dims(mask, -1)
        summed = tf.reduce_sum(feats * mask, axis=1)
        denom = tf.reduce_sum(mask, axis=1)
        return tf.math.divide_no_nan(summed, denom)

    x_video = layers.Lambda(masked_mean, name="clip_fusion")([x_clip, clip_mask])

    # SPEED BRANCH
    x_speed = layers.Dense(32, activation="relu", name="speed_embed")(speed_input)
    x_speed = layers.Dropout(0.20, name="dropout_speed")(x_speed)

    # FUSION
    x = layers.Concatenate(name="fusion")([x_video, x_speed])

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_shared),
        name="dense_shared",
    )(x)
    x = layers.LayerNormalization(name="ln_shared")(x)
    x = layers.Dropout(dropout_shared, name="dropout_shared")(x)

    # HEADS
    # Severity
    x_sev = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_head), name="sev_dense")(x)
    x_sev = layers.Dropout(dropout_head, name="sev_dropout")(x_sev)
    head_severity = layers.Dense(3, activation="softmax", name="head_severity")(x_sev)

    # Offence
    x_off = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_head), name="off_dense")(x)
    x_off = layers.Dropout(dropout_head, name="off_dropout")(x_off)
    head_offence = layers.Dense(1, activation="sigmoid", name="head_offence")(x_off)

    # Action
    x_act = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_head), name="act_dense")(x)
    x_act = layers.Dropout(dropout_head, name="act_dropout")(x_act)
    head_action = layers.Dense(4, activation="softmax", name="head_action")(x_act)

    # Aux Heads
    aux_contact = layers.Dense(1, activation="sigmoid", name="aux_contact")(x)
    aux_bodypart = layers.Dense(3, activation="softmax", name="aux_bodypart")(x)
    aux_touch_ball = layers.Dense(1, activation="linear", name="aux_touch_ball")(x)
    aux_handball = layers.Dense(1, activation="sigmoid", name="aux_handball")(x)
    aux_try_play = layers.Dense(1, activation="linear", name="aux_try_play")(x)

    return Model(
        inputs=[video_input, clip_mask, speed_input],
        outputs={
            "head_severity": head_severity,
            "head_offence": head_offence,
            "head_action": head_action,
            "aux_contact": aux_contact,
            "aux_bodypart": aux_bodypart,
            "aux_touch_ball": aux_touch_ball,
            "aux_handball": aux_handball,
            "aux_try_play": aux_try_play,
        },
        name="ArbItro",
    )