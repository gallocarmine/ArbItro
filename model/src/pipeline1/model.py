import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


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
def build_arbitro_model_speed_aware(input_shape=(16, 224, 398, 3)):

    n_frames, frame_height, frame_width, channels = input_shape

    # INPUT LAYERS
    video_input = layers.Input(shape=input_shape, name='video_input')
    speed_input = layers.Input(shape=(1,), name='speed_input')

    # VIDEO BRANCH (InceptionResNetV2 + BiLSTM)
    base_cnn = tf.keras.applications.InceptionResNetV2(
        include_top=False, weights='imagenet',
        input_shape=(frame_height, frame_width, channels)
    )

    # Freeze 70% (Transfer Learning)
    for layer in base_cnn.layers[:int(len(base_cnn.layers) * 0.7)]:
        layer.trainable = False

    x_video = layers.TimeDistributed(base_cnn, name='td_cnn')(video_input)
    x_video = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name='td_gap')(x_video)
    x_video = layers.TimeDistributed(layers.Dropout(0.5))(x_video)

    x_video = layers.Bidirectional(
        layers.LSTM(256, return_sequences=False, dropout=0.5), name='bilstm'
    )(x_video)
    x_video = layers.Dropout(0.5, name='dropout_video')(x_video)

    # SPEED BRANCH
    x_speed = layers.Dense(32, activation='relu', name='speed_embed')(speed_input)
    x_speed = layers.Dropout(0.3, name='dropout_speed')(x_speed)

    # FUSION
    x_combined = layers.Concatenate(name='fusion')([x_video, x_speed])
    x_combined = layers.Dense(512, activation='relu', name='dense_shared')(x_combined)
    x_combined = layers.BatchNormalization()(x_combined)
    x_combined = layers.Dropout(0.4)(x_combined)


    # HEADS
    # Severity
    x_sev = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x_combined)
    x_sev = layers.Dropout(0.3)(x_sev)
    head_severity = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                                 name='head_severity')(x_sev)

    # Offence
    head_offence = layers.Dense(1, activation='sigmoid', name='head_offence')(x_combined)

    # Action
    x_act = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x_combined)
    x_act = layers.Dropout(0.55)(x_act)
    head_action = layers.Dense(9, activation='softmax', kernel_regularizer=regularizers.l2(0.005), name='head_action')(
        x_act)

    # Aux Heads
    aux_contact = layers.Dense(1, activation='sigmoid', name='aux_contact')(x_combined)
    aux_bodypart = layers.Dense(3, activation='softmax', name='aux_bodypart')(x_combined)
    aux_touch_ball = layers.Dense(1, activation='linear', name='aux_touch_ball')(x_combined)
    aux_handball = layers.Dense(1, activation='sigmoid', name='aux_handball')(x_combined)
    aux_try_play = layers.Dense(1, activation='linear', name='aux_try_play')(x_combined)

    model = Model(
        inputs=[video_input, speed_input],
        outputs=[head_severity, head_offence, head_action, aux_contact, aux_bodypart, aux_touch_ball, aux_handball,
                 aux_try_play],
        name='ArbItro'
    )
    return model