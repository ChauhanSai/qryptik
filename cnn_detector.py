# cnn_detector.py
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

# ---------------- preproc helpers ----------------
def pubkey_to_tensor(pub: np.ndarray) -> np.ndarray:
    """
    Build a 3-channel image-like tensor: (k, n_total, 3)
    C0: raw normalized to [0,1] by dividing by max element (assume field q-1)
    C1: row-wise z-score
    C2: zero mask (1 if exactly zero else 0)
    """
    X = pub.astype(np.float32)
    k, n = X.shape
    maxv = float(max(1, X.max()))
    c0 = X / maxv

    # row-wise z
    mu = c0.mean(axis=1, keepdims=True)
    sd = c0.std(axis=1, keepdims=True) + 1e-6
    c1 = (c0 - mu) / sd

    # zero mask
    c2 = (pub == 0).astype(np.float32)

    out = np.stack([c0, c1, c2], axis=-1)  # (k, n, 3)
    return out

# ---------------- model ----------------
def build_cnn_2d(input_shape, num_classes=2):
    """
    input_shape = (k, n_total, 3)
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 7), padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 7), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 5), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='linear')(x)  # we’ll use from_logits=True

    model = models.Model(inp, out)
    return model

# focal loss
def focal_loss(gamma=2.0, alpha=0.5):
    def _loss(y_true, y_logits):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=2)
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_oh, logits=y_logits)
        p_t = tf.reduce_sum(y_true_oh * tf.nn.softmax(y_logits), axis=1)
        loss = alpha * tf.pow(1. - p_t, gamma) * ce
        return tf.reduce_mean(loss)
    return _loss

# ---------------- save/load ----------------
def save_model_and_meta(model, model_path, meta_path, input_shape, threshold):
    model.save(model_path)
    meta = {"input_shape": input_shape, "threshold": float(threshold)}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

def load_model_and_meta(model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta
# add to cnn_detector.py
def cnn_inference(pub, GF, model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    """
    Returns (suspicious: bool, p_bad: float in [0,1]) using saved threshold.
    """
    model, meta = load_model_and_meta(model_path, meta_path)
    thr = meta.get("threshold", 0.5)

    X = pubkey_to_tensor(np.array(pub, dtype=np.int32))
    X = np.expand_dims(X, axis=0)  # batch
    logits = model.predict(X, verbose=0)[0]
    p = tf.nn.softmax(logits).numpy()
    p_bad = float(p[1])
    return (p_bad >= thr), p_bad
