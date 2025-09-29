# cnn_detector.py
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def pubkey_to_vector(pub_key):
    """
    Convert galois.FieldArray / numpy array to 1D int vector.
    """
    arr = np.array(pub_key, dtype=int)
    return arr.flatten().astype(np.int64)

def build_cnn(input_len, num_classes=2):
    """
    Build a small 1D CNN for flattened public keys.
    Input shape for Keras: (input_len, 1)
    """
    inp = layers.Input(shape=(input_len, 1))
    x = layers.Conv1D(32, kernel_size=5, padding="same", activation="relu")(inp)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_model_and_meta(model, model_path="cnn_model.h5", meta_path="cnn_meta.json", input_len=None):
    """
    Save the Keras model (HDF5) and a small JSON metadata file that stores input_len.
    """
    model.save(model_path)
    meta = {"input_len": int(input_len)}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

def load_model_and_meta(model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    """
    Load model and metadata. Returns (model, input_len)
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)
    input_len = int(meta["input_len"])
    model = models.load_model(model_path)
    return model, input_len

def cnn_inference(pub_key, GF, model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    """
    Run CNN inference: returns boolean suspicious, float score (probability of 'bad' class)
    - pub_key: galois.FieldArray or numpy array (k x n*(r+1))
    - GF: field object (to get GF.order)
    """
    model, input_len = load_model_and_meta(model_path, meta_path)
    vec = pubkey_to_vector(pub_key).astype(np.float32)
    # normalize to [0,1] by dividing by q-1
    q = GF.order
    vec = vec / float(q - 1)
    if vec.shape[0] != input_len:
        raise ValueError(f"Input length {vec.shape[0]} does not match model input_len {input_len}")
    x = vec.reshape(1, input_len, 1)  # batch=1, length, channels=1
    probs = model.predict(x, verbose=0)[0]  # [p_good, p_bad]
    p_bad = float(probs[1])
    suspicious = p_bad > 0.5  # you can tune threshold
    return suspicious, p_bad
