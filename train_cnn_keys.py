# train_cnn_keys_tf.py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from cnn_detector import build_cnn, pubkey_to_vector, save_model_and_meta
# Import your RLCE keygen function
# Replace 'rlce_core' with the filename where KeyGen is defined (no .py)
from main_RLCE_implementation import KeyGen  # <-- change this if your file name is different

def tamper_pubkey_zero_cols(pub, num_zero_cols=4, block_width=1):
    """
    Tamper the public key by zeroing some columns (simple, effective tampering).
    Adjust strategy to generate diverse bad keys in practice.
    """
    pk = np.array(pub, dtype=int).copy()
    n_cols = pk.shape[1]
    start = 0
    # zero out evenly-spaced columns
    cols = np.linspace(0, n_cols-1, num_zero_cols, dtype=int)
    for c in cols:
        pk[:, c:c+block_width] = 0
    return pk

def generate_dataset(n=63, k=36, t=10, r=1, num_good=300, num_bad=300):
    good = []
    bad = []
    GF = None
    for _ in range(num_good):
        pub, priv = KeyGen(n=n, k=k, t=t, r=r)
        GF = priv["GF"]
        good.append(np.array(pub, dtype=int))
    for _ in range(num_bad):
        pub, priv = KeyGen(n=n, k=k, t=t, r=r)
        # tamper in different ways; here we zero some columns
        bad_pub = tamper_pubkey_zero_cols(pub, num_zero_cols=8, block_width=1)
        bad.append(np.array(bad_pub, dtype=int))
    X = np.concatenate(good + bad, axis=0)  # shape [N, k, n*(r+1)] flattened below
    # Flatten each matrix row-major
    X_vecs = np.array([pub.flatten() for pub in (good + bad)])
    # normalization by GF order
    q = GF.order
    X_norm = X_vecs.astype(np.float32) / float(q - 1)
    y = np.array([0]*len(good) + [1]*len(bad), dtype=np.int32)
    return X_norm, y, GF

def train_model(n=63, k=36, t=10, r=1, model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    print("Generating dataset... (this calls KeyGen many times, so it may take a while)")
    X, y, GF = generate_dataset(n=n, k=k, t=t, r=r, num_good=500, num_bad=500)
    N, input_len = X.shape
    print(f"Dataset generated: N={N}, input_len={input_len}, q={GF.order}")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # build model
    model = build_cnn(input_len, num_classes=2)
    model.summary()

    # prepare data for Keras: shape (N, input_len, 1)
    X_train = X_train.reshape((-1, input_len, 1))
    X_val = X_val.reshape((-1, input_len, 1))

    # checkpoint callback to save best model
    checkpoint_cb = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # train
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=15, batch_size=32, callbacks=[checkpoint_cb])

    # Save meta (input_len)
    save_model_and_meta(model, model_path=model_path, meta_path=meta_path, input_len=input_len)
    print("Saved model and metadata.")

if __name__ == "__main__":
    # prototype params (fast)
    train_model(n=63, k=36, t=10, r=1)
    # For production, use larger n/k (but training will be heavier)
