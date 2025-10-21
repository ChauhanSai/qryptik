# train_cnn_keys.py (replace file)
import json, math, random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from cnn_detector import build_cnn_2d, pubkey_to_tensor, save_model_and_meta
from main_RLCE_implementation import KeyGen
from tamper_utils import (
    tamper_pubkey_zero_cols,
    tamper_repeat_columns,
    tamper_small_hamming_columns,
    tamper_block_low_rank,
    tamper_near_duplicate_columns,
    tamper_random_sparse_noise,
    tamper_banding_wipe,
)

rng = np.random.default_rng(42)
random.seed(42)

TAMPERS_EASY = [
    lambda pk: tamper_pubkey_zero_cols(pk, num_zero_cols=8, block_width=1),
    lambda pk: tamper_repeat_columns(pk, repeat_stride=2),
]
TAMPERS_HARD = [
    lambda pk: tamper_small_hamming_columns(pk, num_cols=8, hamming_weight=1),
    lambda pk: tamper_block_low_rank(pk, block_cols=8, rank=1),
    lambda pk: tamper_near_duplicate_columns(pk, num_pairs=6, jitter=1),
    lambda pk: tamper_banding_wipe(pk, band_width=2),
    lambda pk: tamper_random_sparse_noise(pk),
]

def gen_base_keys(n=63, k=36, t=10, r=1, count=600):
    base = []
    for _ in range(count):
        pub, priv = KeyGen(n=n, k=k, t=t, r=r)
        base.append(np.array(pub, dtype=np.int32))
    return base

def make_split_by_base(base_keys, val_frac=0.1):
    idx = np.arange(len(base_keys))
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=42, shuffle=True)
    return train_idx, val_idx

def build_dataset_from_base(base_keys, idxs, per_base_good=1, per_base_bad=6, phase="easy"):
    X, y = [], []
    tampers = TAMPERS_EASY if phase == "easy" else (TAMPERS_EASY + TAMPERS_HARD)
    for i in idxs:
        pk = base_keys[i]
        # goods
        for _ in range(per_base_good):
            X.append(pubkey_to_tensor(pk))
            y.append(0)
        # bads
        for _ in range(per_base_bad):
            f = random.choice(tampers)
            X.append(pubkey_to_tensor(f(pk)))
            y.append(1)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y

def train_model(n=63, k=36, t=10, r=1, model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    # 1) generate bases & split by base (prevents leakage)
    base = gen_base_keys(n, k, t, r, count=800)
    tr_idx, va_idx = make_split_by_base(base, val_frac=0.12)

    input_shape = (k, (n*(r+1)), 3)  # (k, n_total, channels)

    def compile_model():
        model = build_cnn_2d(input_shape, num_classes=2)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy",  # we’ll pass one-hot
                      metrics=["accuracy"])
        return model

    def as_xy(X, y):
        y_oh = tf.keras.utils.to_categorical(y, 2)
        return X, y_oh

    # 2) curriculum: easy -> all
    best_threshold = 0.5
    best_model = None
    for phase, epochs in [("easy", 6), ("all", 12)]:
        print(f"\n=== Training phase: {phase} ===")
        X_tr, y_tr = build_dataset_from_base(base, tr_idx, per_base_good=2, per_base_bad=6, phase=phase)
        X_va, y_va = build_dataset_from_base(base, va_idx, per_base_good=2, per_base_bad=6, phase=phase)

        model = compile_model()
        ckpt = ModelCheckpoint("tmp_best.h5", monitor="val_loss", save_best_only=True, verbose=1)
        early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        rlrop = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

        model.fit(*as_xy(X_tr, y_tr),
                  validation_data=as_xy(X_va, y_va),
                  epochs=epochs, batch_size=32,
                  callbacks=[ckpt, early, rlrop],
                  verbose=1)

        # Compute ROC-AUC + best F1 threshold on validation
        # logits -> probs via softmax; suspicious prob = p[:,1]
        p_va = tf.nn.softmax(model.predict(X_va, verbose=0), axis=1).numpy()[:, 1]
        roc = roc_auc_score(y_va, p_va)
        pr = average_precision_score(y_va, p_va)
        prec, rec, th = precision_recall_curve(y_va, p_va)
        f1s = 2*prec*rec/(prec+rec+1e-9)
        i = np.nanargmax(f1s)
        th_best = th[i] if i < len(th) else 0.5
        f1_best = f1s[i]
        print(f"[VAL] ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  bestF1={f1_best:.4f} @thr={th_best:.3f}")

        # keep best (by ROC-AUC)
        if best_model is None or roc > best_model["roc"]:
            best_model = {"roc": roc, "model": model, "threshold": float(th_best)}

    # 3) save best model + meta (shape + threshold)
    save_model_and_meta(best_model["model"], model_path, meta_path, input_shape, best_model["threshold"])
    print(f"Saved model to {model_path} and meta (threshold={best_model['threshold']:.3f}) to {meta_path}")

if __name__ == "__main__":
    train_model()
