# evaluate_cnn.py (replace file)
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score, classification_report
)
from cnn_detector import load_model_and_meta, pubkey_to_tensor
from main_RLCE_implementation import KeyGen
from tamper_utils import (
    tamper_pubkey_zero_cols, tamper_repeat_columns, tamper_small_hamming_columns,
    tamper_block_low_rank, tamper_near_duplicate_columns, tamper_random_sparse_noise, tamper_banding_wipe
)

TAMPERS = {
    "zero": lambda pk: tamper_pubkey_zero_cols(pk, num_zero_cols=8),
    "repeat": lambda pk: tamper_repeat_columns(pk, repeat_stride=2),
    "small_hamming": lambda pk: tamper_small_hamming_columns(pk, num_cols=6),
    "low_rank": tamper_block_low_rank,
    "near_dup": tamper_near_duplicate_columns,
    "sparse_noise": tamper_random_sparse_noise,
    "banding": tamper_banding_wipe,
}

def generate_eval(n=63, k=36, t=10, r=1, per_class=300):
    good, bad, labs = [], [], []
    per_tamper = per_class // len(TAMPERS)
    for _ in range(per_class):
        pub, _ = KeyGen(n=n, k=k, t=t, r=r)
        good.append(np.array(pub, dtype=int))
    for name, f in TAMPERS.items():
        for _ in range(per_tamper):
            pub, _ = KeyGen(n=n, k=k, t=t, r=r)
            bad.append(np.array(f(pub), dtype=int))
    X = [pubkey_to_tensor(x) for x in (good + bad)]
    y = np.array([0]*len(good) + [1]*len(bad))
    tamper_labels = (["good"]*len(good)) + ([name for name in TAMPERS for _ in range(per_tamper)])
    return np.stack(X, axis=0), y, tamper_labels

def evaluate_model(model_path="cnn_model.h5", meta_path="cnn_meta.json"):
    model, meta = load_model_and_meta(model_path, meta_path)
    thr = meta.get("threshold", 0.5)

    X, y, tags = generate_eval()
    logits = model.predict(X, verbose=0)
    p1 = (logits - logits.max(axis=1, keepdims=True))  # softmax safe
    p1 = np.exp(p1) ; p1 = p1 / p1.sum(axis=1, keepdims=True)
    p_bad = p1[:,1]
    yhat = (p_bad >= thr).astype(int)

    roc = roc_auc_score(y, p_bad)
    pr = average_precision_score(y, p_bad)
    acc = accuracy_score(y, yhat)
    prec = precision_score(y, yhat)
    rec = recall_score(y, yhat)
    f1 = f1_score(y, yhat)

    print("=== CNN Key Validator Evaluation (improved) ===")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"PR-AUC   : {pr:.4f}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y, yhat))
    print("\nClassification Report:")
    print(classification_report(y, yhat, target_names=["Good Key", "Bad Key"]))

    # per-tamper recall
    print("\nPer-tamper recall (on 'bad'):")
    from collections import defaultdict
    hit = defaultdict(lambda: [0,0])
    for tag, yi, yh in zip(tags, y, yhat):
        if tag == "good": continue
        hit[tag][1] += 1
        if yi==1 and yh==1: hit[tag][0] += 1
    for tag in TAMPERS.keys():
        h, t = hit[tag]
        r = h/t if t else 0.0
        print(f"  {tag:12s}: {r*100:.1f}%")

if __name__ == "__main__":
    evaluate_model()
