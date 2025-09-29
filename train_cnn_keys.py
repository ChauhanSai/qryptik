from cnn_detector import train_cnn, preprocess_dataset
from main_RLCE_implementation import KeyGen

import numpy as np

# Step 1: Generate dataset
def make_dataset(num_good=200, num_bad=200, n=63, k=36, t=10, r=1):
    good_keys = []
    bad_keys = []
    GF = None
    for _ in range(num_good):
        pub, priv = KeyGen(n=n, k=k, t=t, r=r)
        good_keys.append(pub)
        GF = priv["GF"]
    for _ in range(num_bad):
        pub, priv = KeyGen(n=n, k=k, t=t, r=r)
        # tamper with A: make it rank-deficient
        bad_pub = pub.copy()
        bad_pub[:, :10] = 0   # wipe some columns as "bad"
        bad_keys.append(bad_pub)
    X_good = preprocess_dataset(good_keys, GF)
    X_bad  = preprocess_dataset(bad_keys, GF)
    X = np.concatenate([X_good, X_bad], axis=0)
    y = np.array([0]*len(X_good) + [1]*len(X_bad))  # labels
    return X, y

if __name__ == "__main__":
    X, y = make_dataset()
    model = train_cnn(X, y, epochs=10, batch_size=32, lr=1e-3)
    print("✅ Training finished. cnn.pth saved.")
