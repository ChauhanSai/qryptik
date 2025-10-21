import numpy as np

def tamper_pubkey_zero_cols(pub, num_zero_cols=4, block_width=1):
    """
    Tamper the public key by zeroing some columns (simple, effective tampering).
    """
    pk = np.array(pub, dtype=int).copy()
    n_cols = pk.shape[1]
    cols = np.linspace(0, n_cols-1, num_zero_cols, dtype=int)
    for c in cols:
        pk[:, c:c+block_width] = 0
    return pk

def tamper_repeat_columns(pub, repeat_stride=2):
    pk = np.array(pub, dtype=int).copy()
    for i in range(0, pk.shape[1], repeat_stride*2):
        end = min(i + repeat_stride, pk.shape[1])
        src = pk[:, i:end]
        dst_start = i + repeat_stride
        dst_end = min(dst_start + repeat_stride, pk.shape[1])
        if dst_start < pk.shape[1]:
            pk[:, dst_start:dst_end] = src[:, :dst_end-dst_start]
    return pk

def tamper_small_hamming_columns(pub, num_cols=8, hamming_weight=1):
    pk = np.array(pub, dtype=int).copy()
    n_cols = pk.shape[1]
    cols = np.random.choice(n_cols, size=num_cols, replace=False)
    for c in cols:
        row_idx = np.random.choice(pk.shape[0], size=hamming_weight, replace=False)
        newcol = np.zeros(pk.shape[0], dtype=int)
        newcol[row_idx] = 1
        pk[:, c] = newcol
    return pk


def tamper_block_low_rank(pub, block_cols=8, rank=1):
    pk = np.array(pub, dtype=int).copy()
    n_cols = pk.shape[1]
    start = np.random.randint(0, max(1, n_cols - block_cols + 1))
    end = start + block_cols
    block = pk[:, start:end]
    # force low rank via outer product of two random integer vectors
    u = np.random.randint(0, 3, size=(block.shape[0], 1))
    v = np.random.randint(0, 3, size=(1, block.shape[1]))
    low_rank = (u @ v)
    pk[:, start:end] = low_rank
    return pk

def tamper_near_duplicate_columns(pub, num_pairs=4, jitter=1):
    pk = np.array(pub, dtype=int).copy()
    n_cols = pk.shape[1]
    for _ in range(num_pairs):
        a, b = np.random.choice(n_cols, size=2, replace=False)
        pk[:, b] = pk[:, a]
        # light jitter on a few rows
        rows = np.random.choice(pk.shape[0], size=min(3, pk.shape[0]), replace=False)
        pk[rows, b] = (pk[rows, b] + np.random.randint(-jitter, jitter+1, size=rows.shape)) 
    return pk

def tamper_random_sparse_noise(pub, flips= int(0.01*1e6) ):
    pk = np.array(pub, dtype=int).copy()
    # choose ~1% entries if shape is big; make it scale with shape
    H, W = pk.shape
    flips = max(1, int(0.01 * H * W))
    idx = np.random.choice(H*W, size=flips, replace=False)
    flat = pk.reshape(-1)
    flat[idx] = 0  # knock out to zero or small
    pk = flat.reshape(H, W)
    return pk

def tamper_banding_wipe(pub, band_width=2):
    pk = np.array(pub, dtype=int).copy()
    H, W = pk.shape
    start = np.random.randint(0, max(1, H - band_width + 1))
    pk[start:start+band_width, :] = 0
    return pk
