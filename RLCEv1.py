import numpy as np
import galois

# Define field GF(2^m)
m = 8  # Can be 4, 8, or 10 for larger examples
GF = galois.GF(2**m)

# Set GRS parameters
n = 255   # Code length (number of columns)
k = 128    # Code dimension (number of rows), k < n

# Choose n distinct elements from the field (evaluation points)
x = GF.Random(n)

# Choose n nonzero multipliers (used to weight the evaluations)
y = GF.Random(n, low=1)  # Ensure all y_i ≠ 0

# Construct GRS generator matrix G_s using the Vandermonde matrix
def grs_generator_matrix(x, y, k):
    V = np.vstack([x**i for i in range(k)]).T  # Shape (n, k)
    G = (y[:, np.newaxis]) * V  # Each row scaled by y_i
    return G

# Generate the GRS generator matrix G_s
Gs = grs_generator_matrix(x, y, k)

# Output
print("x (evaluation points):", x)
print("y (multipliers):", y)
print("GRS Generator Matrix G_s (shape {}):\n".format(Gs.shape), Gs)

def insert_random_columns(Gs, w):
    """
    Inserts w random column vectors into Gs to form G1.
    Gs: original generator matrix of shape (k, n)
    w: number of random columns to insert
    """
    k, n = Gs.shape
    GF = type(Gs)  # Extract the GF(2^m) type from the matrix

    # Copy Gs and convert to list of columns for easy insertion
    Gs_cols = [Gs[:, i] for i in range(n)]

    # Generate w random columns
    random_cols = [GF.Random(k) for _ in range(w)]

    # Choose w distinct random insertion positions
    insert_positions = np.sort(np.random.choice(range(n + w), size=w, replace=False))

    # Insert each random column at the correct position
    G1_cols = []
    ri = 0  # Index for random_cols
    gi = 0  # Index for Gs_cols

    for i in range(n + w):
        if ri < w and i == insert_positions[ri]:
            G1_cols.append(random_cols[ri])
            ri += 1
        else:
            G1_cols.append(Gs_cols[gi])
            gi += 1

    # Combine back into a matrix
    G1 = np.column_stack(G1_cols)
    return G1

# Example usage
w = 4  # Number of random columns to insert
G1 = insert_random_columns(Gs, w)

print("Extended Generator Matrix G1 (shape {}):\n".format(G1.shape), G1)

def generate_invertible_matrix(GF, size):
    """
    Generates a random invertible matrix of shape (size, size) over GF
    """
    while True:
        M = GF.Random((size, size))
        if np.linalg.det(M) != 0:
            return M

def block_diag_gf(*matrices):
    """
    Constructs a block diagonal matrix over GF from smaller square matrices.
    """
    total_rows = sum(m.shape[0] for m in matrices)
    total_cols = sum(m.shape[1] for m in matrices)
    GF = type(matrices[0])  # Assume all matrices use same GF
    result = GF.Zeros((total_rows, total_cols))

    r, c = 0, 0
    for m in matrices:
        rows, cols = m.shape
        result[r:r+rows, c:c+cols] = m
        r += rows
        c += cols
    return result

def generate_block_diagonal_A(GF, total_size, w):
    """
    Constructs block-diagonal matrix A:
    - Identity matrix of size (total_size - w)
    - Followed by w entries as 2x2 random invertible blocks
    """
    n_blocks = w // 2
    identity_size = total_size - w

    # Ensure integer dtype to match GF requirements
    A_blocks = [GF(np.eye(identity_size, dtype=int))]

    # Append random invertible 2x2 blocks
    for _ in range(n_blocks):
        while True:
            block = GF.Random((2, 2))
            if np.linalg.det(block) != 0:
                break
        A_blocks.append(block)

    # Combine into full matrix
    A = block_diag_gf(*A_blocks)
    return A

def generate_permutation_matrix(size):
    """
    Returns a permutation vector and matrix P of given size over integers
    """
    perm = np.random.permutation(size)
    P = np.eye(size, dtype=int)[perm]  # Now integer type, safe for GF
    return perm, P

# Use previous values
k, n_w = G1.shape  # G1 is k x (n+w)
GF = type(G1)

# Step 1: Random invertible S
S = generate_invertible_matrix(GF, k)

# Step 2: Block-diagonal A
w = n_w - Gs.shape[1]  # Number of inserted columns
A = generate_block_diagonal_A(GF, n_w, w)

# Step 3: Permutation matrix P2
perm2, P2 = generate_permutation_matrix(n_w)

# Final public key matrix G
G = S @ G1 @ A @ GF(P2)

# Output
print("Public Key Matrix G (shape {}):\n".format(G.shape), G)

def is_full_rank(G):
    """
    Returns True if G is full rank (i.e., rank = number of rows)
    """
    rank = np.linalg.matrix_rank(G.view(np.ndarray))  # Convert GF array to numpy array
    return rank == G.shape[0]

# Check if G is full rank
if is_full_rank(G):
    print("✅ Matrix G is full rank.")
else:
    print("❌ Matrix G is NOT full rank.")
    print("Rank of G:", np.linalg.matrix_rank(G.view(np.ndarray)))
    print("Expected (k):", G.shape[0])


def rlce_encrypt(GF, G, k, t):
    """
    Encrypt a random message m using public key G and sparse error vector e.
    - GF: the finite field (e.g., GF = galois.GF(2**4))
    - G: public key matrix of shape (k, n + w)
    - k: message length (number of rows of G)
    - t: number of errors to inject (Hamming weight of error vector)

    Returns:
    - m: the message vector
    - e: the error vector
    - c: the ciphertext vector
    """
    # Generate random message
    m = GF.Random(k)

    # Multiply message with public key
    c = m @ G  # shape: (n+w,)

    # Generate sparse error vector e of shape (n+w,)
    n_w = G.shape[1]
    e = GF.Zeros(n_w)

    # Choose t random error positions
    error_positions = np.random.choice(n_w, size=t, replace=False)
    e[error_positions] = GF.Random(t, low=1)  # Ensure non-zero errors

    # Add error vector to ciphertext
    c += e

    return m, e, c

# Parameters
t = 2  # Number of errors to inject

# Encrypt using public key G
m, e, c = rlce_encrypt(GF, G, k=G.shape[0], t=t)

print("Message m:", m)
print("Error vector e:", e)
print("Ciphertext c:", c)

def rlce_decrypt(GF, c, S, A, P2, Gs, t):
    """
    Decrypt an RLCE ciphertext using the private key components.

    Parameters:
    - GF: Galois field
    - c: Ciphertext vector (length n+w)
    - S: Random invertible matrix (k x k)
    - A: Block-diagonal matrix (n+w x n+w)
    - P2: Permutation vector (length n+w)
    - Gs: Original GRS generator matrix (k x n)
    - t: Max number of correctable errors

    Returns:
    - m_recovered: The recovered message (in GF)
    - e_recovered: The recovered error vector (in GF)
    """

    # Step 1: Undo P2 permutation
    P2_inv = np.argsort(P2)
    c_unpermuted = c[P2_inv]

    # Step 2: Undo A transformation
    A_inv = np.linalg.inv(A)
    c_A_reversed = c_unpermuted @ A_inv

    # Step 3: Trim G1-inserted columns (remove w random columns)
    # Assume original Gs had shape (k, n), so we remove w columns
    n = Gs.shape[1]
    n_w = c.shape[0]
    w = n_w - n

    # Identify indices of inserted columns (by comparing G1 to Gs shape)
    # For a real implementation, these indices must be stored or deduced
    # For simplicity here, we assume the inserted columns were at known indices
    # If you stored insert_positions during keygen, you should reuse them
    # For now, assume inserted positions were evenly spaced in the last `w` columns
    original_indices = np.sort(np.setdiff1d(np.arange(n_w), np.arange(n_w - w, n_w)))

    # Step 4: Extract part corresponding to Gs
    c_trimmed = c_A_reversed[original_indices]

    # Step 5: Decode GRS code (c_trimmed = m @ Gs + e_partial)
    try:
        # Use galois Reed-Solomon decoder
        RS = galois.ReedSolomon(Gs.shape[1], Gs.shape[0], field=GF)
        m_decoded = RS.decode(c_trimmed)
    except Exception as e:
        print("❌ GRS decoding failed:", e)
        return None, None

    # Step 6: Undo S transformation
    S_inv = np.linalg.inv(S)
    m_recovered = m_decoded @ S_inv

    # Step 7: Verify error vector
    # Re-encode and compare to trimmed c
    c_reconstructed = m_decoded @ Gs
    e_recovered = c_trimmed - c_reconstructed
    error_weight = np.count_nonzero(e_recovered)

    if error_weight > t:
        print(f"❌ Error weight {error_weight} exceeds threshold {t}")
        return None, None

    print("✅ Decryption successful.")
    return m_recovered, e_recovered

# Then decrypt:
m_recovered, e_recovered = rlce_decrypt(GF, c, S, A, P2, Gs, t=2)

print("Recovered Message:", m_recovered)
print("Recovered Error Vector:", e_recovered)