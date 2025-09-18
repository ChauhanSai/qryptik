import numpy as np
import galois

# ---------- NEW: Gauss–Jordan inverse ----------
def gf_matrix_inverse(A: "galois.FieldArray") -> "galois.FieldArray":
    """
    Invert a square matrix A over a galois FieldArray using Gauss–Jordan.
    Safer than np.linalg.inv for finite fields because it avoids float det issues.
    """
    GF = type(A)
    n = A.shape[0]
    # Build augmented matrix [A | I]
    Aug = np.concatenate([A.copy(), GF(np.eye(n, dtype=int))], axis=1)
    for col in range(n):
        # Find pivot
        pivot = None
        for r in range(col, n):
            if Aug[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            raise np.linalg.LinAlgError("Singular matrix in GF")
        # Swap if needed
        if pivot != col:
            Aug[[col, pivot]] = Aug[[pivot, col]]
        # Normalize pivot row
        piv = Aug[col, col]
        Aug[col] = Aug[col] / piv
        # Eliminate other rows
        for r in range(n):
            if r != col:
                factor = Aug[r, col]
                Aug[r] = Aug[r] - factor * Aug[col]
    return Aug[:, n:]

# ---------- ciphertext validity check ----------
def validate_ciphertext(G, m, y, t):
    """
    Verify that a decrypted message m is consistent with ciphertext y.
    This is done by checking if the residual error vector has Hamming weight <= t.
    """
    if m.ndim == 1:
        m = m.reshape(1, -1)

    # Compute the expected clean codeword c = mG using the public key generator matrix G.
    # Then compute residual = y - c, which represents the error vector introduced during encryption.
    resid = y - (m @ G)

    # Count the number of nonzero entries in the residual vector.
    # This is the Hamming weight of the error vector 
    weight = int(np.count_nonzero(resid != 0))

    # If the Hamming weight exceeds the error-correcting capability t,
    # then the ciphertext is invalid (too many errors).
    if weight > t:
        raise ValueError("Ciphertext invalid (residual weight > t)")

    # Otherwise, the ciphertext is consistent and valid.


def KeyGen(n, k, t, r) -> tuple[galois.FieldArray, dict]: #returns (public key, private key)
    #n = code length
    #k = message dimension
    #t = error correction capability
    #r = dimension of random matrices (r>=1)
    
    rs_code = generate_rs_code(k, n) #generating reed-solomon code
    GF = rs_code.field #finite field for the code
    
    Gs = rs_code.G #reed-solomon generator matrix
    G1 = generate_G1(Gs, r, GF) #inserting r random columns
    A = generate_A(n, r, GF) #block diagonal matrix of random matrices
    S = generate_random_non_singular_matrix(k, GF) #random non-singular matrix
    P = GF(generate_random_permutation_matrix(n*(r+1))) #random permutation matrix
    
    public_key = S @ G1 @ A @ P
    private_key = {"S":S, "P":P, "A":A, "rs_code":rs_code, "GF":GF, "n":n, "r":r, "t":t, "G_pub":public_key}
    
    return (public_key, private_key)

def generate_rs_code(k, n) -> galois.ReedSolomon:
    GF = galois.GF(n+1)
    rs_code = galois.ReedSolomon(n=n, k=k, field=GF)
    return rs_code
    
def generate_G1(Gs: galois.FieldArray, r: int, GF: type[galois.FieldArray]) -> galois.FieldArray:
    k = Gs.shape[0]
    n = Gs.shape[1]
    columns_to_stack = []
    
    for i in range(n):
        columns_to_stack.append(Gs[:, [i]])
        
        for _ in range(r):
            # Convert the random column to a GF array
            column = GF.Random((k, 1))
            columns_to_stack.append(column)
            
    G1 = np.hstack(columns_to_stack)
    return G1

def generate_A(n: int, r: int, GF: type[galois.FieldArray]) -> galois.FieldArray:
    blocks = [generate_random_non_singular_matrix(r + 1, GF) for _ in range(n)]
    A = np.block([
        [blocks[i] if i == j else GF.Zeros((r+1, r+1)) for j in range(n)]
        for i in range(n)
    ])

    return GF(A)

# ---------- MODIFIED: random invertible matrix ----------
def generate_random_non_singular_matrix(n, GF) -> galois.FieldArray:
    """
    Generate a random invertible matrix over GF using Gauss–Jordan to test invertibility.
    Replaces np.linalg.det which can misbehave in finite fields.
    """
    while True:
        M = GF.Random((n, n))
        try:
            _ = gf_matrix_inverse(M)  # Check invertibility
            return M
        except np.linalg.LinAlgError:
            continue
        
def generate_random_permutation_matrix(n) -> np.ndarray:
    permutation_indices = np.random.permutation(n)
    identity_matrix = np.eye(n, dtype=int)
    permutation_matrix = identity_matrix[permutation_indices, :]
    return permutation_matrix


def encrypt(G: galois.FieldArray, message: np.ndarray, weight: int) -> galois.FieldArray:
    GF = type(G) # Get the finite field from the generator matrix
    
    e = GF.Zeros(G.shape[1])
    # Generate an error vector with a specified weight
    one_indices = np.random.choice(G.shape[1], size=weight, replace=False)
    # The error values can be any non-zero element from the field
    error_values = GF.Random(weight, low=1)
    e[one_indices] = error_values
    
    ciphertext = (GF(message) @ G + e)
    
    return ciphertext

def decrypt(private_key: dict, ciphertext: galois.FieldArray) -> galois.FieldArray:
    #unpack private key
    S = private_key["S"]
    P = private_key["P"]
    A = private_key["A"]
    rs_code = private_key["rs_code"]
    n = private_key["n"]
    r = private_key["r"]
    t = private_key["t"]
    G_pub = private_key["G_pub"]
    
    # Use Gauss–Jordan inverse instead of np.linalg.inv
    S_inv = gf_matrix_inverse(S)
    P_inv = gf_matrix_inverse(P)
    A_inv = gf_matrix_inverse(A)
    
    intermediate_vector = ciphertext @ P_inv @ A_inv
    
    # Extract the parts of the vector corresponding to the original RS code
    rs_codeword_indices = [i * (r + 1) for i in range(n)]
    received_codeword = intermediate_vector[rs_codeword_indices]
    
    # Decode the extracted codeword to get the message (scrambled by S)
    decoded_message_scrambled = rs_code.decode(received_codeword, output="message")
    
    # Unscramble with the inverse of S to recover the original message
    decrypted_message = decoded_message_scrambled @ S_inv
    
    # ---------- Call ciphertext validation ----------
    validate_ciphertext(G_pub, decrypted_message, ciphertext, t)
    
    return decrypted_message


if __name__ == "__main__":
    
    # Define system parameters (valid for Reed-Solomon over GF(16))
    n = 255  # Code length
    k = 235   # Message dimension
    t = 10   # Error correction capability (t <= (n-k)/2)
    r = 1   # Dimension of random matrices
    
    # 1. Generate a valid public/private key pair
    public_key, private_key = KeyGen(n=n, k=k, t=t, r=r)
    
    # 2. Define the message (must be k symbols long from GF(16))
    GF = private_key["GF"]
    message_to_encrypt = GF.Random(k) # Create a random message in the correct field
    
    # 3. Encrypt the message
    ciphertext = encrypt(public_key, message_to_encrypt, weight=t)
    
    # 4. Decrypt the message
    decrypted_message = decrypt(private_key, ciphertext)
    
    # 5. Verify the result
    is_correct = np.array_equal(message_to_encrypt, decrypted_message)
    
    print("\n McEliece Cryptosystem Verification  final demo.py:204 - main.py:204")
    print("Original Message:  final demo.py:205 - main.py:205", message_to_encrypt)
    print("Decrypted Message:  final demo.py:206 - main.py:206", decrypted_message)
    print("Success:  final demo.py:207 - main.py:207", is_correct, "✅" if is_correct else "❌")

'''

method to try and crack A within amalgamated code -->

'''

def left_inverse(matrix: galois.FieldArray) -> galois.FieldArray:
    """
    Compute the left inverse (Moore-Penrose pseudo-inverse) of a matrix over GF.
    Only works if matrix has full column rank.
    """
    GF = type(matrix)
    gram = matrix.T @ matrix
    gram_inv = gf_matrix_inverse(gram)
    return gram_inv @ matrix.T

def crack_A(public_key: galois.FieldArray, private_key: dict) -> galois.FieldArray:
    """
    Recover the block diagonal matrix A from the public key and private key components.
    """
    S = private_key["S"]
    P = private_key["P"]
    G1 = generate_G1(private_key["rs_code"].G, private_key["r"], private_key["GF"])
    n = private_key["n"]
    r = private_key["r"]
    GF = private_key["GF"]

    # Invert S and P
    S_inv = gf_matrix_inverse(S)
    P_inv = gf_matrix_inverse(P)

    # Compute intermediate matrix M = S_inv * public_key * P_inv
    M = S_inv @ public_key @ P_inv

    block_size = r + 1
    A_blocks = []

    for i in range(n):
        col_start = i * block_size
        col_end = (i + 1) * block_size

        G1_block = G1[:, col_start:col_end]  # k x (r+1)
        M_block = M[:, col_start:col_end]    # k x (r+1)

        # Compute left inverse of G1_block
        try:
            G1_block_left_inv = left_inverse(G1_block)
        except np.linalg.LinAlgError:
            raise ValueError(f"Cannot invert Gram matrix for block {i}. Rank deficiency.")

        # Recover block A_i
        A_i = G1_block_left_inv @ M_block
        A_blocks.append(A_i)

    # Assemble block diagonal matrix A from blocks
    A_recovered = GF(np.block([
        [A_blocks[i] if i == j else GF.Zeros((block_size, block_size)) for j in range(n)]
        for i in range(n)
    ]))

    return A_recovered


if __name__ == "__main__":
    # ... existing code above ...
    
    # --- CRACK A TEST ---
    print("\nAttempting to recover A from the public key...")
    A_recovered = crack_A(public_key, private_key)
    
    # Check if recovered A matches original A
    A_original = private_key["A"]
    is_A_correct = np.array_equal(A_recovered, A_original)
    print("Recovered A matches original A:", is_A_correct, "✅" if is_A_correct else "❌")

'''
IGNORE BELOW CODE (failed method)


def crack_A(public_key: galois.FieldArray, S: galois.FieldArray, G1: galois.FieldArray, P: galois.FieldArray, n: int, r: int, GF: type[galois.FieldArray]) -> galois.FieldArray:
    """
    Given public_key = S @ G1 @ A @ P,
    recover A by computing A = (S^-1) @ public_key @ (P^-1) @ (G1^-1).
    
    Since G1 is not square, we approximate G1^-1 by using its left-inverse
    (assuming full row rank), which is (G1.T @ G1)^-1 @ G1.T.
    """
    # Compute inverses of S and P
    S_inv = gf_matrix_inverse(S)
    P_inv = gf_matrix_inverse(P)
    
    # Compute left-inverse of G1 (Moore-Penrose pseudo-inverse in GF)
    G1T = G1.T
    try:
        temp = gf_matrix_inverse(G1T @ G1)
    except np.linalg.LinAlgError:
        raise ValueError("Cannot invert G1^T G1, attack failed.")
    G1_left_inv = temp @ G1T
    
    # Estimate A
    A_estimated = S_inv @ public_key @ P_inv @ G1_left_inv
    
    return A_estimated

# ---------- Test cracking A ----------
if __name__ == "__main__":
    print("\n--- Attempting to crack A matrix ---")

    # Extract known components from private key
    S = private_key["S"]
    P = private_key["P"]
    A = private_key["A"]  # Actual A, for comparison
    rs_code = private_key["rs_code"]
    GF = private_key["GF"]
    n = private_key["n"]
    r = private_key["r"]
    
    Gs = rs_code.G
    G1 = generate_G1(Gs, r, GF)
    
    # Try to recover A
    A_cracked = crack_A(public_key, S, G1, P, n, r, GF)
    
    # Check if cracked A matches the original
    match = np.array_equal(A, A_cracked)
    
    print("Original A and Cracked A match:", match, "✅" if match else "❌")

'''