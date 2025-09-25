import numpy as np
import galois
<<<<<<< Updated upstream
from scipy.linalg import block_diag
=======

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
def validate_ciphertext(public_key: dict, m, y, t):
    """
    Verify that a decrypted message m is consistent with ciphertext y.
    This is done by checking if the residual error vector has Hamming weight <= t.
    """
    G=public_key["G"]
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

>>>>>>> Stashed changes

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
    
<<<<<<< Updated upstream
    public_key = S @ G1 @ A @ P
    private_key = {"S":S, "P":P, "A":A, "rs_code":rs_code, "GF":GF, "n":n, "r":r}
=======
    public_key = {"G":(S @ G1 @ A @ P), "n":n, "r":r, "k":k}
    private_key = {"S":S, "P":P, "A":A, "rs_code":rs_code, "GF":GF, "n":n, "r":r, "t":t, "G_pub":public_key}
>>>>>>> Stashed changes
    
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
    a = []
    for _ in range(n):
        a.append(generate_random_non_singular_matrix(r + 1, GF))
        
    A = GF(block_diag(*a))
    return A

def generate_random_non_singular_matrix(n, GF) -> galois.FieldArray:
    while True:
        # Generate matrix directly in the specified Galois Field
        matrix = GF.Random((n, n))
        # Use np.linalg.det, which is overloaded by the galois library
        if np.linalg.det(matrix) != 0:
            return matrix
        
def generate_random_permutation_matrix(n) -> np.ndarray:
    permutation_indices = np.random.permutation(n)
    identity_matrix = np.eye(n, dtype=int)
    permutation_matrix = identity_matrix[permutation_indices, :]
    return permutation_matrix


def encrypt(public_key: dict, message: np.ndarray, weight: int) -> galois.FieldArray:
    G=public_key["G"]
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
    
    # Get the inverse matrices
    S_inv = np.linalg.inv(S)
    P_inv = np.linalg.inv(P)
    A_inv = np.linalg.inv(A)
    
    intermediate_vector = ciphertext @ P_inv @ A_inv
    
    # Extract the parts of the vector corresponding to the original RS code
    rs_codeword_indices = [i * (r + 1) for i in range(n)]
    received_codeword = intermediate_vector[rs_codeword_indices]
    
    # Decode the extracted codeword to get the message (scrambled by S)
    decoded_message_scrambled = rs_code.decode(received_codeword, output="message")
    
    # Unscramble with the inverse of S to recover the original message
    decrypted_message = decoded_message_scrambled @ S_inv
    
    return decrypted_message


def get_row_space(matrix):
    rref_matrix=matrix.row_reduce()
    is_non_zero_row = np.any(rref_matrix, axis=1)
    basis = rref_matrix[is_non_zero_row]
    return basis

def get_schur_square_dimension(matrix: galois.FieldArray) -> int:
    """Calculates the dimension of the Schur square of the row space of the matrix."""
    basis_vector = get_row_space(matrix)
    num_basis_vectors = basis_vector.shape[0]
    
    generating_set = []
    for i in range(num_basis_vectors):
        for j in range(i, num_basis_vectors): # Start j from i to get unique pairs
            v_i = basis_vector[i]
            v_j = basis_vector[j]
            schur_product = v_i * v_j
            generating_set.append(schur_product)
    
    if not generating_set:
        return 0

    # FIX: Get the specific Galois Field class from the input matrix
    GF = type(matrix)
    # Instantiate the schur_matrix using the correct Field class
    schur_matrix = GF(generating_set)
    
    dim_ref = np.linalg.matrix_rank(schur_matrix)
    return dim_ref

def crack_permutation(public_key: dict) -> galois.FieldArray:
    G=public_key["G"]
    total_columns = G.shape[1]
    
    dim_ref = get_schur_square_dimension(G)
    goppa_indices = []
    #foreach column in public_key["G"]
    for j in range(total_columns):
        g_punctured = np.delete(G, j, axis=1)
        dim_ref_punctured = get_schur_square_dimension(g_punctured)
        if(dim_ref_punctured==dim_ref):
            goppa_indices.append(j)
    
    n=public_key["n"]
    if len(goppa_indices) != n:
        raise ValueError(f"Attack failed: Expected to find {n} columns, but found {len(goppa_indices)}.")
    
    
    G_real = G[:, goppa_indices]
    
    return G_real
    





if __name__ == "__main__":
    
    # Use smaller, faster parameters for the demonstration
    n = 15  # Code length (must be <= GF order - 1)
    k = 7   # Message dimension
    t = 4   # Error correction capability (t <= (n-k)/2)
    r = 1   # Dimension of random matrices
    
    # 1. Generate a valid public/private key pair
    print("--- KEY GENERATION ---")
    public_key, private_key = KeyGen(n=n, k=k, t=t, r=r)
    GF = private_key["GF"]
    print(f"Keys generated for parameters n={n}, k={k}, t={t}, r={r} over {GF.name}")
    print(f"Public key matrix shape: {public_key['G'].shape}\n")
    
    # 2. Encrypt and Decrypt Normally (Control Group)
    print("--- NORMAL OPERATION (CONTROL) ---")
    message_to_encrypt = GF.Random(k)
    ciphertext = encrypt(public_key, message_to_encrypt, weight=t)
    decrypted_message = decrypt(private_key, ciphertext)
    is_correct = np.array_equal(message_to_encrypt, decrypted_message)
    print("Original Message: ", message_to_encrypt)
    print("Decrypted Message:", decrypted_message)
    print(f"Success: {is_correct} {'✅' if is_correct else '❌'}\n")
    
<<<<<<< Updated upstream
    print("\n--- McEliece Cryptosystem Verification ---")
    print("Original Message: ", message_to_encrypt)
    print("Decrypted Message:", decrypted_message)
    print("Success:", is_correct, "✅" if is_correct else "❌")
=======
    # 3. --- NEW: SIMULATE ATTACK ---
    print("--- ATTACK SIMULATION ---")
    print("Step 1: Cracking permutation P to find G_real...")
    try:
        # Run the attack to get the unscrambled, real generator matrix
        G_real = crack_permutation(public_key)

        print("\nStep 2: Verifying the cracked key G_real...")
        # Create a new, vulnerable code from the cracked matrix
        # galois.LinearCode is more general than ReedSolomon for this purpose
        cracked_code = galois.LinearCode(G_real)
        
        # Create a new message to test the cracked code
        message_for_attack = GF.Random(k)
        print("New test message:", message_for_attack)

        # Encode the message using the cracked matrix
        codeword = message_for_attack @ G_real
        
        # Create an error vector with weight t for the shorter codeword
        error = GF.Zeros(n)
        error_indices = np.random.choice(n, t, replace=False)
        error[error_indices] = GF.Random(t, low=1)
        
        noisy_codeword = codeword + error
        print(f"Created a noisy codeword with {t} errors.")
        
        # Decode using the new, vulnerable code
        decoded_attack_message = cracked_code.decode(noisy_codeword, output="message")
        print("Decoded message: ", decoded_attack_message)

        # Verify the result
        is_attack_correct = np.array_equal(message_for_attack, decoded_attack_message)
        print(f"Attack verification success: {is_attack_correct} {'✅' if is_attack_correct else '❌'}")

    except ValueError as e:
        print(f"\nAttack failed: {e} ❌")
>>>>>>> Stashed changes
