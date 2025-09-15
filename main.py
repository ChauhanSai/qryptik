import numpy as np
import galois
from scipy.linalg import block_diag

def KeyGen(n, k, t, r) -> tuple: #returns (public key, private key)
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
    private_key = {"S":S, "P":P, "A":A, "rs_code":rs_code, "GF":GF, "n":n, "r":r}
    
    return (public_key, private_key)

def generate_rs_code(k, n) -> galois.FieldArray:
    

    GF = galois.GF(n+1)
    rs_code = galois.ReedSolomon(n=n, k=k, field=GF)
    return rs_code
    
def generate_G1(Gs, r, GF) -> np.ndarray:
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

def generate_A(n, r, GF) -> np.ndarray:
    a = []
    for _ in range(n):
        a.append(generate_random_non_singular_matrix(r + 1, GF))
        
    A = GF(block_diag(*a))
    return A

def generate_random_non_singular_matrix(n, GF) -> np.ndarray:
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


def encrypt(G, message: np.ndarray, weight: int) -> np.ndarray:
    GF = type(G) # Get the finite field from the generator matrix
    
    e = GF.Zeros(G.shape[1])
    # Generate an error vector with a specified weight
    one_indices = np.random.choice(G.shape[1], size=weight, replace=False)
    # The error values can be any non-zero element from the field
    error_values = GF.Random(weight, low=1)
    e[one_indices] = error_values
    
    ciphertext = (GF(message) @ G + e)
    
    return ciphertext

def decrypt(private_key, ciphertext) -> np.ndarray: # returns message
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
    
    print("\n--- McEliece Cryptosystem Verification ---")
    print("Original Message: ", message_to_encrypt)
    print("Decrypted Message:", decrypted_message)
    print("Success:", is_correct, "✅" if is_correct else "❌")