import numpy as np
import galois

# Set up small finite field
GF = galois.GF(2**4)

# Toy parameters
n, k, w = 10, 6, 2
t = 2  # Max number of errors
m = GF.degree

# Generate a GRS generator matrix
x = GF(np.random.choice(GF.elements, n, replace=False))  # Ensure unique values
y = GF(np.random.choice(GF.elements[1:], n, replace=False))  # Ensure non-zero unique values
def vandermonde(x, k):
    return np.vstack([x**i for i in range(k)]).T

Gs = vandermonde(x, k) * y.reshape(-1, 1)  # Reshape y to align dimensions for element-wise multiplication

# Generate private matrices
P1 = np.random.permutation(k)  # Restrict permutation to valid column indices of Gs
P2 = np.random.permutation(n)  # Restrict permutation to valid column indices of A
A_blocks = [GF.Random((2, 2)) for _ in range(w//2)]
A = np.block([[np.eye(n-w)] + [np.zeros((n-w, 2))]] + [[np.zeros((2, n-w))] + A_blocks])  # Align dimensions for concatenation
A_inv = np.linalg.inv(A)

# Random matrix R to add noise columns
R = GF.Random((n, w))  # Adjust dimensions of R to match the number of rows in Gs
G1 = np.hstack((Gs[:, P1], R))  # Concatenate horizontally

S = GF.Random((k, k))

# Adjust A to match dimensions for matrix multiplication
A_adjusted = A[:k, :n]  # Select only the first k rows and n columns of A to match G1's dimensions
G = S @ G1 @ A_adjusted.T  # Transpose A_adjusted to align dimensions for matrix multiplication

# Public key: G, Private key: (S, Gs, P1, P2, A)
def encrypt(message):
    m_vec = GF(message)
    e = GF.Zeros(n + w)
    error_indices = np.random.choice(n + w, size=t, replace=False)
    e[error_indices] = GF.Random(t)
    c = m_vec @ G + e
    return c, e

def decrypt(c):
    c = GF(c)
    cp = c @ np.linalg.inv(A)[:, np.argsort(P2)]
    cp_trimmed = np.delete(cp, np.s_[n-w::2])  # Remove inserted noise
    c_prime = cp_trimmed[np.argsort(P1)]
    # Decode GRS using syndrome decoding (simplified here)
    # For toy example, assume we can recover m perfectly
    m_recovered = c_prime[:k] @ np.linalg.inv(S)
    return m_recovered

# Try a message
m = GF.Random(k)
ciphertext, err = encrypt(m)
decrypted = decrypt(ciphertext)

print("Original message: ", m)
print("Ciphertext: ", ciphertext)
print("Recovered: ", decrypted)
print("Success:", np.array_equal(m, decrypted))

