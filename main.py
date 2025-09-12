import numpy as np
from numpy.linalg import det
from scipy.linalg import block_diag
def KeyGen(n, k, t, r): #returns (public key, private key)
    #n = code length
    #k = message dimension
    #t = error correction capability
    #r = dimension of random matrices (r>=1)
    
    Gs = np.random.randint(0, 2, size=(k, n))
    G1 = generate_G1(Gs, r)
    
    A = generate_A(n, r); #block diagonal matrix of random matrices

    S = generate_random_non_singular_matrix(k); #random non-singular matrix
    P = generate_random_permutation_matrix(n*(r+1)); #random permutation matrix
    
    public_key = S @ G1 @ A @ P
    private_key = {"S":S, "Gs":Gs, "P":P, "A":A}
    
    return (public_key, private_key)

def generate_G1(Gs, r):
    k = Gs.shape[0]
    n = Gs.shape[1]
    columns_to_stack = []
    
    # foreach column in Gs
    for i in range(n):
        # append Gs column to stack
        columns_to_stack.append(Gs[:, [i]])
        
        # append r random columns, each of size k x 1
        for _ in range(r):
            column = np.random.randint(0, 2, size=(k, 1))
            columns_to_stack.append(column)
            
    G1 = np.hstack(columns_to_stack)
    return G1

def generate_A(n, r):
    a = []
    for _ in range(n):
        a.append(generate_random_non_singular_matrix(r + 1))
        
    A = block_diag(*a)
    return A

def generate_random_non_singular_matrix(n):
    while True:
        matrix = np.random.randint(0, 2, size=(n, n))  # Generate a random matrix
        if np.linalg.det(matrix) != 0:  # Check if determinant is non-zero
            return matrix
        
def generate_random_permutation_matrix(n):
    permutation_indices = np.random.permutation(n)
    identity_matrix = np.eye(n)
    permutation_matrix = identity_matrix[permutation_indices, :]
    return permutation_matrix


def encrypt(public_key, message) -> str: # returns ciphertext
    return ""

def decrypt(private_key, ciphertext) -> str: # returns message
    return ""


print(KeyGen(4, 8, 1, 1))