import numpy as np
import galois
import random

m = 8
q = 2 ** m
GF = galois.GF(q)

def printf(p, color):
    """
    Custom print function to format output with color.
    :param p: The message to print.
    :param color: Color code for terminal output.
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors[color]}{p}{colors['reset']}\n")

def generate_gs(n, k):
    """
    Constructs a k × n generator matrix for a [n, k, d] generalized Reed-Solomon code.

    :param n: Code length
    :param k: Code dimension
    :return: Generator matrix G_s
    """
    assert n <= q, "n must be less than or equal to the size of the field"

    # Create the finite field GF(q)

    # Choose n distinct evaluation points (α_i) in the field
    alpha = GF.Range(0, n)  # or GF.list[:n]

    # Choose n nonzero scale factors (v_i)
    v = GF.Random(n)
    v[v == 0] = 1  # Ensure all v_i are non-zero

    # Construct the Vandermonde matrix V of shape (k, n)
    V = GF.Zeros((k, n))
    for i in range(k):
        V[i, :] = alpha ** i

    # Multiply each column of V by the corresponding v_i to get GRS generator matrix
    Gs = V * v

    return Gs

def generate_p(n):
    """
    Generates a random permutation matrix P_1 of size n x n.

    :param n: Size of the permutation matrix
    :return: Permutation matrix P_1
    """
    # Create a permutation of indices
    indices = np.random.permutation(n)

    # Create an identity matrix and permute its rows according to indices
    P = np.eye(n)[indices]

    return P


def get_GF_non_singular_matrix():
    while True:
        matrix = GF.Random((2, 2))  # Generate a random 2x2 matrix in GF(q)
        if GF(np.linalg.det(matrix)) != 0:  # Check if the determinant is non-zero in GF(q)
            product = GF(1)  # Start with the multiplicative identity in GF
            for element in matrix.flatten():
                product *= element
                if product != 0:
                    return matrix.view(np.ndarray)  # Convert to NumPy array


def build_block_diag_matrix(n, w, A_list):
    from scipy.linalg import block_diag
    """
    Constructs a block diagonal matrix A = diag[I_{n-w}, A_0, ..., A_{w-1}]

    Parameters:
        n (int): total number of codeword columns
        w (int): number of 2x2 random matrices
        A_list (list of np.ndarray): list of w non-singular 2x2 matrices

    Returns:
        A (np.ndarray): (n + w) x (n + w) block diagonal matrix
    """
    assert len(A_list) == w, "You must provide exactly w 2x2 matrices"
    for A_i in A_list:
        assert A_i.shape == (2, 2), "Each A_i must be 2x2"

    I = np.eye(n - w)
    A = block_diag(I, *A_list)  # dynamically append all A_i matrices
    return A

def keySetup(n, k, d, t, w):
    """
    Let n, k, d, t > 0, and w ∈ {1,···,n} be given parameters such that n−k +1 ≥ d.
    Generally we have d ≥ 2t + 1,
    though it is allowed to have d < 2t + 1
    in case that eﬃcient list-decoding algorithms exist.
    :param n:
    :param k:
    :param d:
    :param t:
    :param w:
    :return:
    """
    print(f"n: {n}, k: {k}, d: {d}, t: {t}, w: {w}")
    assert (n - k + 1 >= d), f"{n-k+1} ≥ {d}"
    print(n-k+1, "≥", d)
    assert (d >= 2*t + 1), f"{d} ≥ {2*t+1}"
    print(d, "≥", 2*t+1)
    print(f"q: {q}")
    assert (k < n < q), f"{k} < {n} < {q}"
    print(k, "<", n, "<", q)
    print()

    G_s = generate_gs(n, k)
    print("Generator matrix G_s:")
    # Convert Galois GFArray to NumPy array
    G_s_np = G_s.view(np.ndarray)
    printf(G_s_np, 'red')

    P_1 = generate_p(n)
    print("Permutation P_1:")
    printf(P_1, 'red')

    # Multiply G_s by P_1
    G_s_P1 = G_s_np @ P_1
    print("Result of G_s * P_1:")
    printf(G_s_P1, 'green')

    # Draw w columns from GF(q)^k
    r = np.ndarray(shape=(w, k), dtype=object)
    for instance in range(w):
        GF_q_k_instance = GF.Random(k)  # Generate a random vector of size k in GF(q)
        r[instance, :] = GF_q_k_instance.view(np.ndarray) ** k
    print("w column vectors (r):")
    printf(r.transpose(), 'yellow')

    assert (n - w >= 0), f"n-w ({n-w}) ≥ 0"
    G_1 = np.ndarray(shape=(k, n+w), dtype=int)
    G_1[:, 0:n-w] = G_s_P1[:, 0:n-w]  # Copy the first n-w columns
    print("G_1:")
    printf(G_1, 'yellow')
    i = 0
    j = n-w
    while j < n+w:
        G_1[:, j] = G_s_P1[:, n - w + i]  # Copy the first n-w columns
        G_1[:, j+1] = r.transpose()[:, i]
        print(j)
        printf(G_1, 'magenta')
        j+=2
        i+=1

    print("Final G_1:")
    printf(G_1, 'blue')

    # Create A matrix
    print("Non-Singular Matrices (A_i):")
    A_list = []
    for i in range(0, w):
        A_list.append(get_GF_non_singular_matrix())
    printf(A_list, 'cyan')

    A = build_block_diag_matrix(n, w, A_list)
    print("Block Diagonal Matrix A:")
    printf(A, 'cyan')
    assert (A.shape == (n + w, n + w)), f"A shape: {A.shape}, expected: {(n + w, n + w)}"

    # Generate S
    while True:
        S = np.random.randint(1, 10, size=(k, k)) # Generate random integers between 1 and 9
        if np.linalg.det(S) != 0:  # Check if the determinant is non-zero
            break
    print("Matrix S:")
    printf(S, 'green')

    P_2 = generate_p(n + w)
    print("Permutation P_2:")
    printf(P_2, 'green')

    # Compute G
    G = S @ G_1 @ A @ P_2
    print("Public Key G:")
    printf(G, 'green')

    return G, (S, G_s.view(np.ndarray), P_1, P_2, A), G_1


def generate_error_vector(n, w, t, GF):
    """
    Generates a row vector e ∈ GF(q)^(n+w) with Hamming weight wt(e) ≤ t.

    :param n: Length of the codeword
    :param w: Additional columns
    :param t: Maximum Hamming weight
    :param GF: Galois Field object
    :return: Row vector e
    """
    length = n + w
    e = GF.Zeros(length)  # Initialize a zero vector in GF(q)

    # Randomly choose t positions to assign non-zero values
    positions = np.random.choice(length, t, replace=False)
    for pos in positions:
        e[pos] = GF.Random(1)[0]  # Assign a random non-zero value from GF(q)

    return e.view(np.ndarray)  # Convert to NumPy array


def string_to_field_vector(s):
    byte_vals = [ord(c) for c in s]  # Convert characters to ASCII
    return GF(byte_vals)  # Convert to field elements


def field_vector_to_string(v):
    return ''.join(chr(int(x)) for x in v)


def prepare_message(s, k):
    m_vec = string_to_field_vector(s)
    if len(m_vec) > k:
        m_vec = m_vec[:k]
    elif len(m_vec) < k:
        pad = GF.Zeros(k - len(m_vec))
        m_vec = np.hstack((m_vec, pad))
    return m_vec.view(np.ndarray)  # Convert to NumPy array


def enc(G, m, e):
    print("Error vector e:")
    print(e)

    print("Message vector m:")
    print(m)

    c = (m @ G) + e

    return c


def dec(S, G_s, P_1, P_2, A, c, e, G_1):
    c_prime = c @ np.linalg.inv(P_2) @ np.linalg.inv(A)
    selected = list(c_prime[:n - w])

    # For each 2x2 block, select the first element
    for i in range(w):
        block_start = n - w + 2 * i
        selected.append(c_prime[block_start])

    c_prime_prime = selected @ np.linalg.inv(P_1)

    RS = galois.ReedSolomon(n, k, field=GF)
    c_prime_prime_GF = GF(c_prime_prime.astype(int) % 256)
    # Decode the received vector
    decoded = RS.decode(c_prime_prime_GF)

    print(decoded)

    G_s_prime = G_s[:, :k].view(np.ndarray)  # Convert G_s to NumPy array
    print(G_s_prime)
    D = np.linalg.inv(S @ G_s_prime)
    c_1 = decoded[:k]  # Extract the first k elements
    m = c_1.view(np.ndarray) @ D
    print(m)

    m_prime = m[:k] % 255
    print(m_prime)

    return m_prime


if __name__ == "__main__":
    t = 4
    d = 2*t + 1
    k = round(d/4)
    n = 15 # d + k - 1
    w = random.randint(1, n)  # Randomly choose w in the range [1, n]

    publicKey, privateKey, G_1 = keySetup(n, k, d, t, w)

    print("Public Key:")
    printf(publicKey, 'blue')

    print("Private Key:")
    printf(privateKey, 'blue')

    e = generate_error_vector(n, w, t, GF)
    m = prepare_message("Test", k)
    c = enc(publicKey, m, e)

    print("Ciphertext:")
    printf(c, 'green')

    S, G_s, P_1, P_2, A = privateKey
    m_dec = dec(S, G_s, P_1, P_2, A, c, e, G_1)
    print("Decrypted message vector:")
    printf(m_dec, 'red')
    print("Decrypted message string:")
    printf(field_vector_to_string(m_dec), 'red')
