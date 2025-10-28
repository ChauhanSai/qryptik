import numpy as np #imports Numpy for arrays/matric ops
import galois #imports galois for finite-field math

GF = galois.GF(2**6); #creates a finite field where we will build all the matrices/vectors
rng = np.random.default_rng() #a random number generator we'll use for sampling positions and randomness

# --- helper: determinant over GF (minimal Gaussian elimination) ---
def _det_gf(M):
    A = M.copy()
    n = A.shape[0]
    det = GF(1)
    for i in range(n):
        pivot = None
        for r in range(i, n):
            if A[r, i] != 0:
                pivot = r; break
        if pivot is None:
            return GF(0)
        if pivot != i:
            A[[i, pivot], :] = A[[pivot, i], :]
            # sign flip is irrelevant in characteristic 2, but keep flow identical
        det *= A[i, i]
        inv_pivot = GF(1) / A[i, i]
        for r in range(i + 1, n):
            factor = A[r, i] * inv_pivot
            if factor != 0:
                A[r, i:] -= factor * A[i, i:]
    return det

    # --- helper: matrix inverse over GF (Gauss–Jordan) ---
def _inv_gf(M):
    A = M.copy()
    n = A.shape[0]
    I = GF.Zeros((n, n))
    for i in range(n):
        I[i, i] = GF(1)
    # forward + backward elimination
    for c in range(n):
        # find pivot
        p = None
        for r in range(c, n):
            if A[r, c] != 0:
                p = r; break
        if p is None:
            raise ValueError("Matrix is singular in GF")
        if p != c:
            A[[c, p], :] = A[[p, c], :]
            I[[c, p], :] = I[[p, c], :]
        # scale pivot row to make pivot = 1
        inv_piv = GF(1) / A[c, c]
        A[c, :] *= inv_piv
        I[c, :] *= inv_piv
        # eliminate other rows
        for r in range(n):
            if r == c: continue
            f = A[r, c]
            if f != 0:
                A[r, :] -= f * A[c, :]
                I[r, :] -= f * I[c, :]
    return I


def random_invertible_square(n, *, GF=GF):
    while True:
        M = GF.Random((n,n));  #makes a random matrix in the field
        if (_det_gf(M) != 0): #computes a determinant in the field and returns a guaranteed invertible matrix
            return M;

def premutation_matrix(n, *, GF=GF): #build a random permutation matrix
    perm = np.arange(n); 
    rng.shuffle(perm);
    P = GF.Zeros((n,n));
    P[np.arange(n), perm] = 1; #creates a n x n permutation matrix with 1s at (i,perm[i]) and 0s elsewhere
    return P, perm; #returns both P and the index array perm

def block_diag(blocks, *, GF=GF): #Takes a list of square matrices and places them on the diagonal of a larger matrix
    total = sum(b.shape[0] for b in blocks);
    A = GF.Zeros((total, total));
    offset = 0;
    #Fills zero everywhere, then copies each B into the right diagonal window
    for B in blocks:
        r = B.shape[0];
        A[offset:offset+r, offset:offset+r] = B;
        offset += r
    return A;

def hamming_weight(vec): #counts how many entries in the vector are non-zero
    #Hamming weight of a 1xN row vector over GF
    return int(np.count_nonzero(vec != 0));

#example parameters
n , k = 63, 31 #base Reed-solomon code length n and dimension k
t = (n-k) // 2; #Error - correcting capability of RS: t = n-k/2
r = 1; #RLCE expansion factor. Each original column becomes a block of r+1 columns
qN = n * (r+1) #public key length after expansion(number of clumns in RLCE public generator)

#base code and its generator
rs = galois.ReedSolomon(n,k, field = GF); #gives us encode, decode and generator matrix
Gs = rs.G; #The RS generator matrix G8 of shape k x n

#RLCE key generation
Ci_list = [GF.Random((k,r)) for _ in range(n)]; #For each original RS column, we'll append r random colums. Precompute all of them
blocks = [];
for i in range(n):
    gi = Gs[:,[i]]; #gi is the i-th column of G8 as a k x 1 matrix
    Ci = Ci_list[i]; #random k x r block we attatch to that column
    blocks.append(np.concatenate([gi, Ci], axis = 1)); #Concatenate horizontally to make a k x (r+1) block for position i and append that block to a list

G1 = np.concatenate(blocks, axis = 1); #concatenate all n blocks horizontally to form G1​∈GFk×n(r+1)

#Local mixing A
A_blocks = [random_invertible_square(r+1) for _ in range(n)]; #for each expanded column block, make an independent (r+1) x (r+1) random invertible matrix
A = block_diag(A_blocks); #combine those into a big block diagonal matrix

#Global scarmbling
S = random_invertible_square(k); #a invertible k x k matrix to scramble rows
P, _ = premutation_matrix(qN); # A random permutation matrix of size qN to shuffle columns globally

#public key
G_pub = S @ G1 @ A @ P; #computes the public generator matrix
priv = {"S" : S, "Gs": Gs, "A": A, "P": P, "rs": rs, "n": n, "k": k, "r": r, "t": t}; #packs all secret perices and parameters we'll need for decryption

#Encryption
def rlce_encrypt(m_row, G_pub, t):
    assert m_row.shape == (1, G_pub.shape[0]); #checks whether message width matches the number of rows
    qN = G_pub.shape[1]; #get public length(number of columns)
    y = m_row @ G_pub; #compute the codeword mG over GF
    
    #Add a sparse error of Hamming weight exactly t
    positions = rng.choice(qN, size = t, replace=False); #Pick t distinct positions for the error vector
    e = GF.Zeros((1,qN)); #start with an all-zero error row vector
    vals = GF.Random(t); #draw t random field elements(could include 0)
    #Ensures no zero symbols
    for j in range(t):
        if vals[j] == 0:
            vals[j] = GF(1);
    e[0, positions] = vals; #Place the nonzero error symbols into the chosen positions
    return y + e; #final ciphertext y = mG + e

#Decryption
def rlce_decrypt(y_row, priv):
    #1 Undo the global scramble and loacal mixing
    #unpack the secret matrices and parameters
    S, Gs, A, P, rs = (priv[x] for x in ["S", "Gs", "A", "P", "rs"]);
    n, k, r, t = (priv[x] for x in ["n", "k", "r", "t"]);
    P_inv = P.T; #For a permuatation matrix, the inverse equals the transpose
    A_inv = _inv_gf(A); #compute A^-1 over the field #compute A^-1 over the field; #compute A^-1 over the field
    y1 = y_row @ P_inv @ A_inv; #undo global column permutation and loack block mixing

    #2 Collapse: take first cordinate from each (r+1)-block to form length-n word
    y_prime = GF.Zeros((1,n)); #prepare a length-n vector to hold the projected word
    B = r + 1; #block width
    #For each block i: take out its (r+1) entries, take the first coordinate to from the length-n word
    for i in range(n):
        block = y1[0, i*B:(i+1)*B];
        y_prime[0, i] = block[0];

    # 3) RS-decode to get mS, then multiply by S^{-1}
    mS = rs.decode(y_prime); #Use the base RS decoder to correct up to t errors and recover the message in scrambled space mS
    #ensure mS is a 1 x k row vector shape
    if mS.ndim == 1:
        mS = mS.reshape(1, -1);
    S_inv = _inv_gf(S); #compute the inverse of the row-scarmbler #compute the inverse of the row-scarmbler
    m = mS @ S_inv; #Unscramble to get the original message

    # 4) CCA-style recheck: verify wt(y - m G_pub) <= t
    diff = y_row - (m @ G_pub); #re-encode m and compare against the received y_row to compute the residual
    #if residual has weight > t, reject
    if hamming_weight(diff) > t:
        raise ValueError("Ciphertext invalid (weight check failed)");
    return m; #Output the recovered message row vector

#DEMO
if __name__ == "__main__":
    m = GF.Random((1,k)); #Sample a random 1×k message over GF
    y = rlce_encrypt(m, G_pub, t); #Encrypt the message with the public key
    m_rec = rlce_decrypt(y, priv); #Decrypt the ciphertext with the private key
    print("Roundtrip OK: - RLCE method code:178", np.array_equal(m, m_rec)); #Verify we got the same message back
    print(f"Params: RS[n={n},k={k},t={t}], r={r}, public length={qN} - RLCE method code:179"); #print the parameters used so we can see the scale
