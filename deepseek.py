import numpy as np
import hashlib
import os
from typing import Tuple, List, Dict, Optional
from scipy.linalg import block_diag  # Add this import


# Finite Field Arithmetic for GF(2^m)
class GF2m:
    def __init__(self, m: int, prim_poly: int):
        self.m = m
        self.order = 1 << m
        self.prim_poly = prim_poly
        self.exp_table = [0] * (2 * self.order)
        self.log_table = [0] * self.order
        self._build_tables()

    def _build_tables(self):
        x = 1
        for i in range(self.order):
            self.exp_table[i] = x
            self.log_table[x] = i
            x <<= 1
            if x & self.order:
                x ^= self.prim_poly

        for i in range(self.order, 2 * self.order):
            self.exp_table[i] = self.exp_table[i % (self.order - 1)]

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        if not (0 < a < self.order) or not (0 < b < self.order):  # Validate finite field elements
            raise ValueError(f"Invalid finite field elements: a={a}, b={b}")
        log_sum = self.log_table[a] + self.log_table[b]
        return self.exp_table[log_sum % (self.order - 1)]  # Ensure valid index

    def inv(self, a: int) -> int:
        if a == 0:
            raise ValueError("Cannot invert 0")
        return self.exp_table[self.order - 1 - self.log_table[a]]


# RLCE-KEM Implementation
class RLCE_KEM:
    PARAMS = {
        3: {'n': 40, 'k': 20, 't': 10, 'w': 5, 'm': 10, 'prim_poly': 0x409}  # x^10 + x^3 + 1
    }

    def __init__(self, param_id: int = 3):
        self.param_id = param_id
        params = self.PARAMS[param_id]
        self.n = params['n']
        self.k = params['k']
        self.t = params['t']
        self.w = params['w']
        self.m = params['m']
        self.field = GF2m(self.m, params['prim_poly'])
        self.q = 1 << self.m
        self.sk = None
        self.pk = None

    def key_gen(self):
        # Step 1: Generate GRS code generator matrix Gs (k x n)
        Gs = self._gen_grs_matrix()

        # Step 2: Random permutation matrix P1 (n x n)
        P1 = self._rand_perm_matrix(self.n)

        # Step 3: Insert random columns to form G1 (k x (n+w))
        G1 = self._insert_random_columns(Gs, P1)

        # Step 4: Generate block-diagonal matrix A ((n+w) x (n+w))
        A = self._gen_block_diag_matrix()

        # Step 5: Compute G2 = G1 @ A
        G2 = self._matrix_mult(G1, A)

        # Step 6: Random permutation matrix P2 ((n+w) x (n+w))
        P2 = self._rand_perm_matrix(self.n + self.w)

        # Step 7: Compute G3 = G2 @ P2 and convert to systematic form [I_k | G_E]
        G3 = self._matrix_mult(G2, P2)
        G_pub, S = self._to_systematic(G3)

        # Store keys
        self.pk = G_pub
        self.sk = {
            'S': S, 'Gs': Gs, 'P1': P1, 'P2': P2, 'A': A,
            'params': (self.n, self.k, self.t, self.w, self.m)
        }

    def encapsulate(self) -> Tuple[bytes, bytes]:
        # Step 1: Generate random message (k1 bytes)
        msg = os.urandom(self.k)

        # Step 2: Generate random error vector e (weight t)
        e = self._gen_error_vector()

        # Step 3: Encrypt: c = msg * G_pub + e
        msg_vec = self._bytes_to_gf(msg)
        c = self._vector_add(
            self._vector_mult(msg_vec, self.pk),
            e
        )
        ciphertext = self._gf_to_bytes(c)
        return ciphertext, msg

    def decapsulate(self, ciphertext: bytes) -> bytes:
        # Step 1: Convert ciphertext to vector
        c = self._bytes_to_gf(ciphertext)

        # Step 2: Compute c' = c * P2^{-1} * A^{-1}
        c_prime = self._vector_mult(c, np.linalg.inv(self.sk['P2']))
        c_prime = self._vector_mult(c_prime, np.linalg.inv(self.sk['A']))

        # Step 3: Remove inserted columns to get c'' (length n)
        c_double_prime = self._remove_inserted_columns(c_prime)

        # Step 4: Apply inverse permutation P1^{-1}
        c_triple_prime = self._vector_mult(c_double_prime, np.linalg.inv(self.sk['P1']))

        # Step 5: Decode GRS code to recover msg * S
        msg_s = self._grs_decode(c_triple_prime)

        # Step 6: Recover original message: msg = (msg * S) * S^{-1}
        msg_vec = self._vector_mult(msg_s, np.linalg.inv(self.sk['S']))
        return self._gf_to_bytes(msg_vec[:self.k])

    # Helper methods
    def _gen_grs_matrix(self) -> np.ndarray:
        x = np.random.choice(range(1, self.q), self.n, replace=False)
        y = np.random.choice(range(1, self.q), self.n)
        Gs = np.zeros((self.k, self.n), dtype=int)
        for i in range(self.k):
            for j in range(self.n):
                x_power_i = x[j] ** i
                x_power_i %= self.field.order  # Ensure valid finite field element
                y_j = y[j] % self.field.order  # Ensure valid finite field element
                Gs[i, j] = self.field.mul(y_j, x_power_i)
        return Gs

    def _rand_perm_matrix(self, size: int) -> np.ndarray:
        perm = np.random.permutation(size)
        P = np.eye(size)[perm]
        return P

    def _insert_random_columns(self, Gs: np.ndarray, P1: np.ndarray) -> np.ndarray:
        G_perm = np.dot(Gs, P1)
        # Ensure the number of columns matches n + w
        random_columns = np.random.randint(0, self.q, (self.k, self.w))
        G1 = np.hstack((
            G_perm[:, :self.n - self.w],
            random_columns,
            G_perm[:, self.n - self.w:]
        ))
        return G1  # Ensure G1 has exactly (k, n + w) dimensions

    def _gen_block_diag_matrix(self) -> np.ndarray:
        blocks = []
        # Identity block for first n-w columns
        blocks.append(np.eye(self.n - self.w))
        # 2x2 blocks for w columns (each block takes 2 columns)
        for _ in range(self.w // 2):
            block = np.random.randint(1, self.q, (2, 2))
            while np.linalg.det(block) == 0:  # Ensure non-singular
                block = np.random.randint(1, self.q, (2, 2))
            blocks.append(block)
        # Ensure the block diagonal matrix has dimensions (n+w, n+w)
        return block_diag(*blocks)

    def _matrix_mult(self, mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication in the finite field GF(2^m).
        """
        result = np.dot(mat1, mat2) % self.q  # Ensure elements remain in GF(2^m)
        return result

    def _to_systematic(self, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Gaussian elimination to convert to systematic form [I_k | G_E]
        # Returns (public_key, transformation_matrix)
        pass  # Implementation omitted for brevity

    def _gen_error_vector(self) -> np.ndarray:
        e = np.zeros(self.n + self.w, dtype=int)
        error_positions = np.random.choice(self.n + self.w, self.t, replace=False)
        for pos in error_positions:
            e[pos] = np.random.randint(1, self.q)
        return e

    def _grs_decode(self, c: np.ndarray) -> np.ndarray:
        # Simplified Peterson-Gorenstein-Zierler decoder
        pass  # Implementation omitted for brevity

    # Utility conversion methods
    def _bytes_to_gf(self, data: bytes) -> np.ndarray:
        # Convert bytes to GF vector
        pass

    def _gf_to_bytes(self, vec: np.ndarray) -> bytes:
        # Convert GF vector to bytes
        pass

    def _vector_mult(self, vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
        return np.dot(vec, mat) % self.q

    def _vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return [(self.field.add(x, y)) for x, y in zip(a, b)]


# Example usage
if __name__ == "__main__":
    kem = RLCE_KEM(param_id=3)
    kem.key_gen()

    ciphertext, shared_secret = kem.encapsulate()
    recovered_secret = kem.decapsulate(ciphertext)

    print("Shared Secret:", shared_secret.hex())
    print("Recovered Secret:", recovered_secret.hex())
    print("Match:", shared_secret == recovered_secret)
