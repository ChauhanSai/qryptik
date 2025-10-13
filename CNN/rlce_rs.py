import numpy as np
import galois
import pickle

class rlce_rs:
    def __init__(self, n, k, t, r, generate_new=True):
        self.n = n
        self.k = k
        self.t = t
        self.r = r
        
        if generate_new:
            self.GF = galois.GF(2**self.t)
            self.rs_code = self._generate_rs_code()
            self.Gs = self.rs_code.G
            self.G1 = self._generate_G1()
            self.A = self._generate_A()
            self.S = self._generate_random_non_singular_matrix(k)
            self.P = self.GF(self._generate_random_permutation_matrix(n*(r+1)))
            self.public_key = {"G":(self.S @ self.G1 @ self.A @ self.P), "n":n, "r":r, "k":k,"t":t, "GF":self.GF}
    
    def _generate_rs_code(self) -> galois.ReedSolomon:
        rs_code = galois.ReedSolomon(n=self.n, k=self.k, field=self.GF)
        return rs_code
    
    def _generate_G1(self) -> galois.FieldArray:
        columns_to_stack = []
        
        for i in range(self.n):
            columns_to_stack.append(self.Gs[:, [i]])
            
            for _ in range(self.r):
                # Convert the random column to a GF array
                column = self.GF.Random((self.k, 1))
                columns_to_stack.append(column)
                
        G1 = np.hstack(columns_to_stack)
        return G1
    
    def _generate_A(self) -> galois.FieldArray:
        n = self.n
        r = self.r
        GF = self.GF
        blocks = [self._generate_random_non_singular_matrix(r + 1) for _ in range(n)]
        A = np.block([
            [blocks[i] if i == j else GF.Zeros((r+1, r+1)) for j in range(n)]
            for i in range(n)
        ])

        return GF(A)

    def _generate_random_non_singular_matrix(self,n) -> galois.FieldArray:
        while True:
            # Generate matrix directly in the specified Galois Field
            matrix = self.GF.Random((n, n))
            # Use np.linalg.det, which is overloaded by the galois library
            if np.linalg.det(matrix) != 0:
                return matrix

    def _generate_random_permutation_matrix(self,n) -> np.ndarray:
        permutation_indices = np.random.permutation(n)
        identity_matrix = np.eye(n, dtype=int)
        permutation_matrix = identity_matrix[permutation_indices, :]
        return permutation_matrix
    
    @staticmethod
    def encrypt(message: np.ndarray, public_key) -> galois.FieldArray:
        G=public_key["G"]
        GF=public_key["GF"]
        t=public_key["t"]
        
        e = GF.Zeros(G.shape[1])
        # Generate an error vector with a specified weight
        one_indices = np.random.choice(G.shape[1], size=t+1, replace=False)
        # The error values can be any non-zero element from the field
        error_values = GF.Random(t+1, low=1)
        e[one_indices] = error_values
        
        ciphertext = (GF(message) @ G + e)
        
        return ciphertext, e
    
    def decrypt(self, ciphertext: galois.FieldArray) -> galois.FieldArray:

        # Get the inverse matrices
        S_inv = np.linalg.inv(self.S)
        P_inv = np.linalg.inv(self.P)
        A_inv = np.linalg.inv(self.A)

        intermediate_vector = ciphertext @ P_inv @ A_inv

        # Extract the parts of the vector corresponding to the original RS code
        rs_codeword_indices = [i * (self.r + 1) for i in range(self.n)]
        received_codeword = intermediate_vector[rs_codeword_indices]

        # Decode the extracted codeword to get the message (scrambled by S)
        decoded_message_scrambled = self.rs_code.decode(received_codeword, output="message")

        # Unscramble with the inverse of S to recover the original message
        decrypted_message = decoded_message_scrambled @ S_inv

        return decrypted_message
    
    def save(self, filepath):
        """Saves the key components to a compressed NumPy file."""
        # Convert galois arrays to basic NumPy arrays for saving
        np.savez_compressed(
            filepath,
            n=self.n, k=self.k, t=self.t, r=self.r,
            Gs=self.Gs.view(np.ndarray),
            G1=self.G1.view(np.ndarray),
            A=self.A.view(np.ndarray),
            S=self.S.view(np.ndarray),
            P=self.P.view(np.ndarray)
        )
        print(f"Encryptor state saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads an encryptor state from a NumPy file."""
        with np.load(filepath) as data:
            # Create a new, empty instance
            encryptor = cls(
                n=int(data['n']), k=int(data['k']), t=int(data['t']), r=int(data['r']),
                generate_new=False # Important: prevent re-generation
            )
            
            # Re-create the GF and rs_code from saved parameters
            encryptor.GF = galois.GF(2**encryptor.t)
            encryptor.rs_code = galois.ReedSolomon(encryptor.n, encryptor.k, field=encryptor.GF)
            
            # Load the matrices and cast them back into the GF
            encryptor.Gs = encryptor.GF(data['Gs'])
            encryptor.G1 = encryptor.GF(data['G1'])
            encryptor.A = encryptor.GF(data['A'])
            encryptor.S = encryptor.GF(data['S'])
            encryptor.P = encryptor.GF(data['P'])
            
            # Re-create the public key dictionary
            encryptor.public_key = {
                "G":(encryptor.S @ encryptor.G1 @ encryptor.A @ encryptor.P),
                "n":encryptor.n, "r":encryptor.r, "k":encryptor.k, "t":encryptor.t, "GF":encryptor.GF
            }
        print(f"Encryptor state loaded from {filepath}")
        return encryptor