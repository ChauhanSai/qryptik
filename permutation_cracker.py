import numpy as np
import galois

class permutation_cracker:
    def crack(public_key: dict) -> galois.FieldArray:
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