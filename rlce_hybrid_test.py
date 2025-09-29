import numpy as np
import tensorflow as tf
import galois

# --- Assumed local imports ---
# Ensure you have these files in the same directory:
# 1. rlce_rs.py (the new system you provided)
from rlce_rs import rlce_rs

# --- Configuration for RLCE-RS System ---
# These parameters MUST match the ones used during training.
N_RS = 63  # RS code length
K_RS = 51  # RS message dimension
T_RS = 6   # Error correction capability ('t')
R_RS = 1   # Dimension of random matrices

def binarize_syndrome(syndrome_gf, syndrome_len_gf, gf_field):
    """
    Helper function to convert a Galois Field syndrome vector into its
    binary representation, matching the format used for training.
    """
    syndrome_as_ndarray = syndrome_gf.view(np.ndarray)
    syndrome_unpacked = np.unpackbits(syndrome_as_ndarray.view(np.uint8), axis=1, bitorder="little")
    binarized_syndrome = syndrome_unpacked[:, :syndrome_len_gf * gf_field.degree].astype(np.float32)
    return binarized_syndrome

def solve_for_error_values(syndrome_gf, error_locations_indices, H_matrix, gf_field):
    """
    Solves a system of linear equations to find the actual error values
    in the Galois Field, given their locations.
    
    The equation is s = H_errors @ e_values.T
    
    Args:
        syndrome_gf (galois.FieldArray): The syndrome vector.
        error_locations_indices (np.ndarray): Indices where errors were detected.
        H_matrix (galois.FieldArray): The public parity-check matrix.
        
    Returns:
        galois.FieldArray: The calculated error values for the specified locations.
    """
    # Select the columns of H corresponding to the error locations
    H_errors = H_matrix[:, error_locations_indices]
    
    try:
        # Use Least Squares (lstsq) for non-square systems. This is more robust
        # when the number of predicted errors doesn't match the syndrome dimension.
        # It finds the best-fit solution to s = H_errors * x.
        # np.linalg.lstsq is overloaded by the galois library for GF operations.
        error_values, _, _, _ = np.linalg.lstsq(H_errors, syndrome_gf)
        return error_values
    except np.linalg.LinAlgError:
        # This can happen if the columns of H for the predicted errors are not
        # linearly independent, meaning the model's prediction was likely incorrect.
        print("Could not solve for error values. The predicted error locations may be incorrect.")
        return None

if __name__ == '__main__':
    print("--- Loading Pre-Trained Model and Setting up Environment ---")
    
    # 1. Load the trained Keras model
    try:
        model = tf.keras.models.load_model('rlce_decoder_best_model.keras')
    except Exception as e:
        print(f"ERROR: Could not load 'rlce_decoder_best_model.keras'. Make sure you have run the training script first.")
        print(f"Details: {e}")
        exit()
        
    # 2. Re-create the RLCE-RS instance to get the public matrices
    encryptor = rlce_rs(n=N_RS, k=K_RS, t=T_RS, r=R_RS)
    G_pub = encryptor.public_key['G']
    GF = encryptor.GF
    
    # 3. Re-derive the public Parity Check Matrix (H)
    G_sys = G_pub.row_reduce()
    k_dim = G_sys.shape[0]
    P = G_sys[:, k_dim:]
    n_dim = G_sys.shape[1]
    identity_matrix = GF.Identity(n_dim - k_dim)
    H_pub = np.hstack([-P.T, identity_matrix])
    H_pub = GF(H_pub)

    # --- Simulation: Run multiple trials to get statistics ---
    print("\n--- Starting Bulk Decoding Test ---")
    
    total_trials = 1000
    cnn_success_count = 0
    classical_success_count = 0
    
    for i in range(total_trials):
        print(f"\rRunning trial {i + 1}/{total_trials}...", end="")
        
        # Create a single random message and ciphertext for this trial
        original_message = GF.Random(K_RS)
        ciphertext_with_errors = encryptor.encrypt(original_message, encryptor.public_key)

        # --- Path 1: CNN-Assisted Decoding ---
        syndrome_gf = ciphertext_with_errors @ H_pub.T
        syndrome_len_gf = H_pub.shape[0]
        binarized_syndrome = binarize_syndrome(syndrome_gf.reshape(1, -1), syndrome_len_gf, GF)
        
        predicted_probabilities = model.predict(binarized_syndrome, verbose=0) # verbose=0 hides per-prediction logs
        predicted_error_locations = (predicted_probabilities[0] > 0.5).astype(int)
        error_indices = np.where(predicted_error_locations == 1)[0]
        
        reconstructed_error_vector = None
        if len(error_indices) == 0:
            reconstructed_error_vector = GF.Zeros(n_dim)
        else:
            error_values = solve_for_error_values(syndrome_gf, error_indices, H_pub, GF)
            if error_values is not None:
                reconstructed_error_vector = GF.Zeros(n_dim)
                reconstructed_error_vector[error_indices] = error_values

        if reconstructed_error_vector is not None:
            corrected_codeword = ciphertext_with_errors - reconstructed_error_vector
            cnn_decrypted_message = encryptor.decrypt(corrected_codeword)
            if np.array_equal(original_message, cnn_decrypted_message):
                cnn_success_count += 1

        # --- Path 2: Classical Algebraic Decoding ---
        try:
            classical_decrypted_message = encryptor.decrypt(ciphertext_with_errors)
            if np.array_equal(original_message, classical_decrypted_message):
                classical_success_count += 1
        except galois.errors.GaloisError:
            pass # The decoder failed, so we don't increment the counter

    # --- Final Results ---
    print("\n\n--- Test Complete ---")
    cnn_accuracy = (cnn_success_count / total_trials) * 100
    classical_accuracy = (classical_success_count / total_trials) * 100
    
    print(f"CNN Path Success Rate:       {cnn_accuracy:.2f}% ({cnn_success_count}/{total_trials})")
    print(f"Classical Path Success Rate: {classical_accuracy:.2f}% ({classical_success_count}/{total_trials})")

