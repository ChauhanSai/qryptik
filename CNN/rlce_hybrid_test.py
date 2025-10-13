import numpy as np
import tensorflow as tf
import galois
import tensorflow_addons as tfa
from rlce_rs import rlce_rs

# --- Configuration ---
N_RS = 63
K_RS = 51
T_RS = 6
R_RS = 1
NOISE_FACTOR = 0.00 # MUST MATCH THE FACTOR USED DURING TRAINING

def binarize_syndrome(syndrome_gf, syndrome_len_gf, gf_field):
    syndrome_as_ndarray = syndrome_gf.view(np.ndarray)
    syndrome_unpacked = np.unpackbits(syndrome_as_ndarray.view(np.uint8), axis=1, bitorder="little")
    binarized_syndrome = syndrome_unpacked[:, :syndrome_len_gf * gf_field.degree].astype(np.float32)
    return binarized_syndrome

def solve_for_error_values(syndrome_gf, error_locations_indices, H_matrix, gf_field):
    H_errors = H_matrix[:, error_locations_indices]
    try:
        error_values, _, _, _ = np.linalg.lstsq(H_errors, syndrome_gf, rcond=None)
        return error_values
    except np.linalg.LinAlgError:
        return None

if __name__ == '__main__':
    print("--- Loading Pre-Trained Model and Environment ---")
    
    try:
        model = tf.keras.models.load_model('rlce_decoder_best_model.keras')
        encryptor = rlce_rs.load('rlce_key_01.npz') # Load the consistent key
    except Exception as e:
        print(f"ERROR: Could not load files. Details: {e}")
        exit()
        
    GF = encryptor.GF
    G_pub = encryptor.public_key['G']
    
    G_sys = G_pub.row_reduce()
    k_dim = G_sys.shape[0]
    n_dim = G_sys.shape[1]
    P = G_sys[:, k_dim:]
    identity_matrix = GF.Identity(n_dim - k_dim)
    H_pub = np.hstack([-P.T, identity_matrix])
    H_pub = GF(H_pub)

    # --- Sequential Simulation ---
    total_trials = 1000 # Using a smaller number for the slow sequential test
    error_weight_to_test = T_RS + 1
    
    print(f"\n--- Starting Sequential Decoding Test ---")
    print(f"Testing {total_trials} cases with {error_weight_to_test} errors and noise factor {NOISE_FACTOR}...")

    cnn_success_count = 0
    classical_success_count = 0
    
    for i in range(total_trials):
        if (i + 1) % 50 == 0:
            print(f"\rRunning trial {i + 1}/{total_trials}...", end="")
        
        # 1. Generate a single message and error
        original_message = GF.Random(K_RS)
        ciphertext_with_errors, true_error_vector = encryptor.encrypt(
            original_message, encryptor.public_key
        )

        # --- Path 1: CNN-Assisted Decoding ---
        syndrome_gf = ciphertext_with_errors @ H_pub.T
        binarized_syndrome = binarize_syndrome(syndrome_gf.reshape(1, -1), H_pub.shape[0], GF)
        
        # Add the same noise as in training
        noise = tf.random.normal(shape=tf.shape(binarized_syndrome), stddev=NOISE_FACTOR)
        noisy_syndrome = binarized_syndrome + noise
        
        # Predict on the single noisy syndrome
        predicted_probabilities = model.predict(noisy_syndrome, verbose=0)
        predicted_error_locations = (predicted_probabilities[0] > 0.5).astype(int)
        error_indices = np.where(predicted_error_locations == 1)[0]
        
        reconstructed_error_vector = GF.Zeros(n_dim)
        if len(error_indices) > 0:
            error_values = solve_for_error_values(syndrome_gf, error_indices, H_pub, GF)
            if error_values is not None:
                reconstructed_error_vector[error_indices] = error_values
        
        # The true test: compare the predicted error vector to the actual one
        if np.array_equal(reconstructed_error_vector, true_error_vector):
            cnn_success_count += 1

        # --- Path 2: Classical Algebraic Decoding ---
        try:
            classical_decrypted_message = encryptor.decrypt(ciphertext_with_errors)
            if np.array_equal(original_message, classical_decrypted_message):
                classical_success_count += 1
        except galois.errors.GaloisError:
            pass # The decoder failed

    # --- Final Results ---
    print("\n\n--- Test Complete ---")
    cnn_accuracy = (cnn_success_count / total_trials) * 100
    classical_accuracy = (classical_success_count / total_trials) * 100
    
    print(f"Error Weight Tested: {error_weight_to_test}")
    print(f"CNN Path Success Rate:       {cnn_accuracy:.2f}% ({cnn_success_count}/{total_trials})")
    print(f"Classical Path Success Rate: {classical_accuracy:.2f}% ({classical_success_count}/{total_trials})")