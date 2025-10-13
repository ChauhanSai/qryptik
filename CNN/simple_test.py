import numpy as np
import tensorflow as tf
from rlce_rs import rlce_rs


# --- Configuration ---
# These MUST match the parameters used to generate your key and train your model.
N_RS = 63
K_RS = 51
T_RS = 6
R_RS = 1
NOISE_FACTOR = 0.00
ERROR_WEIGHT_TO_TEST = T_RS + 1

# --- Helper function from your training script ---
def binarize_syndrome(syndrome_gf, syndrome_len_gf, gf_field):
    """Converts a Galois Field syndrome vector into its binary representation."""
    syndrome_as_ndarray = syndrome_gf.view(np.ndarray)
    syndrome_unpacked = np.unpackbits(syndrome_as_ndarray.view(np.uint8), axis=1, bitorder="little")
    binarized_syndrome = syndrome_unpacked[:, :syndrome_len_gf * gf_field.degree].astype(np.float32)
    return binarized_syndrome

if __name__ == '__main__':
    print("--- Loading Model and Key ---")
    try:
        model = tf.keras.models.load_model('rlce_decoder_best_model.keras')
        encryptor = rlce_rs.load('rlce_key_01.npz')
    except Exception as e:
        print(f"ERROR: Could not load files. Make sure your model and key files are present. Details: {e}")
        exit()

    # Extract components from the loaded encryptor
    GF = encryptor.GF
    G_pub = encryptor.public_key['G']
    n_dim = G_pub.shape[1]
    
    # Re-derive H_pub from the consistent, loaded public key
    G_sys = G_pub.row_reduce()
    k_dim = G_sys.shape[0]
    P = G_sys[:, k_dim:]
    identity_matrix = GF.Identity(n_dim - k_dim)
    H_pub = np.hstack([-P.T, identity_matrix])
    H_pub = GF(H_pub)

    print(f"\n--- Generating a single test case with {ERROR_WEIGHT_TO_TEST} errors ---")

    # 1. Generate one message, ciphertext, and the true error vector
    original_message = GF.Random(K_RS)
    # MODIFICATION: Explicitly pass the number of errors to generate.
    # This ensures the ground truth has the same number of errors we ask the model to find.
    ciphertext, true_error_vector_gf = encryptor.encrypt(
        original_message, 
        encryptor.public_key
    )
    
    # 2. Create the TRUE binary label vector that the CNN was trained on
    true_error_locations_binary = (true_error_vector_gf != 0).astype(np.float32)

    # 3. Calculate the syndrome and prepare it for the model (with noise)
    syndrome_gf = ciphertext @ H_pub.T
    syndrome_binary = binarize_syndrome(syndrome_gf.reshape(1, -1), H_pub.shape[0], GF)
    noise = tf.random.normal(shape=tf.shape(syndrome_binary), stddev=NOISE_FACTOR)
    syndrome_noisy = syndrome_binary + noise

    # --- THE MOMENT OF TRUTH ---
    print("\n--- Getting CNN Prediction ---")
    predicted_probabilities = model.predict(syndrome_noisy, verbose=0)[0] # Get the single prediction vector

    # --- MODIFIED LOGIC: Use Top-K method instead of a fixed threshold ---
    print(f"--- Identifying Top {ERROR_WEIGHT_TO_TEST} Most Likely Error Locations ---")
    
    # Use argsort to get the indices of the top K probabilities.
    # np.argsort returns indices from smallest to largest, so we take the last K indices.
    top_k_indices = np.argsort(predicted_probabilities)[-ERROR_WEIGHT_TO_TEST:]

    # Create a new binary vector based on the top K indices for comparison
    predicted_locations_binary = np.zeros_like(predicted_probabilities, dtype=np.float32)
    predicted_locations_binary[top_k_indices] = 1.0


    # --- Compare the binary outputs ---
    print("\n--- Comparing True vs. Predicted Error Locations ---")
    true_indices = np.where(true_error_locations_binary == 1)[0]
    pred_indices = np.where(predicted_locations_binary == 1)[0] # This now uses the top-k result

    # Sort for consistent display
    true_indices.sort()
    pred_indices.sort()

    print(f"True error locations ({len(true_indices)}):      {true_indices}")
    print(f"Predicted error locations ({len(pred_indices)}): {pred_indices}")
    
    if np.array_equal(true_indices, pred_indices):
        print("\n✅ SUCCESS: The CNN predicted the error locations perfectly!")
    else:
        print("\n❌ FAILURE: The CNN did not predict the error locations correctly.")
        
    # --- Deeper Analysis ---
    print("\n--- Diagnostic: Probabilities at True Error Locations ---")
    probs_at_true_locations = predicted_probabilities[true_indices]
    print(f"The model assigned these probabilities to the CORRECT error locations:\n{probs_at_true_locations}")

    if len(pred_indices) > 0:
        print("\n--- Diagnostic: Probabilities at Incorrectly Predicted Locations ---")
        # Find indices that are in pred but not in true
        false_positive_indices = np.setdiff1d(pred_indices, true_indices)
        if len(false_positive_indices) > 0:
            probs_at_fp_locations = predicted_probabilities[false_positive_indices]
            print(f"The model assigned these probabilities to INCORRECT locations:\n{probs_at_fp_locations}")
        else:
            print("No false positives were predicted.")

