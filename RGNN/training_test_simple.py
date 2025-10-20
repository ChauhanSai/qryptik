import os
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_addons as tfa

# Import your custom modules
from rgnn import rgnn
from rlce_rs import rlce_rs

# --- 1. Configuration ---
RLCE_KEY_NAME = 'keys/rlce_key_03.npz'
rlce_key = rlce_rs.load(RLCE_KEY_NAME)
KEY_ID = rlce_key.KEY_ID
MODEL_ID = 2 # Match the model version you want to test

# This should point to the model saved by your training script's ModelCheckpoint
TRAINED_MODEL_FILENAME = f'models/trained/rgnn_base_model_3.03'
ERROR_WEIGHT_TO_TEST = 1

# --- Extract graph structure once ---
H_indices = np.where(np.array(rlce_key.H, dtype=np.int32) == 1)
ADJACENCY_DATA = {
    'c_to_v_sources': tf.constant(H_indices[0], dtype=tf.int64),
    'c_to_v_targets': tf.constant(H_indices[1], dtype=tf.int64),
    'v_to_c_sources': tf.constant(H_indices[1], dtype=tf.int64),
    'v_to_c_targets': tf.constant(H_indices[0], dtype=tf.int64),
    'num_checks': rlce_key.H.shape[0],
    'num_vars': rlce_key.H.shape[1]
}

if __name__ == '__main__':
    print("--- Loading Trained Model ---")
    try:
        model = tf.keras.models.load_model(
            TRAINED_MODEL_FILENAME,
            custom_objects={'rgnn': rgnn}
        )
    except Exception as e:
        print(f"ERROR: Could not load the model file. Details: {e}")
        exit()

    # --- 2. Generate a Test Case ---
    print(f"\n--- Generating a test case with {ERROR_WEIGHT_TO_TEST} errors ---")
    original_message = rlce_key.GF.Random(rlce_key.k)
    ciphertext, true_error_vector_binary = rlce_key.encrypt(
        original_message,
        rlce_key.public_key,
        num_errors=ERROR_WEIGHT_TO_TEST
    )
    syndrome_gf = ciphertext @ rlce_key.H.T
    syndrome_binary = syndrome_gf.view(np.ndarray).astype(np.int32)

    # --- 3. Prepare Input for the RGNN Model ---
    graph_input = rgnn.create_graph_tensor(syndrome_binary, ADJACENCY_DATA)
    batched_graph_input = graph_input.merge_batch_to_components()

    # --- 4. Get the RGNN Prediction ---
    print("\n--- Getting RGNN Prediction ---")
    # Call the model directly as a function to avoid `predict` issues
    predicted_probabilities = model(batched_graph_input, training=False).numpy()

    # --- 5. Analyze the Prediction (Top-K method) ---
    print(f"--- Identifying Top {ERROR_WEIGHT_TO_TEST} Most Likely Error Locations ---")
    top_k_indices = np.argsort(predicted_probabilities)[-ERROR_WEIGHT_TO_TEST:]
    predicted_locations_binary = np.zeros_like(predicted_probabilities, dtype=np.float32)
    predicted_locations_binary[top_k_indices] = 1.0

    # --- 6. Compare True vs. Predicted ---
    print("\n--- Comparison ---")
    true_indices = np.where(true_error_vector_binary == 1)[0]
    pred_indices = np.where(predicted_locations_binary == 1)[0]
    true_indices.sort()
    pred_indices.sort()

    print(f"True error locations ({len(true_indices)}):      {true_indices}")
    print(f"Predicted error locations ({len(pred_indices)}):  {pred_indices}")
    
    if np.array_equal(true_indices, pred_indices):
        print("\n✅ SUCCESS: The RGNN predicted the error locations perfectly!")
    else:
        print("\n❌ FAILURE: The RGNN did not predict the error locations correctly.")
        
    # --- 7. NEW: Detailed Diagnostics ---
    print("\n--- Detailed Diagnostics ---")
    
    # Separate the probabilities for correct '1's and correct '0's
    probs_at_true_errors = predicted_probabilities[true_error_vector_binary == 1]
    probs_at_non_errors = predicted_probabilities[true_error_vector_binary == 0]
    
    # Calculate statistics
    avg_prob_at_error = np.mean(probs_at_true_errors)
    avg_prob_at_non_error = np.mean(probs_at_non_errors)
    std_dev_all_probs = np.std(predicted_probabilities)
    max_prob = np.max(predicted_probabilities)
    max_prob_index = np.argmax(predicted_probabilities)
    
    print(f"Avg. probability assigned to TRUE error locations:   {avg_prob_at_error:.4f}")
    print(f"Avg. probability assigned to NON-error locations:  {avg_prob_at_non_error:.4f}")
    print(f"Standard deviation of all probabilities:         {std_dev_all_probs:.4f}")
    print("-" * 20)
    print(f"Highest predicted probability: {max_prob:.4f} at index {max_prob_index}")
    if true_error_vector_binary[max_prob_index] == 1:
        print("  -> This was a CORRECT prediction.")
    else:
        print("  -> This was an INCORRECT prediction.")

