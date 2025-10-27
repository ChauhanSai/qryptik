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
TRAINED_MODEL_FILENAME = f'models/trained/EMERGENCY_rgnn_3.05_best'
ERRORS_TO_FIND = 1
ERRORS_TO_INJECT = 2


TEST_BATCH_SIZE = 1000

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

    print("Model loaded successfully.")

    success, fail, total = 0, 0, 0
    avg_prob_at_error_total, avg_prob_at_non_error_total = 0.0, 0.0
    avg_standard_deviation_total = 0.0

    for i in range(TEST_BATCH_SIZE):
        # generate test case
        original_message = rlce_key.GF.Random(rlce_key.k)
        ciphertext, true_error_vector_binary = rlce_key.encrypt(
            original_message,
            rlce_key.public_key,
            num_errors=ERRORS_TO_INJECT
        )
        syndrome_gf = ciphertext @ rlce_key.H.T
        syndrome_binary = syndrome_gf.view(np.ndarray).astype(np.int32)
        
        graph_input = rgnn.create_graph_tensor(syndrome_binary, ADJACENCY_DATA)
        batched_graph_input = graph_input.merge_batch_to_components()

        # Get the RGNN Prediction
        predicted_logits = model(batched_graph_input, training=False).numpy()
        predicted_probabilities = tf.sigmoid(predicted_logits).numpy()

        top_k_indices = np.argsort(predicted_probabilities)[-ERRORS_TO_FIND:]
        predicted_locations_binary = np.zeros_like(predicted_probabilities, dtype=np.float32)
        predicted_locations_binary[top_k_indices] = 1.0

        true_indices = np.where(true_error_vector_binary == 1)[0]
        pred_indices = np.where(predicted_locations_binary == 1)[0]

        if np.all(np.isin(pred_indices, true_indices)):
            success += 1
            print(f"[Trial #{i+1}] ✅ SUCCESS: Predicted Indicies:", pred_indices, " True Indicies:", true_indices)
        else:
            print(f"[Trial #{i+1}] ❌ FAILURE: Predicted Indicies:", pred_indices, " True Indicies:", true_indices)
            fail += 1
        total += 1

        probs_at_true_errors = predicted_probabilities[true_error_vector_binary == 1]
        probs_at_non_errors = predicted_probabilities[true_error_vector_binary == 0]
        
        # Calculate statistics
        avg_prob_at_error = np.mean(probs_at_true_errors)
        avg_prob_at_non_error = np.mean(probs_at_non_errors)
        std_dev_all_probs = np.std(predicted_probabilities)
        max_prob = np.max(predicted_probabilities)
        max_prob_index = np.argmax(predicted_probabilities)
        print(f"\tAvg. at errors: {avg_prob_at_error:.4f}, Avg. at non-errors: {avg_prob_at_non_error:.4f}, standard deviation: {std_dev_all_probs:.4f}")
        print(f"\tHighest prob: {max_prob:.4f} at index {max_prob_index}\n")

        if i > 1:
            avg_prob_at_error_total = (avg_prob_at_error_total*(i-1) + avg_prob_at_error)/i
            avg_prob_at_non_error_total = (avg_prob_at_non_error_total*(i-1) + avg_prob_at_non_error)/i
            avg_standard_deviation_total = (avg_standard_deviation_total*(i-1) + std_dev_all_probs)/i
        else:
            avg_prob_at_error_total = avg_prob_at_error
            avg_prob_at_non_error_total = avg_prob_at_non_error
            avg_standard_deviation_total = std_dev_all_probs


    print(f"--- Testing Complete: {success} Successes, {fail} Failures, out of {total} Trials ---")
    success_rate = (success / total) * 100 if total > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Average Probability at True Error Locations: {avg_prob_at_error_total:.4f}")
    print(f"Average Probability at Non-Error Locations: {avg_prob_at_non_error_total:.4f}")
    print(f"Average Standard Deviation of Probabilities: {avg_standard_deviation_total:.4f}")  
    