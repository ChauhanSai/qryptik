import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from absl import logging

# Import your custom modules
from rgnn import rgnn
from rlce_rs import rlce_rs

# Suppress expected warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.saved_model.nested_structure_coder')

# --- 1. Load Key and Define Graph Structure ---
print("--- Loading RLCE Key and Deriving Graph Structure ---")
RLCE_KEY_NAME = 'keys/rlce_key_01.npz'
rlce_key = rlce_rs.load(RLCE_KEY_NAME)
KEY_ID = rlce_key.KEY_ID
H_matrix = rlce_key.H

# Derive node indices directly from the H matrix
H_dense = np.array(H_matrix, dtype=np.int32)
H_indices = np.where(H_dense == 1)

check_node_indices = H_indices[0]
variable_node_indices = H_indices[1]

NUM_CHECK_NODES = H_matrix.shape[0]
NUM_VAR_NODES = H_matrix.shape[1]

# This dictionary defines the static graph structure for the model
ADJACENCY_DATA = {
    'c_to_v_sources': tf.constant(check_node_indices, dtype=tf.int64),
    'c_to_v_targets': tf.constant(variable_node_indices, dtype=tf.int64),
    'v_to_c_sources': tf.constant(variable_node_indices, dtype=tf.int64),
    'v_to_c_targets': tf.constant(check_node_indices, dtype=tf.int64),
    'num_checks': NUM_CHECK_NODES,
    'num_vars': NUM_VAR_NODES
}
print("Graph structure derived successfully.")

# --- 2. Instantiate and Build the Model ---
print("\n--- Instantiating and Building the RGNN Model ---")
model = rgnn(
    adjacency_lists=ADJACENCY_DATA,
    num_iterations=18
)

# Build the model by creating and calling it with a sample input
print("Building model by calling it with a dummy batched graph...")
dummy_syndrome_1d = tf.zeros(NUM_CHECK_NODES, dtype=tf.int32)
# ✅ Call the static method directly from the class
dummy_graph = rgnn.create_graph_tensor(dummy_syndrome_1d, ADJACENCY_DATA)
batched_dummy_graph = dummy_graph.merge_batch_to_components()
_ = model(batched_dummy_graph) # This call builds the model's weights
print("Model successfully built.")

model.summary()

# --- 3. Compile the Model ---
LEARNING_RATE = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy']
)
print("\nModel compiled successfully.")

# --- 4. Save the Untrained Model ---
MODEL_ID = 9 # Or your desired version number
BASE_MODEL_FILENAME = f'models/untrained/base_rgnn_model_{KEY_ID}.{MODEL_ID:02d}'
print(f"\nSaving untrained model to {BASE_MODEL_FILENAME}...")

# Temporarily change logging verbosity to suppress benign save warnings
original_absl_verbosity = logging.get_verbosity()
logging.set_verbosity(logging.ERROR)
try:
    model.save(BASE_MODEL_FILENAME, save_format='tf')
    print("Untrained model saved successfully.")
finally:
    logging.set_verbosity(original_absl_verbosity) # Restore original setting