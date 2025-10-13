import os
import warnings
# Suppress warnings at the very top, before other imports.
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.framework.indexed_slices')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.saved_model.nested_structure_coder')

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_addons as tfa
# Use the Keras callbacks compatible with your TensorFlow version
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import your custom modules
from rgnn import rgnn
from rlce_rs import rlce_rs

# --- 1. Configuration and Setup ---
RLCE_KEY_NAME = 'keys/rlce_key_01.npz'
rlce_key = rlce_rs.load(RLCE_KEY_NAME)
KEY_ID = rlce_key.KEY_ID
MODEL_ID = 10 # Increment for the new training run

# This should point to the base model you created with the new architecture (e.g., hidden_dim=128)
BASE_MODEL_FILENAME = f'models/untrained/base_rgnn_model_{KEY_ID}.09' 
# The checkpoint will now save the best model directly to this path
TRAINED_MODEL_FILENAME = f'models/trained/rgnn_model_{KEY_ID}.{MODEL_ID:02d}'

# Training Hyperparameters
NUM_ERRORS_TO_TRAIN = [rlce_key.t + 1, rlce_key.t + 2]
BATCH_SIZE = 64
EPOCHS = 100 # Increase epochs significantly to give callbacks time to work
STEPS_PER_EPOCH = 200
VALIDATION_STEPS = 50
LEARNING_RATE = 1e-5 # Start with a higher learning rate; ReduceLROnPlateau will manage it

# --- Extract necessary data BEFORE the pipeline ---
H_matrix_np = np.array(rlce_key.H, dtype=np.float32)
H_indices = np.where(np.array(rlce_key.H, dtype=np.int32) == 1)
ADJACENCY_DATA = {
    'c_to_v_sources': tf.constant(H_indices[0], dtype=tf.int64),
    'c_to_v_targets': tf.constant(H_indices[1], dtype=tf.int64),
    'v_to_c_sources': tf.constant(H_indices[1], dtype=tf.int64),
    'v_to_c_targets': tf.constant(H_indices[0], dtype=tf.int64),
    'num_checks': rlce_key.H.shape[0],
    'num_vars': rlce_key.H.shape[1]
}
NUM_VARS = ADJACENCY_DATA['num_vars']
NUM_CHECKS = ADJACENCY_DATA['num_checks']

# --- 2. Pure NumPy Data Generator ---
def data_generator(h_matrix, num_vars, possible_num_errors):
    while True:
        error_vector = np.zeros(num_vars, dtype=np.float32)
        num_errors = np.random.choice(possible_num_errors)
        error_indices = np.random.choice(num_vars, num_errors, replace=False)
        error_vector[error_indices] = 1.0
        
        syndrome_float = np.dot(h_matrix, error_vector)
        syndrome = (syndrome_float.astype(np.int32)) % 2
        
        yield syndrome, error_vector

# --- 3. Model Loading and Compilation ---
print("Loading base model...")
model = tf.keras.models.load_model(
    BASE_MODEL_FILENAME,
    custom_objects={'rgnn': rgnn}
)
print("Model loaded successfully.")

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.6, gamma=15.0),
    metrics=['binary_accuracy', 'mse']
)
model.summary()

# --- 4. Define Callbacks ---
print("\nSetting up training callbacks...")
checkpoint_callback = ModelCheckpoint(
    filepath=TRAINED_MODEL_FILENAME,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=7, # Stop if val_loss doesn't improve for 7 epochs
    verbose=1,
    restore_best_weights=True
)
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,   # Reduce LR by a factor of 5
    patience=3,   # Reduce if val_loss doesn't improve for 3 epochs
    min_lr=1e-8,
    verbose=1
)

# --- 5. Create the tf.data.Dataset Pipeline ---
print("\nBuilding tf.data pipeline...")
output_signature = (
    tf.TensorSpec(shape=(NUM_CHECKS,), dtype=tf.int32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32)
)

def create_graph_map_fn(syndrome, error_vector):
    graph = rgnn.create_graph_tensor(syndrome, ADJACENCY_DATA)
    return graph, error_vector

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(H_matrix_np, NUM_VARS, NUM_ERRORS_TO_TRAIN),
    output_signature=output_signature
)
validation_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(H_matrix_np, NUM_VARS, NUM_ERRORS_TO_TRAIN),
    output_signature=output_signature
)

train_dataset = (train_dataset
                 .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .map(lambda graph, labels: (graph.merge_batch_to_components(), tf.reshape(labels, [-1])))
                 .prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset
                      .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .map(lambda graph, labels: (graph.merge_batch_to_components(), tf.reshape(labels, [-1])))
                      .prefetch(tf.data.AUTOTUNE))

print("Data pipeline built successfully.")

# --- 6. Model Training ---
print("\nStarting model training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
    callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback] # ✅ Callbacks added
)
print("Training complete.")
# The final model save is now handled by the ModelCheckpoint, which saves the best version.

