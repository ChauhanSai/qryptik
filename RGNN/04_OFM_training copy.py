import os
import warnings
from absl import logging
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from rgnn import rgnn
from rlce_rs import rlce_rs

# ============================================================================
# EMERGENCY SIMPLE CONFIG - GUARANTEED TO WORK
# ============================================================================
RLCE_KEY_NAME = 'keys/rlce_key_04.npz'
KEY_ID = 4
MODEL_ID = 1

# SIMPLE architecture for 1-from-1
NUM_ITERATIONS = 12  # Increase from 8
HIDDEN_DIM = 96      # Increase from 64
DROPOUT_RATE = 0.15  # Reduce dropout

# Training: 1 error from 1 error (simplest case)
NUM_ERRORS = 1
BATCH_SIZE = 32
EPOCHS = 200         # More epochs
STEPS_PER_EPOCH = 500  # More steps
VALIDATION_STEPS = 100  # More validation
LEARNING_RATE = 1e-3  # Lower starting LR

MODEL_SAVE_PATH = f'models/trained/EMERGENCY_rgnn_{KEY_ID}.{MODEL_ID:02d}'

# ============================================================================
# SETUP
# ============================================================================
print("="*60)
print("EMERGENCY SIMPLE TRAINING")
print("="*60)

rlce_key = rlce_rs.load(RLCE_KEY_NAME)
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

print(f"Graph: {NUM_CHECKS} checks, {NUM_VARS} vars")
print(f"Task: Find {NUM_ERRORS} error from {NUM_ERRORS} total")

# ============================================================================
# SIMPLE DATA GENERATOR - NO FANCY WEIGHTS
# ============================================================================
def simple_generator(h_matrix, num_vars, num_errors):
    """Dead simple generator for 1-from-1"""
    while True:
        # Create 1 error
        error_idx = np.random.randint(0, num_vars)
        error_vector = np.zeros(num_vars, dtype=np.float32)
        error_vector[error_idx] = 1.0
        
        # Calculate syndrome
        syndrome_float = np.dot(h_matrix, error_vector)
        syndrome = (syndrome_float.astype(np.int32)) % 2
        
        yield syndrome, error_vector

# ============================================================================
# BUILD MODEL FROM SCRATCH
# ============================================================================
print("\n--- Creating Model ---")
model = rgnn(
    adjacency_lists=ADJACENCY_DATA,
    num_iterations=NUM_ITERATIONS,
    hidden_dim=HIDDEN_DIM,
    dropout_rate=DROPOUT_RATE
)

# Build
dummy_syndrome = tf.zeros(NUM_CHECKS, dtype=tf.int32)
dummy_graph = rgnn.create_graph_tensor(dummy_syndrome, ADJACENCY_DATA)
_ = model(dummy_graph.merge_batch_to_components())

print(f"Model: {NUM_ITERATIONS} iters, {HIDDEN_DIM} dim")
model.summary()

# Compile with focal loss (handles imbalance automatically)
import tensorflow_addons as tfa

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# ============================================================================
# SIMPLE DATA PIPELINE - NO SAMPLE WEIGHTS
# ============================================================================
print("\n--- Building Pipeline ---")

output_signature = (
    tf.TensorSpec(shape=(NUM_CHECKS,), dtype=tf.int32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32)
)

def create_graph(syndrome, error_vector):
    graph = rgnn.create_graph_tensor(syndrome, ADJACENCY_DATA)
    return graph, error_vector

train_dataset = tf.data.Dataset.from_generator(
    lambda: simple_generator(H_matrix_np, NUM_VARS, NUM_ERRORS),
    output_signature=output_signature
)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: simple_generator(H_matrix_np, NUM_VARS, NUM_ERRORS),
    output_signature=output_signature
)

train_dataset = (train_dataset
                 .map(create_graph, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .map(lambda g, l: (g.merge_batch_to_components(), tf.reshape(l, [-1])))
                 .prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset
                      .map(create_graph, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .map(lambda g, l: (g.merge_batch_to_components(), tf.reshape(l, [-1])))
                      .prefetch(tf.data.AUTOTUNE))

# Validate
for batch in train_dataset.take(1):
    _, labels = batch
    print(f"Labels sum: {tf.reduce_sum(labels).numpy()} (expect ~{NUM_ERRORS * BATCH_SIZE})")

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH + '_best',
        monitor='val_recall',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_recall',
        patience=50,  # Much more patience!
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # More patience here too
        min_lr=1e-6,
        verbose=1
    )
]

# ============================================================================
# TRAIN
# ============================================================================
print("\n" + "="*60)
print("TRAINING")
print("="*60 + "\n")

logging.set_verbosity(logging.ERROR)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
    callbacks=callbacks
)

# ============================================================================
# TEST
# ============================================================================
print("\n--- Quick Test ---")
successes = 0
for i in range(100):
    msg = rlce_key.GF.Random(rlce_key.k)
    ct, true_errors = rlce_key.encrypt(msg, rlce_key.public_key, num_errors=1)
    true_idx = np.where(true_errors == 1)[0][0]
    
    syndrome = (ct @ rlce_key.H.T).view(np.ndarray).astype(np.int32)
    graph = rgnn.create_graph_tensor(syndrome, ADJACENCY_DATA)
    pred = model(graph.merge_batch_to_components(), training=False).numpy()
    
    pred_idx = np.argmax(pred)
    if pred_idx == true_idx:
        successes += 1

print(f"\nTest Success Rate: {successes}% (should be 85%+)")

# ============================================================================
# SAVE
# ============================================================================
model.save(MODEL_SAVE_PATH, save_format='tf')
print(f"\nModel saved: {MODEL_SAVE_PATH}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Final recall: {history.history['recall'][-1]:.4f}")
print(f"Best val_recall: {max(history.history['val_recall']):.4f}")
print(f"Test accuracy: {successes}%")
print("="*60)