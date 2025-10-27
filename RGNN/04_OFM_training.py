import os
import warnings
from absl import logging
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.framework.indexed_slices')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.saved_model.nested_structure_coder')

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from rgnn import rgnn
from rlce_rs import rlce_rs

# ============================================================================
# CONFIGURATION
# ============================================================================
RLCE_KEY_NAME = 'keys/rlce_key_03.npz'
KEY_ID = 4
MODEL_ID = 1

# Model architecture
NUM_ITERATIONS = 24
HIDDEN_DIM = 256
DROPOUT_RATE = 0.4

# Training configuration
NUM_ERRORS_TOTAL = [1]  # List of error counts (e.g., [1] for 1-from-1, [3,4] for mixed)
NUM_ERRORS_TO_LABEL = 'all'  # How many to label (use 'all' to label all errors)
BATCH_SIZE = 32
EPOCHS = 500
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 100
LEARNING_RATE = 1e-3
SAMPLE_WEIGHT_ON_ERRORS = 15.0

# Optional: Load pretrained model instead of creating new
LOAD_PRETRAINED = False
PRETRAINED_MODEL_PATH = 'models/trained/rgnn_model_3.3.01'

# Save paths
MODEL_SAVE_PATH = f'models/trained/1F1_rgnn_model_{KEY_ID}.{MODEL_ID:02d}'

# ============================================================================
# SETUP
# ============================================================================
print("="*60)
print("RGNN TRAINING SCRIPT")
print("="*60)

# Load key and create graph structure
print("\n--- Loading RLCE Key ---")
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

print(f"Key loaded: n={rlce_key.n}, k={rlce_key.k}, t={rlce_key.t}")
print(f"Graph: {NUM_CHECKS} check nodes, {NUM_VARS} variable nodes")

# ============================================================================
# DATA GENERATOR
# ============================================================================
def data_generator(h_matrix, num_vars, error_counts, num_to_label):
    """
    Generate training examples
    - If num_to_label == 'all': labels all errors
    - Otherwise: labels random subset and masks unlabeled ones
    """
    while True:
        total_errors = np.random.choice(error_counts)
        
        # Generate errors
        all_error_indices = np.random.choice(num_vars, total_errors, replace=False)
        error_vector = np.zeros(num_vars, dtype=np.float32)
        error_vector[all_error_indices] = 1.0
        
        # Calculate syndrome
        syndrome_float = np.dot(h_matrix, error_vector)
        syndrome = (syndrome_float.astype(np.int32)) % 2
        
        # Create labels and weights
        if num_to_label == 'all' or num_to_label >= total_errors:
            # Label all errors
            label = error_vector.copy()
            sample_weights = np.ones(num_vars, dtype=np.float32)
            sample_weights[all_error_indices] = SAMPLE_WEIGHT_ON_ERRORS
        else:
            # Label subset and mask unlabeled errors
            labeled_indices = np.random.choice(all_error_indices, num_to_label, replace=False)
            unlabeled_indices = np.setdiff1d(all_error_indices, labeled_indices)
            
            label = np.zeros(num_vars, dtype=np.float32)
            label[labeled_indices] = 1.0
            
            sample_weights = np.ones(num_vars, dtype=np.float32)
            sample_weights[unlabeled_indices] = 0.0  # Mask unlabeled
            sample_weights[labeled_indices] = SAMPLE_WEIGHT_ON_ERRORS
        
        yield syndrome, label, sample_weights

# ============================================================================
# CUSTOM TRAINING WRAPPER (for weighted loss)
# ============================================================================
class WeightedRGNN(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
        self.precision_tracker = tf.keras.metrics.Precision(name="precision")
        self.recall_tracker = tf.keras.metrics.Recall(name="recall")
        
    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)
    
    def train_step(self, data):
        x, y_true, sample_weights = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            loss = tf.reduce_mean(bce * sample_weights)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y_true, y_pred)
        self.precision_tracker.update_state(y_true, y_pred)
        self.recall_tracker.update_state(y_true, y_pred)
        
        return {
            "loss": self.loss_tracker.result(),
            "binary_accuracy": self.accuracy_tracker.result(),
            "precision": self.precision_tracker.result(),
            "recall": self.recall_tracker.result()
        }
    
    def test_step(self, data):
        x, y_true, sample_weights = data
        y_pred = self(x, training=False)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        loss = tf.reduce_mean(bce * sample_weights)
        
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y_true, y_pred)
        self.precision_tracker.update_state(y_true, y_pred)
        self.recall_tracker.update_state(y_true, y_pred)
        
        return {
            "loss": self.loss_tracker.result(),
            "binary_accuracy": self.accuracy_tracker.result(),
            "precision": self.precision_tracker.result(),
            "recall": self.recall_tracker.result()
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker, self.precision_tracker, self.recall_tracker]

# ============================================================================
# MODEL CREATION OR LOADING
# ============================================================================
print("\n--- Model Setup ---")

if LOAD_PRETRAINED:
    print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
    base_model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, custom_objects={'rgnn': rgnn})
    print("Pretrained model loaded.")
else:
    print("Creating new model...")
    base_model = rgnn(
        adjacency_lists=ADJACENCY_DATA,
        num_iterations=NUM_ITERATIONS,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    )
    
    # Build model
    dummy_syndrome = tf.zeros(NUM_CHECKS, dtype=tf.int32)
    dummy_graph = rgnn.create_graph_tensor(dummy_syndrome, ADJACENCY_DATA)
    _ = base_model(dummy_graph.merge_batch_to_components())
    print("New model created and built.")

base_model.summary()

# Wrap model for weighted training
model = WeightedRGNN(base_model)
dummy_graph = rgnn.create_graph_tensor(tf.zeros(NUM_CHECKS, dtype=tf.int32), ADJACENCY_DATA)
_ = model(dummy_graph.merge_batch_to_components())

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
model.compile(optimizer=optimizer, weighted_metrics=[])

print(f"\nTraining configuration:")
print(f"  Find {NUM_ERRORS_TO_LABEL} error(s) from {NUM_ERRORS_TOTAL}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

# ============================================================================
# DATA PIPELINE
# ============================================================================
print("\n--- Building Data Pipeline ---")

output_signature = (
    tf.TensorSpec(shape=(NUM_CHECKS,), dtype=tf.int32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32)
)

def create_graph_map_fn(syndrome, error_vector, sample_weights):
    graph = rgnn.create_graph_tensor(syndrome, ADJACENCY_DATA)
    return graph, error_vector, sample_weights

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(H_matrix_np, NUM_VARS, NUM_ERRORS_TOTAL, NUM_ERRORS_TO_LABEL),
    output_signature=output_signature
)
validation_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(H_matrix_np, NUM_VARS, NUM_ERRORS_TOTAL, NUM_ERRORS_TO_LABEL),
    output_signature=output_signature
)

train_dataset = (train_dataset
                 .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .map(lambda g, l, w: (g.merge_batch_to_components(), tf.reshape(l, [-1]), tf.reshape(w, [-1])))
                 .prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset
                      .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .map(lambda g, l, w: (g.merge_batch_to_components(), tf.reshape(l, [-1]), tf.reshape(w, [-1])))
                      .prefetch(tf.data.AUTOTUNE))

# Validate pipeline
print("Validating data pipeline...")
for batch in train_dataset.take(1):
    _, labels, weights = batch
    print(f"  Labels sum: {tf.reduce_sum(labels).numpy()} (expect ~{NUM_ERRORS_TO_LABEL * BATCH_SIZE})")
    print(f"  Weights range: [{tf.reduce_min(weights).numpy()}, {tf.reduce_max(weights).numpy()}]")
print("Data pipeline ready.")

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH + '_checkpoint',
        monitor='val_recall',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_recall',
        patience=20,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

original_verbosity = logging.get_verbosity()
logging.set_verbosity(logging.ERROR)

try:
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
        callbacks=callbacks
    )
finally:
    logging.set_verbosity(original_verbosity)

# ============================================================================
# SAVE AND SUMMARY
# ============================================================================
print("\n--- Saving Model ---")
model.base_model.save(MODEL_SAVE_PATH, save_format='tf')
print(f"Model saved to: {MODEL_SAVE_PATH}")

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Configuration: Find {NUM_ERRORS_TO_LABEL} from {NUM_ERRORS_TOTAL} errors")
print(f"Model: {NUM_ITERATIONS} iterations, {HIDDEN_DIM} hidden dim")
print(f"Initial loss: {history.history['loss'][0]:.4f}")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
print(f"Best val_recall: {max(history.history['val_recall']):.4f}")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print("="*60)