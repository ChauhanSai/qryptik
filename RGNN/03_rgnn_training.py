import os
import warnings
from absl import logging
# Suppress warnings at the very top, before other imports.
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.framework.indexed_slices')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow.python.saved_model.nested_structure_coder')

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
# Use the Keras callbacks compatible with your TensorFlow version
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Import your custom modules
from rgnn import rgnn
from rlce_rs import rlce_rs

# --- 1. Configuration and Setup ---
RLCE_KEY_NAME = 'keys/rlce_key_03.npz'  # Update to your t=8 key
rlce_key = rlce_rs.load(RLCE_KEY_NAME)
KEY_ID = rlce_key.KEY_ID
MODEL_ID = 3 # Increment for the new training run

# This should point to the base model you created
BASE_MODEL_FILENAME = f'models/untrained/base_rgnn_model_{KEY_ID}.02' 
# The checkpoint will save the best wrapped model
TRAINED_MODEL_FILENAME = f'models/trained/rgnn_model_{KEY_ID}.{MODEL_ID:02d}'
# After training, we'll extract and save just the base model
TRAINED_BASE_MODEL_FILENAME = f'models/trained/rgnn_base_model_{KEY_ID}.{MODEL_ID:02d}'

# Training Hyperparameters - START EASY
NUM_ERRORS_TO_TRAIN = [1]
BATCH_SIZE = 32
EPOCHS = 200
STEPS_PER_EPOCH = 500  # More data
VALIDATION_STEPS = 100
LEARNING_RATE = 5e-4  # Conservative learning rate

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

# --- 2. Weighted Data Generator ---
def data_generator_weighted(h_matrix, num_vars, possible_num_errors):
    """
    Generates syndrome, error vector, and sample weights.
    Sample weights heavily penalize misclassifying error positions.
    """
    while True:
        error_vector = np.zeros(num_vars, dtype=np.float32)
        num_errors = np.random.choice(possible_num_errors)
        error_indices = np.random.choice(num_vars, num_errors, replace=False)
        error_vector[error_indices] = 1.0
        
        syndrome_float = np.dot(h_matrix, error_vector)
        syndrome = (syndrome_float.astype(np.int32)) % 2
        
        # Create sample weights: much higher weight for error positions
        # This tells the model "getting errors right is 15x more important"
        sample_weights = np.ones(num_vars, dtype=np.float32)
        sample_weights[error_indices] = 800.0
        
        yield syndrome, error_vector, sample_weights

# --- 3. Custom Training Wrapper ---
class WeightedRGNN(tf.keras.Model):
    """Wrapper that handles per-element sample weights in train_step"""
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
            # Forward pass will now return logits
            y_pred_logits = self(x, training=True)
            
            # Compute loss from logits for numerical stability
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred_logits, from_logits=True)
            weighted_bce = bce * sample_weights
            loss = tf.reduce_mean(weighted_bce)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        # For accuracy, convert logits to probabilities first
        self.accuracy_tracker.update_state(y_true, tf.sigmoid(y_pred_logits))
        self.precision_tracker.update_state(y_true, tf.sigmoid(y_pred_logits))
        self.recall_tracker.update_state(y_true, tf.sigmoid(y_pred_logits))
        
        return {
            "loss": self.loss_tracker.result(),
            "binary_accuracy": self.accuracy_tracker.result(),
            "precision": self.precision_tracker.result(), # Add this
            "recall": self.recall_tracker.result()       # Add this
        }
    
    def test_step(self, data):
        # Validation step (same as train but no gradient update)
        x, y_true, sample_weights = data
        
        # Forward pass
        y_pred = self(x, training=False)
        
        # Compute weighted loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        weighted_bce = bce * sample_weights
        loss = tf.reduce_mean(weighted_bce)
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(y_true, tf.sigmoid(y_pred))
        self.precision_tracker.update_state(y_true, tf.sigmoid(y_pred))
        self.recall_tracker.update_state(y_true, tf.sigmoid(y_pred))
        
        return {
            "loss": self.loss_tracker.result(),
            "binary_accuracy": self.accuracy_tracker.result(),
            "precision": self.precision_tracker.result(), # Add this
            "recall": self.recall_tracker.result()       # Add this
        }
    
    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

# --- 4. Model Loading and Wrapping ---
print("Loading base model...")
base_model = tf.keras.models.load_model(
    BASE_MODEL_FILENAME,
    custom_objects={'rgnn': rgnn}
)
print("Base model loaded successfully.")

# Wrap the model with our custom trainer
model = WeightedRGNN(base_model)

# Build the wrapper by calling it once
dummy_syndrome = tf.zeros((NUM_CHECKS,), dtype=tf.int32)
dummy_graph = rgnn.create_graph_tensor(dummy_syndrome, ADJACENCY_DATA)
_ = model(dummy_graph.merge_batch_to_components(), training=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, clipvalue=0.5)
model.compile(optimizer=optimizer, weighted_metrics=[])

print("\nWrapped model ready for training.")
model.base_model.summary()

# --- 5. Define Callbacks ---
print("\nSetting up training callbacks...")
checkpoint_callback = ModelCheckpoint(
    filepath=TRAINED_MODEL_FILENAME,
    monitor='val_recall',
    save_best_only=True,
    mode='max',
    verbose=1
)
# early_stopping_callback = EarlyStopping(
#     monitor='val_loss',
#     patience=15,  # More patience for weighted training
#     verbose=1,
#     restore_best_weights=True
# )
early_stopping_callback = EarlyStopping(
    monitor='val_recall',   # The metric to watch
    patience=20,            # How many epochs to wait for improvement
    mode='max',             # We want to maximize recall
    restore_best_weights=True # Automatically restore the best model weights at the end
)
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# --- 6. Create the tf.data.Dataset Pipeline with Weights ---
print("\nBuilding tf.data pipeline with sample weights...")
output_signature = (
    tf.TensorSpec(shape=(NUM_CHECKS,), dtype=tf.int32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32),
    tf.TensorSpec(shape=(NUM_VARS,), dtype=tf.float32)  # Sample weights
)

def create_graph_map_fn(syndrome, error_vector, sample_weights):
    graph = rgnn.create_graph_tensor(syndrome, ADJACENCY_DATA)
    return graph, error_vector, sample_weights

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_weighted(H_matrix_np, NUM_VARS, NUM_ERRORS_TO_TRAIN),
    output_signature=output_signature
)
validation_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator_weighted(H_matrix_np, NUM_VARS, NUM_ERRORS_TO_TRAIN),
    output_signature=output_signature
)

# Process datasets: batch, merge graphs, reshape outputs
train_dataset = (train_dataset
                 .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(BATCH_SIZE)
                 .map(lambda graph, labels, weights: (
                     graph.merge_batch_to_components(), 
                     tf.reshape(labels, [-1]),
                     tf.reshape(weights, [-1])
                 ))
                 .prefetch(tf.data.AUTOTUNE))

validation_dataset = (validation_dataset
                      .map(create_graph_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(BATCH_SIZE)
                      .map(lambda graph, labels, weights: (
                          graph.merge_batch_to_components(), 
                          tf.reshape(labels, [-1]),
                          tf.reshape(weights, [-1])
                      ))
                      .prefetch(tf.data.AUTOTUNE))

print("Data pipeline built successfully.")

# --- 7. Validate Data Pipeline ---
print("\nValidating data pipeline...")
for batch_data in train_dataset.take(1):
    graph_batch, labels_batch, weights_batch = batch_data
    print(f"Graph batch type: {type(graph_batch)}")
    print(f"Labels shape: {labels_batch.shape}, dtype: {labels_batch.dtype}")
    print(f"Weights shape: {weights_batch.shape}, dtype: {weights_batch.dtype}")
    print(f"Labels - Min: {tf.reduce_min(labels_batch)}, Max: {tf.reduce_max(labels_batch)}")
    print(f"Weights - Min: {tf.reduce_min(weights_batch)}, Max: {tf.reduce_max(weights_batch)}")
    print(f"Sum of labels (should be ~num_errors * batch_size): {tf.reduce_sum(labels_batch)}")
    print("Data pipeline validation successful.\n")

# --- 8. Model Training ---
original_verbosity = logging.get_verbosity()
try: # Set higher verbosity to see training progress without annoying warnings
    logging.set_verbosity(logging.ERROR)
    print("Starting model training with weighted loss...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
        callbacks=[checkpoint_callback,early_stopping_callback, reduce_lr_callback]
    )
    print("Training complete.") 
finally:
    logging.set_verbosity(original_verbosity)


# --- 9. Extract and Save Base Model ---
print(f"\nExtracting base model and saving to {TRAINED_BASE_MODEL_FILENAME}...")
model.base_model.save(TRAINED_BASE_MODEL_FILENAME, save_format='tf')
print("Base model saved successfully. Use this file for inference.")

# --- 10. Training Summary ---
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Initial loss: {history.history['loss'][0]:.4f}")
print(f"Final loss: {history.history['loss'][-1]:.4f}")
print(f"Loss improvement: {history.history['loss'][0] - history.history['loss'][-1]:.4f}")
print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
print(f"Best model wrapper saved to: {TRAINED_MODEL_FILENAME}")
print(f"Base model (for inference) saved to: {TRAINED_BASE_MODEL_FILENAME}")
print("="*60)