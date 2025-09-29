import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import galois

# --- Assumed local imports ---
# Ensure you have these files in the same directory:
# 1. rlce_rs.py (the new system you provided)
# 2. syndrome_cnn.py (your CNN model definition)
from rlce_rs import rlce_rs
from syndrome_cnn import build_sequential_syndrome_cnn

# --- Configuration for RLCE-RS System ---
N_RS = 63  # RS code length
K_RS = 51  # RS message dimension
T_RS = 6   # Error correction capability ('t')
R_RS = 1   # Dimension of random matrices

# --- Training Hyperparameters ---
SAMPLES_PER_EPOCH = 320000  # 2500 steps * 128 batch size
EPOCHS = 50
VALIDATION_SAMPLES = 6400 # 50 steps * 128 batch size

def binarize_data(gf_array, expected_len, gf_field):
    """
    Helper function to convert a Galois Field array into its
    binary representation for the neural network.
    """
    ndarray_view = gf_array.view(np.ndarray)
    unpacked = np.unpackbits(ndarray_view.view(np.uint8), axis=1, bitorder="little")
    return unpacked[:, :expected_len * gf_field.degree].astype(np.float32)

def generate_epoch_data(num_samples, H_matrix, n_dim, max_error_weight, gf_field):
    """
    Generates a full dataset for one epoch and returns it as NumPy arrays.
    This avoids multi-threading issues with Python generators in Keras.
    """
    syndrome_len_gf = H_matrix.shape[0]
    
    # Initialize arrays for the full dataset
    syndromes_gf = gf_field.Zeros((num_samples, syndrome_len_gf))
    errors_binary = np.zeros((num_samples, n_dim), dtype=np.float32)

    print(f"Generating {num_samples} samples...")
    for i in range(num_samples):
        # Focus 100% on generating errors at or beyond the code's capability.
        hard_error_cases = [max_error_weight, max_error_weight + 1, max_error_weight + 2]
        error_weight = np.random.choice(hard_error_cases)
        error_weight = min(error_weight, n_dim)

        # Create an error vector with the chosen weight
        e = gf_field.Zeros(n_dim)
        error_indices = np.random.choice(n_dim, size=error_weight, replace=False)
        error_values = gf_field.Random(error_weight, low=1)
        e[error_indices] = error_values
        
        # Calculate the syndrome: s = H * e^T
        s = e @ H_matrix.T
        
        # Store the results
        syndromes_gf[i] = s
        errors_binary[i, error_indices] = 1.0

    # Binarize the entire dataset at once for efficiency
    syndromes_binary = binarize_data(syndromes_gf, syndrome_len_gf, gf_field)
    
    return syndromes_binary, errors_binary

if __name__ == '__main__':
    print("--- Setting up RLCE-RS environment ---")
    encryptor = rlce_rs(n=N_RS, k=K_RS, t=T_RS, r=R_RS)
    G_pub = encryptor.public_key['G']
    GF = encryptor.GF
    print(f"Using {GF.name} for calculations.")

    print("Deriving public Parity Check Matrix (H) in systematic form...")
    try:
        G_sys = G_pub.row_reduce()
        k_dim = G_sys.shape[0]
        n_dim = G_sys.shape[1]
        
        if np.linalg.matrix_rank(G_sys) != k_dim:
            print("ERROR: Public generator matrix G_pub is not full rank. Cannot create H.")
            exit()
            
        P = G_sys[:, k_dim:]
        identity_matrix = GF.Identity(n_dim - k_dim)
        H_pub = np.hstack([-P.T, identity_matrix])
        H_pub = GF(H_pub)
    except Exception as e:
        print(f"An error occurred while deriving H: {e}")
        exit()

    print("Building the CNN model...")
    syndrome_len_gf = H_pub.shape[0]
    syndrome_len_binary = syndrome_len_gf * GF.degree
    model = build_sequential_syndrome_cnn(syndrome_length=syndrome_len_binary, n=n_dim)
    model.summary()

    # Generate the validation data ONCE.
    print("\n--- Generating Validation Data ---")
    X_val, y_val = generate_epoch_data(VALIDATION_SAMPLES, H_pub, n_dim, T_RS, GF)

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath='rlce_decoder_best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1,
        restore_best_weights=True
    )

    print("\n--- Starting Model Training ---")
    # This loop trains for one epoch at a time, generating new data for each one.
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
        
        # Generate fresh training data for this epoch
        print("--- Generating Training Data for Epoch ---")
        X_train, y_train = generate_epoch_data(SAMPLES_PER_EPOCH, H_pub, n_dim, T_RS, GF)
        
        # Train the model on the generated data
        history = model.fit(
            X_train, y_train,
            batch_size=128,
            epochs=1, # Train for a single epoch on the generated data
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback, early_stopping_callback],
            verbose=1
        )
        
        # Check if early stopping was triggered
        if model.stop_training:
            print("Early stopping triggered. Ending training.")
            break

    print("\n--- Training complete. ---")
    print("The final model state can be found in the best checkpoint: 'rlce_decoder_best_model.keras'.")

