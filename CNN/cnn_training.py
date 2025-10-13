import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import galois

# --- Assumed local imports ---
from rlce_rs import rlce_rs
from syndrome_cnn import build_sequential_syndrome_cnn

# --- Configuration for RLCE-RS System ---
N_RS = 63   # RS code length
K_RS = 51   # RS message dimension
T_RS = 6    # Error correction capability ('t')
R_RS = 1    # Dimension of random matrices

# --- Training Hyperparameters ---
EPOCHS = 200
BATCH_SIZE = 128
STEPS_PER_EPOCH = 3000 # SAMPLES_PER_EPOCH // BATCH_SIZE
VALIDATION_SAMPLES = 6400

def binarize_data(gf_array, expected_len, gf_field):
    """
    Helper function to convert a Galois Field array into its
    binary representation for the neural network.
    """
    ndarray_view = gf_array.view(np.ndarray)
    unpacked = np.unpackbits(ndarray_view.view(np.uint8), axis=1, bitorder="little")
    return unpacked[:, :expected_len * gf_field.degree].astype(np.float32)

def generate_data_batch(num_samples, H_matrix, n_dim, max_error_weight, gf_field):
    """
    Generates a single batch of data using efficient vectorized operations.
    This function was previously named generate_epoch_data.
    """
    syndrome_len_gf = H_matrix.shape[0]
    syndromes_gf = gf_field.Zeros((num_samples, syndrome_len_gf))
    errors_binary = np.zeros((num_samples, n_dim), dtype=np.float32)

    hard_error_cases = [max_error_weight + 1, max_error_weight + 2]
    error_weights = np.random.choice(hard_error_cases, size=num_samples)
    error_weights = np.minimum(error_weights, n_dim)

    for w in np.unique(error_weights):
        sample_indices = np.where(error_weights == w)[0]
        batch_size = len(sample_indices)
        if batch_size == 0:
            continue

        rand_matrix = np.random.rand(batch_size, n_dim)
        error_locs = np.argpartition(rand_matrix, -w, axis=1)[:, -w:]
        error_vals = gf_field.Random(batch_size * w, low=1).reshape(batch_size, w)
        e_batch = gf_field.Zeros((batch_size, n_dim))
        np.put_along_axis(e_batch, error_locs, error_vals, axis=1)
        s_batch = e_batch @ H_matrix.T
        
        syndromes_gf[sample_indices] = s_batch
        np.put_along_axis(errors_binary[sample_indices], error_locs, 1.0, axis=1)

    syndromes_binary = binarize_data(syndromes_gf, syndrome_len_gf, gf_field)
    return syndromes_binary, errors_binary

def data_generator(batch_size, H_matrix, n_dim, max_error_weight, gf_field, noise_factor=0.0):
    """
    A Python generator that yields batches of training data indefinitely,
    with optional noise addition.
    """
    while True:
        # 1. Generate one clean batch of data
        X_batch, y_batch = generate_data_batch(
            batch_size, H_matrix, n_dim, max_error_weight, gf_field
        )
        
        # 2. Add Gaussian noise to the syndrome data if a noise factor is provided
        if noise_factor > 0.0:
            noise = tf.random.normal(shape=tf.shape(X_batch), stddev=noise_factor)
            X_batch = X_batch + noise
        
        # 3. Yield the (potentially noisy) batch
        yield X_batch, y_batch

if __name__ == '__main__':
    print("--- Setting up RLCE-RS environment ---")
    encryptor = rlce_rs.load('rlce_key_01.npz')
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

    
    NOISE_FACTOR = 0.0  # Standard deviation of Gaussian noise added to inputs


    # Generate the validation data ONCE. This needs to be small enough
    # to fit comfortably in memory along with the model.
    print("\n--- Generating Validation Data ---")
    X_val, y_val = generate_data_batch(VALIDATION_SAMPLES, H_pub, n_dim, T_RS, GF)

    print(f"--- Adding noise (factor: {NOISE_FACTOR}) to validation data ---")
    val_noise = tf.random.normal(shape=tf.shape(X_val), stddev=NOISE_FACTOR)
    X_val_noisy = X_val + val_noise

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
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,     # Reduce LR by a factor of 5 (1 * 0.2)
        patience=3,     # Reduce if val_loss doesn't improve for 3 epochs
        min_lr=1e-7,    # Don't let the LR go below this value
        verbose=1
    )

    print("\n--- Starting Model Training ---")
    
    # # Create the training data generator
    train_gen = data_generator(BATCH_SIZE, H_pub, n_dim, T_RS, GF, noise_factor=NOISE_FACTOR)
    

    #
    # INITIAL TRAINING RUN
    #

    # Use model.fit with the generator. Keras will handle the epoch loop
    # and fetch batches from the generator as needed.
    history = model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=(X_val_noisy, y_val),
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1
    )

    print("\n--- Training complete. ---")
    print("The final model state can be found in the best checkpoint: 'rlce_decoder_best_model.keras'.")


    #
    # FINE-TUNING RUN (Optional)
    #

    # print("\n--- Loading best model for FINE-TUNING ---")
    # # Load the model you already trained
    # model = tf.keras.models.load_model('rlce_decoder_best_model.keras')

    # # Set a new, much smaller learning rate
    # fine_tune_lr = 1e-6 
    # model.optimizer.lr.assign(fine_tune_lr)
    # print(f"New learning rate for fine-tuning set to: {model.optimizer.lr.numpy()}")

    # # Set this to the epoch number where your last training stopped
    # initial_epoch_count = 100

    # print("\n--- Continuing Training (Fine-Tuning) ---")
    # history_fine_tune = model.fit(
    #     train_gen,
    #     steps_per_epoch=STEPS_PER_EPOCH,
    #     epochs=initial_epoch_count + 50, # Train for 50 more epochs
    #     initial_epoch=initial_epoch_count, # Start the epoch counter here
    #     validation_data=(X_val_noisy, y_val),
    #     callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
    #     verbose=1
    # )
    # print("--- Fine-tuning complete. ---")