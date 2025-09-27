import numpy as np
import tensorflow as tf
import galois

# --- Assumed local imports ---
# Ensure you have these files in the same directory:
# 1. syndrome_decoder_sequential.py (your model architecture)
# 2. rlce_rs.py (the new system you provided)
from syndrome_cnn import build_sequential_syndrome_cnn
from rlce_rs import rlce_rs

# --- Configuration for RLCE-RS System ---
# Parameters for the Reed-Solomon based code
N_RS = 15  # RS code length
K_RS = 7   # RS message dimension
T_RS = 4   # Error correction capability (MAX number of errors to inject)
R_RS = 1   # Dimension of random matrices

# --- Training Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 128
STEPS_PER_EPOCH = 2000
VALIDATION_STEPS = 50
LEARNING_RATE = 1e-4

# --- Data Generation using RLCE-RS ---

def training_data_generator(H_matrix, batch_size, codeword_len, max_error_weight, gf_field):
    """
    A Python generator that yields batches of binarized training data.
    It uses the RLCE-RS system to generate errors and syndromes over a Galois Field,
    then converts them to a binary format suitable for the CNN.
    
    Args:
        H_matrix (galois.FieldArray): The public Parity Check Matrix over the GF.
        batch_size (int): The number of samples per batch.
        codeword_len (int): The length of the public codeword.
        max_error_weight (int): The MAXIMUM number of non-zero errors to inject.
        gf_field (galois.GF): The Galois Field for calculations.
        
    Yields:
        tuple: A tuple containing (binarized_syndromes, error_locations) for one batch.
    """
    syndrome_len_gf = H_matrix.shape[0]
    
    while True:
        # --- Create Error Vectors over the Galois Field ---
        error_vectors_gf = gf_field.Zeros((batch_size, codeword_len))
        
        for i in range(batch_size):
            # Choose a *random* number of errors to inject for this sample,
            # from 1 up to the maximum capability. This makes the model more robust.
            num_errors = np.random.randint(1, max_error_weight + 1)
            
            # Choose 'num_errors' unique random positions for errors
            error_positions = np.random.choice(codeword_len, num_errors, replace=False)
            # Assign random non-zero values from the field to these positions
            error_values = gf_field.Random(num_errors, low=1)
            error_vectors_gf[i, error_positions] = error_values
            
        # --- Create Syndromes over the Galois Field ---
        # s = e * H^T (calculation is in the specified Galois Field)
        syndromes_gf = error_vectors_gf @ H_matrix.T
        
        # --- Binarize Data for the CNN ---
        
        # 1. Convert syndromes to binary representation
        # Each GF element becomes a binary vector of length 't' (e.g., GF(2^4) -> 4 bits)
        # First, view as a standard numpy array, then as uint8 for unpackbits.
        syndromes_as_ndarray = syndromes_gf.view(np.ndarray)
        syndromes_unpacked = np.unpackbits(syndromes_as_ndarray.view(np.uint8), axis=1, bitorder="little")
        binarized_syndromes = syndromes_unpacked[:, :syndrome_len_gf * gf_field.degree].astype(np.float32)

        # 2. Create binary error location vectors (the training labels)
        # The CNN will predict *where* errors are, not what their GF value is.
        error_locations = (error_vectors_gf != 0).astype(np.float32)

        yield (binarized_syndromes, error_locations)

# --- Main Training Execution ---
if __name__ == '__main__':
    print("--- Setting up RLCE-RS environment ---")
    # 1. Instantiate the RLCE-RS system to get the public key
    encryptor = rlce_rs(n=N_RS, k=K_RS, t=T_RS, r=R_RS)
    G_pub = encryptor.public_key['G']
    GF = encryptor.GF
    print(f"Using {GF.name} for calculations.")
    
    # 2. Derive the public Parity Check Matrix (H) from the public Generator Matrix (G)
    print("Deriving public Parity Check Matrix (H) in systematic form...")
    try:
        # Convert G to its systematic form G_sys = [I_k | P] by finding its Reduced Row Echelon Form.
        G_sys = G_pub.row_reduce()
    except np.linalg.LinAlgError:
        # This can happen if the randomly generated G is not full rank.
        print("\nERROR: The generated public key matrix G is not full rank.")
        print("This is a rare issue with random matrix generation. Please try running the script again.")
        exit()

    # Extract the P matrix from G_sys
    k_dim = G_sys.shape[0]
    P = G_sys[:, k_dim:]
    
    # Create the parity-check matrix H = [-P.T | I_(n-k)]
    n_dim = G_sys.shape[1]
    identity_matrix = GF.Identity(n_dim - k_dim)
    
    # The galois library overloads the transpose operator and concatenation
    H_pub = np.hstack([-P.T, identity_matrix])
    H_pub = GF(H_pub) # Ensure it's a Galois Field array
    
    # 3. Calculate the dimensions for the CNN model
    CODEWORD_LEN = G_pub.shape[1]
    SYNDROME_LEN_GF = H_pub.shape[0]
    # The input to the CNN is the binarized length of the syndrome
    SYNDROME_LEN_BINARY = SYNDROME_LEN_GF * GF.degree
    
    print(f"Public Codeword Length (CNN Output Dim): {CODEWORD_LEN}")
    print(f"Binarized Syndrome Length (CNN Input Dim): {SYNDROME_LEN_BINARY}")

    # 4. Build the CNN model with the new dimensions
    print("\nBuilding the CNN model...")
    model = build_sequential_syndrome_cnn(
        syndrome_length=SYNDROME_LEN_BINARY, 
        n=CODEWORD_LEN, 
        learning_rate=LEARNING_RATE
    )
    model.summary()
    
    # 5. Create the data generators for training and validation
    print("Setting up data generators...")
    train_generator = training_data_generator(H_pub, BATCH_SIZE, CODEWORD_LEN, T_RS, GF)
    validation_generator = training_data_generator(H_pub, BATCH_SIZE, CODEWORD_LEN, T_RS, GF)
    
    # 6. Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    print("--- Model Training Finished ---\n")
    
    # 7. Save the trained model for later use
    model.save('rlce_decoder_model.keras')
    print(f"Trained model saved to 'rlce_decoder_model.keras'")

