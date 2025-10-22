import numpy as np
import torch
import galois
import torch.nn as nn
import torch.optim as optim # <-- NEW IMPORT for optimization
import time

# ==============================================================================
# -------------------- RNN PARAMETERS (ATTACK TARGET) --------------------------
# ==============================================================================

N_TARGET = 255 # Actual Codeword length (n) from KeyGen
K_TARGET = 235 # Actual Message dimension (k) from KeyGen
M_TARGET = N_TARGET - K_TARGET # syndrome length (n-k)
SNR_DB = 1.0  # signal to noise ratio
ERROR_WEIGHT_TARGET = 10 # max error weight

# ==============================================================================
# -------------------- SYNDROME-RNN DECODER (THE ATTACK) -----------------------
# ==============================================================================


class SyndromeRNN(nn.Module):
    """
    A Syndrome-Based Recurrent Neural Network (Syndrome-RNN) Decoder.
    Takes combined Syndrome + LLRs to predict the Error Vector.
    """
    def __init__(self, N, M, hidden_size=64, num_layers=2):
        super(SyndromeRNN, self).__init__()
        self.n = N 
        self.m = M 
        input_size = M + N # Syndrome length + LLR length
        
        # Recurrent Layer (LSTM) - The core learning component
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output Layer: Maps final hidden state to the N-bit error vector
        self.fc_out = nn.Linear(hidden_size, N)

    def forward(self, combined_input):
        # Unroll the single input vector N times to simulate iterative decoding over N bits
        # combined_input shape: (batch_size, M + N)
        # sequence_input shape: (batch_size, N, M + N)
        sequence_input = combined_input.unsqueeze(1).repeat(1, self.n, 1)
        
        # Forward pass through the LSTM
        rnn_output, _ = self.rnn(sequence_input)
        
        # Use the output of the final time step
        # final_rnn_output shape: (batch_size, hidden_size)
        final_rnn_output = rnn_output[:, -1, :] 
        
        # Predict the N-bit error vector logits
        # final_output shape: (batch_size, N)
        final_output = self.fc_out(final_rnn_output) 
        
        # Convert logits to probability (0 to 1) using Sigmoid
        error_probability = torch.sigmoid(final_output)
        
        # Hard decision: predict error (1) if probability > 0.5 (used for final prediction/analysis)
        predicted_error = (error_probability > 0.5).int()
        
        return predicted_error, error_probability

# ==============================================================================
# -------------------- ATTACK DATA & TRAINING LOGIC ----------------------------
# ==============================================================================

def train_syndrome_rnn(model, H_matrix, num_epochs=50
, batch_size=64, learning_rate=1e-3):
    """
    Trains the SyndromeRNN model using synthetically generated noisy samples.
    """
    # 1. Define Loss Function and Optimizer
    # BCELoss is used because the output (error_probability) is sigmoid-activated (0 to 1)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set the model to training mode
    model.train()
    
    # Define a number of steps per epoch to simulate continuous data generation
    steps_per_epoch = 1000 
    
    print(f"\n[TRAINING] Starting training for {num_epochs} epochs...")
    print(f"[TRAINING] {steps_per_epoch} steps per epoch, Batch Size: {batch_size}")
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        start_time = time.time()
        
        for step in range(steps_per_epoch):
            
            # --- Data Generation (Creates a batch of synthetic samples on the fly) ---
            batch_inputs, batch_labels = [], []
            for _ in range(batch_size):
                # Generate a single sample (input and true error vector/label)
                rnn_input, true_error, _ = generate_noisy_input(H_matrix)
                batch_inputs.append(rnn_input)
                batch_labels.append(true_error)
            
            # Combine individual tensors into batch tensors
            inputs = torch.cat(batch_inputs, dim=0)
            labels = torch.cat(batch_labels, dim=0) 
            
            # 2. PyTorch Training Step
            optimizer.zero_grad()      # Reset gradients
            _, predicted_probabilities = model(inputs) # Forward pass
            loss = criterion(predicted_probabilities, labels) # Compute loss
            loss.backward()            # Backpropagation (compute gradients)
            optimizer.step()           # Update weights
            
            total_loss += loss.item()
            
        avg_loss = total_loss / steps_per_epoch
        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s")

    print("[TRAINING] Training Complete.")
    return model

def generate_rlce_H():
    """Generates a dense random binary H matrix for the attack target simulation."""
    N, M = N_TARGET, M_TARGET
    H_random = np.random.randint(0, 2, size=(M, N), dtype=int)
    
    # Ensure all columns have at least one '1'
    for j in range(N):
        if np.sum(H_random[:, j]) == 0:
            H_random[np.random.randint(M), j] = 1
    return H_random

def generate_noisy_input(H):
    """Generates LLRs, true error vector, and Syndrome required by the RNN."""
    N = H.shape[1]
    
    # 1. Generate Error Vector (e)
    true_error_vector = np.zeros(N, dtype=int)
    one_indices = np.random.choice(N, size=ERROR_WEIGHT_TARGET, replace=False)
    true_error_vector[one_indices] = 1
    
    # Received codeword y (hard decision) = e
    received_codeword_hard = true_error_vector
    
    # 2. Calculate Syndrome (s = y * H^T)
    syndrome = (received_codeword_hard @ H.T) % 2
    
    # 3. Generate Soft-Decision LLRs
    sigma = 10**(-SNR_DB / 20)
    bpsk_signal = 1 - 2 * received_codeword_hard
    noise = np.random.normal(0, sigma, N)
    received_signal = bpsk_signal + noise 
    LLR_input = 2 * received_signal / (sigma**2)

    # Combine Syndrome and LLRs
    rnn_input = np.concatenate([syndrome, LLR_input])
    
    # Tensors must be float32 for model input/labels
    return (
        torch.tensor(rnn_input, dtype=torch.float32).unsqueeze(0), 
        torch.tensor(true_error_vector, dtype=torch.float32).unsqueeze(0), 
        syndrome
    )

# ==============================================================================
# -------------------- MAIN EXECUTION BLOCK ------------------------------------
# ==============================================================================

if __name__ == "__main__":
    
    # The original RLCE validation block has been removed as those functions
    # (KeyGen, encrypt, decrypt) were not available.
    
    print("-" * 70)
    print("SYNDROME-RNN ATTACK SIMULATION AND TRAINING")
    print("Simulating cryptanalytic attack on dense random binary code.")
    print("-" * 70)
    
    # 1. Define the random binary H matrix (the structure the RNN attacks)
    H_rlce_target = generate_rlce_H()
    
    # 2. Initialize the Syndrome-RNN Decoder model
    # Use larger hidden size and layers for a more robust (though slower) model
    model = SyndromeRNN(N_TARGET, M_TARGET, hidden_size=128, num_layers=3)
    
    print(f"Attack Params: N={N_TARGET}, K={K_TARGET}, M={M_TARGET}, Errors={ERROR_WEIGHT_TARGET}, SNR={SNR_DB} dB")
    
    # 3. TRAIN THE MODEL HERE
    # The training function is called to optimize the model's weights.
    trained_model = train_syndrome_rnn(
        model, 
        H_rlce_target, 
        num_epochs=50,
        batch_size=128, # Training samples per step
        learning_rate=5e-4
    )

    # 4. Run the Decoder (Inference on a fresh, unseen sample)
    print("\n--- INFERENCE ON A FRESH SAMPLE (Using Trained Model) ---")
    combined_input_test, true_error_test, syndrome_test = generate_noisy_input(H_rlce_target)
    
    # Set model to evaluation mode (disables dropout, etc.)
    trained_model.eval() 
    attack_start_time = time.time()
    with torch.no_grad(): # Disable gradient calculation for efficiency
        predicted_error, _ = trained_model(combined_input_test)
    attack_end_time = time.time()
        
    # 5. Analyze Results
    true_error_int = true_error_test.int()
    predicted_error_int = predicted_error.int()
    
    # Calculate difference (Hamming distance between true and predicted error vectors)
    error_vector_diff = (predicted_error_int != true_error_int).sum().item()
    
    print(f"Inference Time: {attack_end_time - attack_start_time:.6f}s")
    print("\nSyndrome RNN Decoding Results (Test Sample):")
    print(f"Input Syndrome (Binary): {syndrome_test.tolist()}")
    print(f"True Error Vector Hamming Weight: {true_error_int.sum().item()}")
    
    print("\n--- ATTACK OUTCOME ---")
    if error_vector_diff == 0:
        print("Perfect Decoding Success! The trained RNN correctly predicted the error vector.")
    else:
        print(f"Decoding Failure: {error_vector_diff} bit errors remain in the prediction.")
    print("-" * 70)
