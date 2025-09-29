import numpy as np
import torch
import galois
import torch.nn as nn
import time
from main import KeyGen, encrypt, decrypt


# ==============================================================================
# -------------------- SYNDROME-RNN DECODER (THE ATTACK) -----------------------
# ==============================================================================

# --- RNN PARAMETERS FOR ATTACK SIMULATION ---
# These parameters are for the *simulated* attack target, scaled down 
# from the massive size of the actual RLCE public key.
N_ATTACK = 60  # Simulated Codeword length (N)
K_ATTACK = 40  # Simulated Message dimension (K)
M_ATTACK = N_ATTACK - K_ATTACK # Syndrome length (M)
SNR_DB = 1.0       # Signal-to-Noise Ratio (simulating channel noise)
ERROR_WEIGHT = 5   # Simulated errors introduced (t)

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
        sequence_input = combined_input.unsqueeze(1).repeat(1, self.n, 1)
        
        # Forward pass through the LSTM
        rnn_output, _ = self.rnn(sequence_input)
        
        # Use the output of the final time step
        final_rnn_output = rnn_output[:, -1, :] 
        
        # Predict the N-bit error vector
        final_output = self.fc_out(final_rnn_output) 
        
        # Convert logits to probability (0 to 1)
        error_probability = torch.sigmoid(final_output)
        
        # Hard decision: predict error (1) if probability > 0.5
        predicted_error = (error_probability > 0.5).int()
        
        return predicted_error, error_probability

# --- ATTACK HELPER FUNCTIONS ---

def generate_rlce_H():
    """Generates a dense random binary H matrix for the attack target simulation."""
    N, M = N_ATTACK, M_ATTACK
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
    one_indices = np.random.choice(N, size=ERROR_WEIGHT, replace=False)
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
    
    return (
        torch.tensor(rnn_input, dtype=torch.float32).unsqueeze(0), 
        torch.tensor(true_error_vector, dtype=torch.float32).unsqueeze(0), 
        syndrome
    )

# ==============================================================================
# -------------------- MAIN EXECUTION BLOCK ------------------------------------
# ==============================================================================

if __name__ == "__main__":
    
    # --- PART 1: Run Original RLCE Validation (Using imported functions) ---
    
    n = 255 
    k = 235 
    t = 10 
    r = 1 
    
    print("-" * 70)
    print("PART 1: ORIGINAL RLCE ENCRYPTION/DECRYPTION VALIDATION (via import)")
    
    try:
        start_time = time.time()
        public_key, private_key = KeyGen(n=n, k=k, t=t, r=r)
        GF = private_key["GF"]
        message_to_encrypt = GF.Random(k)
        ciphertext = encrypt(public_key, message_to_encrypt, weight=t)
        decrypted_message = decrypt(private_key, ciphertext)
        is_correct = np.array_equal(message_to_encrypt, decrypted_message)
        end_time = time.time()
        
        print(f"Time: {end_time - start_time:.4f}s")
        print(f"Original System Params: n={n}, k={k}, t={t}, r={r}")
        print("Success:", is_correct, "✅" if is_correct else "❌")
    except Exception as e:
        print(f"RLCE Validation Failed: {e}")
        print("Please ensure your 'rlce_encryption.py' file is accessible and correct.")
        
    print("-" * 70)
    
    # --- PART 2: Syndrome-RNN Cryptanalytic Attack Simulation ---
    
    print("\nPART 2: SYNDROME-RNN ATTACK SIMULATION")
    print("Simulating attack on scaled-down binary target (N=60, K=40)")
    print("-" * 70)
    
    # 1. Define the random binary H matrix (the structure the RNN attacks)
    H_rlce_target = generate_rlce_H()
    
    # 2. Generate the necessary inputs for the RNN
    combined_input, true_error, syndrome = generate_noisy_input(H_rlce_target)
    
    # 3. Initialize the Syndrome-RNN Decoder model
    model = SyndromeRNN(N_ATTACK, M_ATTACK)
    
    print(f"Attack Params: N={N_ATTACK}, K={K_ATTACK}, M={M_ATTACK}, Errors={ERROR_WEIGHT}, SNR={SNR_DB} dB")
    print("Model Status: UNTRAINED (Real attack requires millions of training examples)")

    # 4. Run the Decoder (Inference)
    attack_start_time = time.time()
    with torch.no_grad():
        predicted_error, _ = model(combined_input)
    attack_end_time = time.time()
        
    # 5. Analyze Results
    true_error_int = true_error.int()
    num_correct_bits = (predicted_error == true_error_int).sum().item()
    num_errors_in_prediction = N_ATTACK - num_correct_bits
    
    print(f"Inference Time: {attack_end_time - attack_start_time:.6f}s")
    print("\nSyndrome RNN Decoding Results:")
    print(f"Input Syndrome (Binary): {syndrome.tolist()}")
    print(f"True Error Vector:       {true_error_int.tolist()[0]}")
    print(f"Predicted Error Vector:  {predicted_error.tolist()[0]}")
    
    print("\n--- ATTACK OUTCOME ---")
    if num_errors_in_prediction == 0:
        print(f"Perfect Decoding Success! The RNN correctly predicted the error vector.")
    else:
        print(f"Decoded with {num_errors_in_prediction} errors remaining. (Attack Failure - Requires Training)")
    print("-" * 70)
