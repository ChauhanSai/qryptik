# In generate_key.py

from rlce_rs import rlce_rs

# Define the parameters for your key
N_RS = 63
K_RS = 51
T_RS = 6
R_RS = 1

print("--- Generating new RLCE-RS key ---")
# Create a new instance, which generates the random key
encryptor = rlce_rs(n=N_RS, k=K_RS, t=T_RS, r=R_RS)

# Save the entire object to a file
encryptor.save('rlce_key_01.npz')

print("--- Key generation complete. ---")