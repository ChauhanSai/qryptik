import os
from rlce_rs import rlce_rs

# --- 1. CONFIGURATION ---
N_RS = 63
K_RS = 51
T_RS = 6
R_RS = 1

# --- New Configuration: Key Index and Directory ---
KEY_ID = 3  # Manually increment this for each new key (e.g., 1, 2, 3...)
SUBDIR = 'keys'
FILENAME = f'rlce_key_{KEY_ID:02d}.npz' # :02d ensures '01', '02', etc.
FULL_PATH = os.path.join(SUBDIR, FILENAME)


## 2. Key Generation Logic

print(f"--- Generating new RLCE-RS key (ID: {KEY_ID}) ---")

# A. Create the subdirectory if it doesn't exist
os.makedirs(SUBDIR, exist_ok=True)

# B. Create the new key instance
encryptor = rlce_rs(n=N_RS, k=K_RS, t=T_RS, r=R_RS, KEY_ID=KEY_ID)

# C. Save the entire object to the indexed file path
encryptor.save(FULL_PATH)

print(f"--- Key generation complete. Key saved to {FULL_PATH} ---")