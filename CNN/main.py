import numpy as np
import galois
import rlce_rs as rlce 
    
if __name__ == "__main__":
    
    # Use smaller, faster parameters for the demonstration
    n = 15  # Code length (must be <= GF order - 1)
    k = 7   # Message dimension
    t = 4   # Error correction capability (t <= (n-k)/2)
    r = 1   # Dimension of random matrices
    
    print("Generating keys...\n")
    encryptor = rlce.rlce_rs(n, k, t, r)
    
    
    print(f"Keys generated for parameters n={n}, k={k}, t={t}, r={r} over {encryptor.GF.name}")
    print(f"Public key matrix shape: {encryptor.public_key['G'].shape}\n")
    
    message_to_encrypt = encryptor.GF.Random(k)
    ciphertext = encryptor.encrypt(message_to_encrypt, encryptor.public_key)
    decrypted_message = encryptor.decrypt(ciphertext)
    is_correct = np.array_equal(message_to_encrypt, decrypted_message)
    print("Original Message: ", message_to_encrypt)
    print("Decrypted Message:", decrypted_message)
    print(f"Success: {is_correct} {'✅' if is_correct else '❌'}\n")