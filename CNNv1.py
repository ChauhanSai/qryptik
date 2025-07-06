import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import base64

def printf(p, color):
    """
    Custom print function to format output with color.
    :param p: The message to print.
    :param color: Color code for terminal output.
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors[color]}{p}{colors['reset']}\n")


def build_keygen_cnn(input_shape=(32, 32, 1)):
    # Build CNN model
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='sigmoid')  # 32-byte output
    ])
    return model


def generate_seed(model, input_data):
    # Generate CNN seed
    pred = model.predict(input_data[np.newaxis, :, :, np.newaxis])
    seed = (pred[0] * 255).astype(np.uint8)

    print("Seed (CNN):")
    printf(seed, 'blue')

    return seed.tobytes()


from Crypto.Hash import SHAKE256

def derive_keypair(seed: bytes):
    # Expand seed and derive mock public/private key
    shake = SHAKE256.new()
    shake.update(seed)
    expanded = shake.read(64)
    public_key = expanded[:32]
    private_key = expanded[32:]
    return public_key, private_key


from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def enc(plaintext: str, public_key: bytes):
    # Encrypt text using public key (as AES-GCM key)
    key = SHAKE256.new(public_key).read(16)  # AES-128 key

    print("Key (hex):")
    printf(key.hex(), 'blue')

    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode())
    return {
        'ciphertext': base64.b64encode(ciphertext).decode(),
        'nonce': base64.b64encode(cipher.nonce).decode(),
        'tag': base64.b64encode(tag).decode()
    }

def dec(encrypted: dict, public_key: bytes):
    key = SHAKE256.new(public_key).read(16)
    cipher = AES.new(key, AES.MODE_GCM, nonce=base64.b64decode(encrypted['nonce']))
    plaintext = cipher.decrypt_and_verify(
        base64.b64decode(encrypted['ciphertext']),
        base64.b64decode(encrypted['tag'])
    )
    return plaintext.decode()

# Main execution
if __name__ == "__main__":
    # Use random noise as input
    random_input = np.random.rand(32, 32).astype(np.float32)

    # Build and use CNN
    model = build_keygen_cnn()
    seed = generate_seed(model, random_input)
    public_key, private_key = derive_keypair(seed)

    # Encrypt message
    encrypted = enc("Test", public_key)

    print("Encrypted:")
    printf(encrypted, 'green')

    # Decrypt (for demo)
    decrypted = dec(encrypted, public_key)
    print("Decrypted:")
    printf(decrypted, 'red')