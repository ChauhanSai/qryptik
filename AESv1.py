from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

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

def enc(plaintext: str, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    return cipher.iv + ct_bytes  # Prepend IV for decryption

def dec(ciphertext: bytes, key: bytes) -> str:
    iv = ciphertext[:AES.block_size]
    ct = ciphertext[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()

# Example usage
if __name__ == "__main__":
    key = get_random_bytes(16)  # AES-128: In symmetric cryptography like AES, there is only one key, and both parties must share this secret key securely.
    print("Key (hex):")
    printf(key.hex(), 'blue')

    ciphertext = enc("Test", key)
    print("Ciphertext (hex):")
    printf(ciphertext.hex(), 'green')

    decrypted = dec(ciphertext, key)
    print("Decrypted:")
    printf(decrypted, 'red')
