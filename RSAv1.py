# Import necessary modules from pycryptodome
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# This module converts binary data to hexadecimal
from binascii import hexlify

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


def enc(plaintext: str, public_key) -> bytes:
    # Step 2: Encrypt using public key
    # Create a PKCS1_OAEP cipher object with the public key for encryption
    data_to_encrypt = b"Test"
    cipher_rsa = PKCS1_OAEP.new(public_key)

    # Encrypt the provided data using the public key
    encrypted = cipher_rsa.encrypt(data_to_encrypt)

    return encrypted

def dec(encrypted: bytes, private_key) -> bytes:
    # Step 3: Decrypt using private key
    # Create a PKCS1_OAEP cipher object with the private key for decryption
    cipher_rsa = PKCS1_OAEP.new(private_key)
    decrypted = cipher_rsa.decrypt(encrypted)

    return decrypted

if __name__ == "__main__":
    # Step 1: Generate new RSA key
    # Create an RSA key pair with a key size of 1024 bits
    key = RSA.generate(1024)

    # Set the private_key variable to the generated key
    private_key = key
    # Derive the public key from the generated key
    public_key = key.publickey()

    # Display the public keys in hexadecimal format
    print("Public Key (hex), Private Key (hex):")
    printf(str(hexlify(public_key.export_key(format='DER'))) + "\n" + str(hexlify(private_key.export_key(format='DER'))), 'blue')

    ciphertext = enc("Test", public_key)
    # Convert binary data to hexadecimal for display using hexlify
    print("Encrypted:")
    printf(hexlify(ciphertext), 'green')

    m_dec = dec(ciphertext, private_key)
    # Display the decrypted result as a UTF-8 encoded string
    print("Decrypted:")
    printf(m_dec.decode("utf-8"), 'red')