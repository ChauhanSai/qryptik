# ----------------------------------------------------------------------------
# OpenSSH 10.0 has been officially released, introducing a number of protocol
# changes and security upgrades, including a key enhancement for post-quantum
# security. The release makes the mlkem768x25519-sha256 algorithm the default
# for key agreement. This hybrid algorithm combines ML-KEM (a NIST-standardized
# key encapsulation mechanism) with the classical X25519 elliptic curve method,
# offering quantum-resistant properties while maintaining compatibility and
# performance.
# https://quantumcomputingreport.com/openssh-10-0-introduces-default-post-quantum-key-exchange-algorithm/
# ----------------------------------------------------------------------------

from pqc.kem import kyber768
import base64

kyber768.patent_notice(patents=[''],subject='',severity=0,links=[''],stacklevel=0)

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


if __name__ == "__main__":
    # Key generation
    public_key, secret_key = kyber768.keypair()
    # Generate a shared secret and encapsulate
    shared_secret_enc, ciphertext = kyber768.encap(public_key)

    print("Ciphertext (Base64):")
    printf(base64.b64encode(ciphertext).decode(), 'green')

    # Receiver decrypts it
    shared_secret_dec = kyber768.decap(ciphertext, secret_key)

    # Verify both sides match
    assert shared_secret_enc == shared_secret_dec, "Shared secrets do not match!"
    print("Shared Secret:")
    printf(base64.b64encode(shared_secret_enc).decode(), 'blue')

    # You can now use the shared_secret as a key to encrypt the actual text.
    # For example, use AES-GCM (from Cryptography library)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import os

    # Use first 32 bytes of shared secret as AES-256 key
    aes_key = shared_secret_enc[:32]
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)

    # Encrypt the message
    plaintext = "Test"
    plaintext_bytes = plaintext.encode()

    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
    print("AES-GCM Encrypted Message (Base64):")
    printf(base64.b64encode(ciphertext).decode(), 'green')

    # Decrypt
    decrypted = aesgcm.decrypt(nonce, ciphertext, None)
    print("Decrypted Message:")
    printf(decrypted.decode(), 'red')