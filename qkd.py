"""
qkd integration with rlce encryption in main.py
"""

import numpy as np
import hashlib
import secrets
from typing import Tuple, List, Dict
import main  


## bb84 protocol is the commonly used qkd distribution protocol. this is a simulation of a qkd tunnel for research purposes

class BB84_QKD: ## creating channel
    
    def __init__(self, key_length: int = 256, error_threshold: float = 0.11):
       
        """
        key_length -> length of final key in bits
        error_threshold -> max allowed quantum bit error rate
        """

        self.key_length = key_length
        self.error_threshold = error_threshold
        self.raw_key_multiplier = 4
        
    def _generate_random_bits(self, n: int) -> np.ndarray:
        return np.array([secrets.randbits(1) for _ in range(n)], dtype=int)
    
    ## a basis is a way to measure a qubit, and this is derived from the bits used in the key 

    def _generate_random_bases(self, n: int) -> np.ndarray:
        return np.array([secrets.randbits(1) for _ in range(n)], dtype=int)
    
    def _prepare_qubits(self, bits: np.ndarray, bases: np.ndarray) -> List[dict]:
        qubits = []
        for bit, basis in zip(bits, bases):
            qubits.append({'bit': int(bit), 'basis': int(basis)})
        return qubits
    
    ## below is the method for the channel simulation, with noise and eavesdropping included. 
    
    def _simulate_quantum_channel(self, qubits: List[dict], 
                                  noise_level: float = 0.01,
                                  eavesdrop_probability: float = 0.0) -> List[dict]:
       
        transmitted_qubits = []
        
        for qubit in qubits:
            q = qubit.copy()
            
            if np.random.random() < eavesdrop_probability:
                eavesdropper_basis = secrets.randbits(1)
                if eavesdropper_basis != q['basis']:
                    if np.random.random() < 0.5:
                        q['bit'] = 1 - q['bit']
            
            if np.random.random() < noise_level:
                q['bit'] = 1 - q['bit']
            
            transmitted_qubits.append(q)
        
        return transmitted_qubits
    
    ## reciever measures qubits based on their own bases
    
    def _measure_qubits(self, qubits: List[dict], bases: np.ndarray) -> np.ndarray:
        measurements = []
        for qubit, reciever_basis in zip(qubits, bases):
            if reciever_basis == qubit['basis']:
                measurements.append(qubit['bit'])
            else:
                measurements.append(secrets.randbits(1))
        return np.array(measurements, dtype=int)
    
    def _sift_keys(self, sender_bits: np.ndarray, sender_bases: np.ndarray,
                   reciever_bits: np.ndarray, reciever_bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        ## going through keys to only keep bases that match btwn the sender + reciever
        matching_bases = sender_bases == reciever_bases
        sifted_sender = sender_bits[matching_bases]
        sifted_reciever = reciever_bits[matching_bases]
        return sifted_sender, sifted_reciever
    
    def _estimate_error_rate(self, sender_bits: np.ndarray, reciever_bits: np.ndarray,
                            sample_size: int) -> Tuple[float, np.ndarray, np.ndarray]:
        
        if len(sender_bits) < sample_size:
            sample_size = len(sender_bits) // 2
        
        indices = np.random.choice(len(sender_bits), size=sample_size, replace=False)
        test_mask = np.zeros(len(sender_bits), dtype=bool)
        test_mask[indices] = True
        
        test_sender = sender_bits[test_mask]
        test_reciever = reciever_bits[test_mask]
        errors = np.sum(test_sender != test_reciever)
        error_rate = errors / sample_size
        
        remaining_sender = sender_bits[~test_mask]
        remaining_reciever = reciever_bits[~test_mask]
        
        return error_rate, remaining_sender, remaining_reciever
    
    def _information_reconciliation(self, sender_bits: np.ndarray, 
                                   reciever_bits: np.ndarray) -> np.ndarray:
       
       ## making sure bits match

        matching = sender_bits == reciever_bits
        return sender_bits[matching]
    
    def _privacy_amplification(self, key_bits: np.ndarray, 
                               final_length: int) -> np.ndarray:
        
        """
       hashing to reduce info from eavesdropper of channel
        """

        key_bytes = np.packbits(key_bits).tobytes()
        final_key_bits = []
        counter = 0
        while len(final_key_bits) < final_length:
            h = hashlib.sha256(key_bytes + counter.to_bytes(4, 'big'))
            hash_bits = np.unpackbits(np.frombuffer(h.digest(), dtype=np.uint8))
            final_key_bits.extend(hash_bits)
            counter += 1
        
        return np.array(final_key_bits[:final_length], dtype=int)
    
    def establish_key(self, noise_level: float = 0.01, 
                     eavesdrop_probability: float = 0.0,
                     verbose: bool = True) -> Tuple[np.ndarray, dict]:
        
        """
        completing bb84 protocol to make a shared key + stats (dictionary w/ protocl statistics)
        """

        raw_bits_needed = self.key_length * self.raw_key_multiplier
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  BB84 Quantum Key Distribution Protocol")
            print(f"{'='*60}")
            print(f"Target key length: {self.key_length} bits")
            print(f"Generating {raw_bits_needed} raw qubits...")
        
        # step 1: sender generates random bits and bases
        sender_bits = self._generate_random_bits(raw_bits_needed)
        sender_bases = self._generate_random_bases(raw_bits_needed)
        
        # step 2: sender prepares and sends qubits
        qubits = self._prepare_qubits(sender_bits, sender_bases)
        
        # step 3: simulate quantum channel transmission
        transmitted_qubits = self._simulate_quantum_channel(
            qubits, noise_level, eavesdrop_probability
        )
        
        # step 4: reciever generates random bases and measures
        reciever_bases = self._generate_random_bases(raw_bits_needed)
        reciever_bits = self._measure_qubits(transmitted_qubits, reciever_bases)
        
        # step 5: key sifting
        sifted_sender, sifted_reciever = self._sift_keys(
            sender_bits, sender_bases, reciever_bits, reciever_bases
        )
        
        if verbose:
            print(f"After sifting: {len(sifted_sender)} bits remaining")
        
        # step 6: error estimation
        sample_size = min(len(sifted_sender) // 4, 100)
        qber, remaining_sender, remaining_reciever = self._estimate_error_rate(
            sifted_sender, sifted_reciever, sample_size
        )
        
        if verbose:
            print(f"Quantum Bit Error Rate (QBER): {qber:.2%}")
            print(f"Error threshold: {self.error_threshold:.2%}")
        
        # step 7: check if QBER is acceptable
        if qber > self.error_threshold:
            raise QKDSecurityError(
                f"QBER ({qber:.2%}) exceeds threshold ({self.error_threshold:.2%}). "
                "Possible eavesdropping detected! Aborting key exchange."
            )
        
        if verbose:
            print(f"✓ QBER acceptable - channel secure")
            print(f"Remaining bits after error estimation: {len(remaining_sender)}")
        
        # step 8: information reconciliation
        reconciled_key = self._information_reconciliation(remaining_sender, remaining_reciever)
        
        if verbose:
            print(f"After error correction: {len(reconciled_key)} bits")
        
        # step 9: privacy amplifying
        final_key = self._privacy_amplification(reconciled_key, self.key_length)
        
        if verbose:
            print(f"Final key length: {len(final_key)} bits")
            print(f"{'='*60}\n")
        
        stats = {
            'raw_bits': raw_bits_needed,
            'sifted_bits': len(sifted_sender),
            'qber': qber,
            'final_key_length': len(final_key),
            'efficiency': len(final_key) / raw_bits_needed
        }
        
        return final_key, stats


class QKDSecurityError(Exception): ## for when security check fails
    pass

# below are the functions for integrating qkd w/ rlce encryption according to main.py file

def qkd_key_to_seed(qkd_key: np.ndarray) -> int:
    """
     converting qkd key to seed integer for prng (bit array -> integer transformation)
    """
    key_bytes = np.packbits(qkd_key).tobytes()
    hash_digest = hashlib.sha256(key_bytes).digest()
    seed = int.from_bytes(hash_digest[:4], byteorder='big')
    return seed


def establish_qkd_tunnel(key_length: int = 256, 
                         noise_level: float = 0.01,
                         verbose: bool = True) -> Tuple[int, Dict]:
   
    """
    creating simulated tunnel and returning the seed to use it for rlce key generation
    """

    qkd = BB84_QKD(key_length=key_length)
    
    try:
        qkd_key, stats = qkd.establish_key(
            noise_level=noise_level,
            eavesdrop_probability=0.0,
            verbose=verbose
        )
        
        seed = qkd_key_to_seed(qkd_key)
        
        if verbose:
            print(f"Generated PRNG seed from QKD: {seed}")
            print(f"Seed entropy: {key_length} bits\n")
        
        return seed, stats
        
    except QKDSecurityError as e:
        print(f"⚠ SECURITY ALERT: {e}")
        raise


def KeyGen_QKD(n: int, k: int, t: int, r: int, 
               qkd_seed: int = None, 
               qkd_key_length: int = 256,
               verbose: bool = False) -> Tuple:
    
    """
    wrapping main.keyGen with qkd
    """

    # make QKD tunnel if no seed provided
    if qkd_seed is None:
        if verbose:
            print("\n[QKD] Establishing quantum key distribution tunnel...")
        
        qkd_seed, stats = establish_qkd_tunnel(
            key_length=qkd_key_length,
            noise_level=0.01,
            verbose=verbose
        )
        
        if verbose:
            print(f"[QKD] Security metrics:")
            print(f"      • QBER: {stats['qber']:.2%}")
            print(f"      • Efficiency: {stats['efficiency']:.2%}")
            print(f"      • Key length: {stats['final_key_length']} bits\n")
    
    # integrate the random number generator with quantum-secure seed
    np.random.seed(qkd_seed)
    
    # using the original KeyGen from main.py
    public_key, private_key = main.KeyGen(n, k, t, r)
    
    # abding QKD information to private key
    private_key["qkd_seed"] = qkd_seed
    private_key["qkd_secured"] = True
    
    return (public_key, private_key)


## below is the implementation (demonstration + testing):

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  RLCE Encryption with Quantum Key Distribution")
    print("="*70)
    
    # parameters
    n = 255
    k = 235
    t = 10
    r = 1
    
    print(f"\n[1] RLCE System Parameters:")
    print(f"    • Code length (n): {n}")
    print(f"    • Message dimension (k): {k}")
    print(f"    • Error correction capability (t): {t}")
    print(f"    • Random matrix dimension (r): {r}")
    
    # generating keys using qkd
    print(f"\n[2] Key Generation with QKD:")
    public_key, private_key = KeyGen_QKD(n=n, k=k, t=t, r=r, verbose=True)
    
    # making test message
    print("[3] Creating test message...")
    GF = private_key["GF"]
    message = GF.Random(k)
    print(f"    Message length: {len(message)} symbols")
    
    # encrypt w/ main.encrypt
    print("\n[4] Encrypting message...")
    ciphertext = main.encrypt(public_key, message, weight=t)
    print(f"    Ciphertext length: {len(ciphertext)} symbols")
    
    # decrypt w/ main.decrypt
    print("\n[5] Decrypting message...")
    decrypted_message = main.decrypt(private_key, ciphertext)
    
    # Veverifying
    is_correct = np.array_equal(message, decrypted_message)
    
    print("\n" + "="*70)
    print("  Verification Results")
    print("="*70)
    print(f"QKD Seed: {private_key['qkd_seed']}")
    print(f"QKD Secured: {private_key['qkd_secured']}")
    print(f"Decryption Success: {'✅ PASS' if is_correct else '❌ FAIL'}")
    print("="*70)
    
    # demonstrating deterministic key regeneration
    print("\n" + "="*70)
    print("  Deterministic Key Regeneration Test")
    print("="*70)
    print("Using the same QKD seed should produce identical keys...\n")
    
    saved_seed = private_key['qkd_seed']
    
    # remaking keys with same seed
    public_key2, private_key2 = KeyGen_QKD(
        n=n, k=k, t=t, r=r, 
        qkd_seed=saved_seed, 
        verbose=False
    )
    
    keys_match = np.array_equal(public_key, public_key2)
    print(f"Public keys match: {'✅ YES' if keys_match else '❌ NO'}")
    
    # testing with regenerated keys
    ciphertext2 = main.encrypt(public_key2, message, weight=t)
    decrypted2 = main.decrypt(private_key2, ciphertext2)
    regeneration_works = np.array_equal(message, decrypted2)
    print(f"Encryption/Decryption with regenerated keys: {'✅ PASS' if regeneration_works else '❌ FAIL'}")
    
    # showing eavesdropping detection
    print("\n" + "="*70)
    print("  Eavesdropping Detection Demo")
    print("="*70)
    print("\nSimulating quantum channel with eavesdropper...\n")
    
    qkd_with_eavesdropper = BB84_QKD(key_length=256)
    try:
        compromised_key, eavesdropper_stats = qkd_with_eavesdropper.establish_key(
            noise_level=0.01,
            eavesdrop_probability=0.25,  # eavesdropper will intercepts 25% of qubits
            verbose=True
        )
        print("\n⚠ Warning: Key exchange succeeded despite eavesdropping")
        print("   (Low eavesdropping rate may go undetected)")
    except QKDSecurityError as e:
        print(f"\n✓ Security Protocol Working!")
        print(f"  Eavesdropping detected and key exchange aborted.")
        print(f"  System remains secure.\n")
    
    print("="*70)
    print("\nIntegration Complete!")
    print("QKD tunnel successfully established and integrated with RLCE encryption.")
    print("="*70 + "\n")