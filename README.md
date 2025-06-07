![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Quantum Resistant Random Linear Code Based Public Key Encryption Scheme (RLCE)

## 📌 Project Summary
This repository contains an implementation of the encryption method proposed in [this paper](https://cs.paperswithcode.com/paper/quantum-resistant-random-linear-code-based)

This paper proposes linear code based encryption scheme RLCE which shares many characteristics with random linear codes. 

## 🎯 Motivation
In recent years, lattice-based and linear code-based encryption methods have received growing attention because they are seen as strong candidates for **post-quantum cryptography**. The LLL reduction algorithm is a common tool used to break lattice-based systems. However, attacks on linear code-based systems often depend on the specific design of the scheme. Some powerful cryptanalysis techniques have been developed to target these types of encryption, and even though these methods are fairly new, they have already been used to break several systems. 
Thus it is important to design linear code-based encryption schemes that can resist these kinds of attacks.

## 🧩 Novelty
- **Based on GRS codes**: Uses generalized Reed-Solomon codes as the foundation for encryption.
- **Random insertion**: Inserts random vectors into the generator matrix to hide its structure.
- **Linear transformations**: Applies random invertible matrices to disguise the code further.
- **Permutation obfuscation**: Uses permutation matrices to make the public key appear random.
- **Resistant to known attacks**: Designed to withstand algebraic, filtration, and Sidelnikov-Shestakov attacks.
- **Indistinguishability**: Makes the public key statistically indistinguishable from a random matrix.

## 🧠 Methodology

The RLCE encryption scheme involves three main components:
1. Key Generation (linear)
    - A GRS generator matrix Gₛ is constructed.
    - Random matrices and vectors are inserted into Gₛ to form an extended matrix G₁.
    - The matrix G₁ is then transformed using:
      - a random invertible matrix S,
      - a block-diagonal random matrix A,
      - and a permutation matrix P₂.
      - The final public key is G = S ⋅ G₁ ⋅ A ⋅ P₂, and the private key includes S, Gₛ, permutations, and inverse matrices.
2. Encryption
    - A message vector m is multiplied with the public key G and added to a randomly generated sparse error vector e, forming ciphertext c = mG + e.
    - The error vector has a fixed weight t, and its non-zero positions are selected randomly.
3. Decryption
    - The receiver uses private matrices to reverse the permutation and transformation.
    - A decoding algorithm for GRS codes recovers the original message vector m, assuming the number of errors is within the decoding threshold.
  - The decrypted output is verified by checking if the reconstructed error vector matches the o  riginal error distribution.

The scheme also includes a padding function (RLCEpad) based on OAEP+ to ensure semantic security against adaptive chosen ciphertext attacks (IND-CCA2).

## 🌍 Impact
The RLCE scheme offers a promising post-quantum secure alternative to traditional encryption systems. As quantum computers become more capable, encryption methods like RSA and ECC may be broken. RLCE provides strong security foundations due to the NP-hardness of decoding random linear codes, a problem believed to be hard even for quantum computers. By using code structures that are both efficient to decode and hard to distinguish, RLCE contributes to the development of practical and secure post-quantum cryptography.
