![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Qryptik

## 📌 Project Summary
**Qryptik** is a cutting-edge research project combining post-quantum encryption with deep learning to improve security, efficiency, and practical deployment. Our work focuses on **Random Linear Code Encryption (RLCE)**, enhancing its post-quantum security while making it lightweight enough for mobile and embedded devices. By integrating neural network-based validation and decoding systems, Qryptik pushes the boundaries of cryptography for a quantum-enabled world.

## 🎯 Motivation
The rise of quantum computers threatens classical encryption standards like **RSA** and **AES**, which could be broken in mere days. Traditional encryption methods are also too resource-intensive for small, everyday devices. Our motivation is twofold:
1. **Security against quantum attacks:** RLCE’s design — using random columns, mixing matrices (A, S), and global permutations (P) — produces highly randomized public keys. Our experiments confirm that brute-forcing these structures is computationally infeasible.
2. **Practicality and efficiency:** Current RLCE key sizes are too large for mobile deployment. By introducing deep learning layers, we aim to reduce key sizes, strengthen security, and improve decoding speed, enabling post-quantum encryption on everyday devices.

## 🧩 Novelty
Qryptik’s novelty lies in the fusion of cryptography and deep learning:
- Key validation using CNNs: Our model detects weak or tampered keys in real time, adding an extra security layer during key generation.
- Hybrid neural decoders: Using BP-RNN and syndrome-based CNN architectures, we can correct errors beyond classical algebraic bounds and improve decoding efficiency.
- Integration of diverse random linear codes: Beyond Reed–Solomon, other code families are explored to mitigate structural attacks and optimize decoding performance.

This combination of approaches creates a mobile-ready, quantum-resistant encryption framework that leverages machine learning in a way rarely applied in cryptography.

## 🧠 Methodology
1. Key Strengthening and Validation:
  - CNN-based filters evaluate candidate public keys for tampering or weaknesses.
  - Only strong, validated keys are published, improving system security.
2. Neural Network Decoding:
  a. BP-RNN: Converts RLCE ciphertexts into parity-check graphs and iteratively decodes beyond the classical error-correction limit.
  b. Syndrome-based CNN: Computes codeword syndromes, compares with original message syndromes, and applies neural correction if classical decoding fails.

## 🌍 Impact
Qryptik has far-reaching implications for the future of secure computing:
1. Mobile and embedded security: Reduced key sizes allow RLCE deployment on devices with limited memory, such as Cortex-M microprocessors.
2. Reliable communication under noise: Improved error-correcting capabilities support applications in high-noise environments, including high-frequency trading and military communications.
3. Post-quantum readiness: Neural network-assisted validation ensures the continued strength of public keys, safeguarding information against emerging quantum threats.
4. Cryptography + ML innovation: Qryptik demonstrates a novel application of machine learning in cryptography, opening avenues for future research in secure, efficient, and quantum-resistant encryption systems.

## 👥 Team
Developers ⭐:
- Avery Brown
- Karthik Sobhirala
- Maryam Maalin
- Siri Appalaneni

Project Manager 🤺: Sai Chauhan
