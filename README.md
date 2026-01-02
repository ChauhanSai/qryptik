![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Qryptik
**🏆 First place at UTD's ACM Research Symposium Fall 2025**

![ACM Research Banner Light](https://github.com/ChauhanSai/qryptik/blob/c061d1ae21e3ac74373a5d0216e854ebcb70bc42/POSTER.jpg)

## 📌 Project Summary
**Qryptik** is an applied cryptography research project that investigates how **deep learning–assisted decoding** can strengthen **post-quantum, code-based encryption systems**. Our work focuses on **Random Linear Code Encryption (RLCE)**, a McEliece-style cryptosystem designed to resist quantum attacks through heavy structural randomization.

Rather than modifying RLCE’s public construction, Qryptik integrates **neural network–based decoders and key validation mechanisms** that operate alongside the classical system. This hybrid approach improves error correction reliability, enhances robustness in noisy or adversarial environments, and explores how machine learning can extend cryptographic guarantees beyond traditional algebraic bounds.

By integrating neural network-based validation and decoding systems, Qryptik pushes the boundaries of cryptography for a quantum-enabled world.

## 🎯 Motivation
The rapid advancement of quantum computing poses a direct threat to widely used cryptographic standards such as **RSA** and **ECC**, motivating a global shift toward **quantum-safe encryption**. Governments and industry leaders have recognized this urgency—most notably through large-scale investment in post-quantum research and infrastructure migration toward quantum-resistant systems.

RLCE is a promising candidate in this space due to its extreme public-key randomization. However, like many code-based schemes, it faces two key challenges:

1. **Decoding under real-world noise:** Classical Reed–Solomon decoding guarantees correction only up to a fixed algebraic bound *t*. In practical or adversarial channels, errors often exceed this limit.
2. **Long-term cryptographic resilience:** Even if RLCE’s structure remains secure, future advances—quantum or otherwise—may weaken individual randomization layers.

Qryptik addresses these challenges by applying deep learning to decoding and validation, strengthening RLCE *without altering its public system*.

## 🧩 Novelty
Qryptik contributes a novel intersection of **post-quantum cryptography** and **machine learning**:

- **Neural decoding beyond algebraic bounds:** We implement belief-propagation-inspired recurrent models and graph neural networks that learn parity relationships directly from RLCE’s structure.
- **Syndrome-based learning:** Models are trained exclusively on syndromes, preserving cryptographic safety while enabling scalable learning.
- **Hybrid classical–neural decoding:** Neural decoders act as a fallback or enhancement when Reed–Solomon decoding fails, improving reliability under high noise.
- **Key validation via CNNs:** A neural filter detects weak or tampered keys during generation, ensuring only high-quality keys are published.

Rather than replacing cryptography with machine learning, Qryptik demonstrates how neural networks can *augment* established post-quantum systems in a safe and principled way. This combination of approaches creates a mobile-ready, quantum-resistant encryption framework that leverages machine learning in a way rarely applied in cryptography.

## 🧠 Methodology
### RLCE Foundation

All experiments are built on a Python implementation of RLCE using the **Galois** library. The system operates over **GF(2⁸)** with Reed–Solomon base codes. The public key is constructed as:

```
G_pub = S × G₁ × A × P
```

Where:

* **S** performs random row mixing
* **A** is a block-diagonal matrix that locally mixes real and random columns
* **P** globally permutes columns

This construction produces a highly randomized public key resistant to structural attacks.

### Dataset Generation

* Training data consists of **syndromes**, capturing the difference between received ciphertexts and valid codewords.
* New message–error pairs *(m, e)* are generated **every batch and epoch**, preventing memorization.
* Training follows a **curriculum strategy**, starting with single-error cases before scaling to multi-error regimes.

### Neural Decoding Approaches

1. **BP-RNN (Belief Propagation RNN):** Converts ciphertexts into parity-check graphs and applies tied-weight recurrent belief propagation to iteratively relax errors beyond the classical bound.

2. **R-GNN (Recurrent Graph Neural Network):** A custom graph-based model derived from the RLCE parity-check matrix. It identifies error locations one at a time, recursively reducing the error count until classical decoding succeeds.

3. **Syndrome-Based CNN / RNN Models:** Non-graph baselines that predict error masks directly from syndromes, enabling comparison across architectures.

All models preserve the original cryptosystem and operate strictly at the decoding layer.

## 🌍 Impact
Qryptik has far-reaching implications for the future of secure computing:
1. **More reliable post-quantum communication:** Enhanced decoding improves robustness in noisy or adversarial channels such as wireless, high-frequency financial, and military systems including Cortex-M microprocessors.
2. **Stronger cryptographic rigidity:** Pushing beyond algebraic bounds increases resistance to information-set decoding attacks.
3. **Safe integration of ML and cryptography:** Syndrome-based learning preserves security assumptions while enabling performance gains.
4. **Foundations for future research:** This work opens pathways for hybrid cryptosystems that combine formal security with adaptive intelligence.

## 👥 Team
Developers ⭐:
- Avery Brown
- Karthik Sobhirala
- Maryam Maalin
- Siri Appalaneni

Faculty Advisor 🧑‍🔬: Dr. Andrew Nemec

Project Manager 🤺: Sai Chauhan
