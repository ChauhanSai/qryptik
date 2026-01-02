![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# qryptik/avery-brown
## 📌 Branch Summary
Implements a recurrent graph neural network as a learned decoder and validator that runs alongside the classical RLCE (Random Linear Code Encryption) pipeline.

## 🧠 Methodology
### R-GNN (Recurrent Graph Neural Network)
We represent each codeword as a graph, where nodes correspond to code symbols and edges capture relationships such as symbol-symbol or symbol-parity connections. Node features include channel information (LLRs), positional embeddings, and optional key-context encodings.

The model is a recurrent graph neural network (R-GNN) that iteratively performs message passing:
* Edge messages are computed from node states and edge features via MLPs.
* Node states are updated using a GRU to carry information across iterations.
* Readout layers produce bitwise probabilities or symbol predictions, optionally supplemented by a global validator to flag suspicious ciphertexts.

Training uses a combination of losses: bitwise decoding loss, parity consistency loss, and validator classification loss, with regularization techniques (dropout, weight decay, KL penalties) and curriculum or adversarial training strategies.

For RLCE integration, the R-GNN acts as a drop-in decoder, estimating error vectors or corrected codewords and pre-filtering potentially invalid ciphertexts to reduce computational overhead.

**Implementation:** PyTorch with PyTorch Geometric or DGL, typical dimensions D=M=128, T=10–20 iterations, batch size 8–32, mixed precision recommended. Optimizers include Adam/AdamW with gradient clipping and learning rate scheduling.

**Evaluation:** Bit error rate, block error rate, decoding success under noise, and validator AUC/ROC. Models are trained and deployed to avoid leaking secret structure or weakening post-quantum security.
