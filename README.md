![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# qryptik/siri-appalaneni
## 📌 Branch Summary
Implements a neural key-validation subsystem that uses convolutional neural networks (CNNs) to decide whether a candidate private key (or a candidate decoded plaintext) is valid for a given RLCE ciphertext/public key pair. The model improves practical reliability of RLCE decryption by filtering false decodings and enabling ML-assisted post-processing without altering the public-key scheme. This branch provides a probabilistic validity score and (optionally) per-bit confidence maps that integrate with classical decoders and key-recovery pipelines

## 🧠 Methodology
### Key validation via CNNs
Inputs: Features derived from the RLCE public key, ciphertext, syndrome, and a candidate decoding. These are typically represented as bitstrings or small integer arrays.

### Data Generation
*   **Source:** A synthetic dataset is generated using RLCE parameter distributions matching target deployment.
*   **Process:** For each example, a secret key and plaintext are sampled, and a ciphertext is produced. Multiple candidate decodings are then generated:
    *   **Valid (Label 1):** The correct decode.
    *   **Invalid (Label 0):** Classical decoder failures, random decodings, and adversarially perturbed decodings.
*   **Augmentation:** Realistic noise (bit flips, burst errors) is injected, error weights are varied, and public-key scramblings are altered to improve model robustness.

### Architecture
*   **Input Representation:** Bitstring inputs are reshaped into 2D arrays or stacked as channels (e.g., syndrome, candidate, key fingerprint) to exploit spatial/local patterns with CNNs.
*   **Core Network:** A CNN processes the structured input.
    *   **Initial Block:** Convolutional layers with BatchNorm and GELU activation, followed by pooling.
    *   **Residual Blocks:** 3-5 blocks with increasing width and skip connections for stable deep training.
    *   **Feature Aggregation:** Global average pooling produces a final feature vector.
*   **Dual-Output Head:**
    1.  **Classification Head:** Fully connected layers output a single probability for decode validity.
    2.  **Localization Head (Optional):** A transposed convolution path outputs a per-bit confidence heatmap for error localization.

### Training & Losses
*   **Primary Loss:** Binary Cross-Entropy (BCE) for the validity classification.
*   **Auxiliary Losses (Multi-Task):** Can include per-bit BCE for localization and contrastive loss to better separate valid/invalid candidate embeddings.
*   **Total Loss:** A weighted sum: `Total = BCE_validity + α * BCE_localization + β * metric_loss`.
*   **Setup:** AdamW optimizer with cosine annealing, mixed precision training, and early stopping based on validation AUC.

### Evaluation
*   **Primary Metrics:** ROC AUC and Precision-Recall AUC for overall performance.
*   **Threshold Selection:** The operating point is chosen by balancing false-accept (security) and false-reject (availability) rates. Equal Error Rate (EER) is reported.
*   **Localization Metrics:** For the optional heatmap, per-bit precision/recall and IoU-like metrics are used.

**Integration with RLCE Pipeline:** At decryption time, the classical decoder produces candidate keys. The CNN validates each candidate: 
* If probability ≥ threshold **τ**, the key is accepted.
*   If rejected, the system triggers a recovery protocol or backoff.

If localization is active, the heatmap can guide an iterative decoder by flipping the most likely error bits.
