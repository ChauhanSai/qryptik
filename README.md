![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# qryptik/maryam-maalin
## 📌 Branch Summary
Implements syndrome-based deep decoders for RLCE: neural decoders that take the syndrome (and optionally public ciphertext statistics) and predict sparse error patterns to assist classical McEliece-style decoding. These models raise decoding success probability and robustness (noisy/adversarial inputs) without altering public-key construction and add a learned key-validation stage to reduce false accepts.

## 🧠 Methodology
### Syndrome-Based CNN / RNN Models
Combines CNN-based feature extraction, recurrent refinement, iterative message passing, and a candidate key validation network.
**Inputs & Targets:** The model receives parity-check syndromes (and optional side features) and predicts sparse binary error vectors. Training data consists of simulated RLCE encryptions with varying error weights, augmented with noise and adversarial examples.

### Architecture:
  * **CNN encoder:** Captures local syndrome patterns and parity constraints; uses residual and dilated convolutions to model long-range dependencies.
	* **RNN refinement:** GRU/LSTM layers model sequential dependencies across code positions, improving predictions beyond the CNN alone.
	* **Iterative message passing:** Neural belief-propagation style updates refine error estimates over multiple iterations.
	* **Key-validation network:** Small MLP or CNN evaluates candidate keys/ciphertexts for acceptance, reducing false positives.

**Training:** Optimized with Adam, using losses combining per-bit BCE, syndrome consistency, and sparsity regularization. Class imbalance, dropout, and early stopping are applied.

**Inference:** The neural model predicts candidate error patterns, which are corrected using classical decoders. The validator network selects the most likely valid results, optionally with iterative refinement before falling back to classical decoding.

**Robustness & Security:** Adversarial training and input noise improve reliability, while final decisions rely on algebraic verification to maintain cryptographic security.

**Metrics:** Performance is tracked via bit/frame error rates, decoding success, validated-key false-accept rate, and inference latency.
