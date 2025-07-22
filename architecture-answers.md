# LLM Architecture Quiz (With Solutions)

---

## Section 1: Architectural Intuitions and Differences

### Q1.
Suppose you're designing a decoder-only transformer for causal language modeling. What are the *minimum* constraints you'd need to impose on the attention mechanism to prevent information leakage during training?

- A. Mask all tokens after the current one  
- B. Use absolute positional embeddings  
- C. Share key and query weights across layers  
- D. Ensure attention is only computed from left to right  

**Answer: D**

**Explanation:**  
To prevent future tokens from leaking into the prediction of the current token, you must ensure strictly left-to-right attention — causal attention. Absolute embeddings or shared weights do not enforce causality.

---

### Q2.
The Mamba architecture replaces attention with an SSM (state-space model) based operator. Which of the following would be a likely *limitation* of this design compared to Transformers?

- A. Lower inference-time memory footprint  
- B. Difficulty modeling long-range dependencies in practice  
- C. Inability to handle variable-length sequences  
- D. Lack of theoretical expressive power  

**Answer: B**

**Explanation:**  
Mamba improves efficiency, but struggles to learn long-range dependencies as effectively as attention, which has explicit content-based memory.

---

### Q3.
Which of the following architectural innovations is most directly responsible for allowing Mistral to outperform larger models like LLaMA-2 with fewer parameters?

- A. Rotary embeddings  
- B. MoE routing  
- C. Sliding Window Attention + Grouped Query Attention  
- D. Tied embeddings  

**Answer: C**

**Explanation:**  
Mistral uses a combination of Sliding Window Attention (SWA) and Grouped Query Attention (GQA) to reduce memory and compute while maintaining performance.

---

### Q4.
In the RWKV architecture, the model computes a weighted moving average over tokens. What behavior does this approximate in transformer attention?

- A. Value projection  
- B. Causal masking  
- C. Key-query dot product  
- D. Softmax weighting of values  

**Answer: D**

**Explanation:**  
RWKV approximates attention by computing a learned moving average, similar in function to how softmax weights value vectors based on keys and queries.

---

## Section 2: Mathematical Reasoning

### Q5.
Suppose a Transformer attention layer with dimensionality `d` has query, key, and value projections `W_Q`, `W_K`, `W_V` in `R^{d x d}`. Show that softmax attention is invariant to adding the same constant vector `c in R^d` to all key vectors.

**Answer:**  
Adding `c` to each key shifts all dot products by the same scalar (since Q * (K + c)^T = QK^T + Qc^T). Softmax is shift-invariant, so the output remains unchanged.

---

### Q6.
Consider an SSM with transition matrix `A in R^{d x d}`. Under what conditions does the output of the SSM converge to a stationary distribution, and how does this relate to model stability?

**Answer:**  
The system converges if all eigenvalues of `A` have magnitude less than 1 (spectral radius < 1). This ensures the hidden state decays over time and avoids instability during training or inference.

---

## Section 3: Practical Gotchas and Toy Setups

### Q7.
You're pretraining a 7B decoder-only transformer. Which of the following changes will *not* reduce VRAM usage significantly during training?

- A. Switching from full attention to sliding window attention  
- B. Using flash attention  
- C. Replacing attention with an SSM layer like in Mamba  
- D. Decreasing batch size by 2× but doubling sequence length  

**Answer: D**

**Explanation:**  
Attention cost is quadratic in sequence length. So doubling it increases memory usage even if batch size is halved.

---

### Q8.
A sequence contains the tokens "The cat chased the dog". Suppose your tokenizer splits it into: ["The", "cat", "chased", "the", "dog"]. Which attention pattern would allow a decoder-only model to predict "dog" given all previous context, while still allowing efficient streaming inference?

- A. Full attention  
- B. Bidirectional attention  
- C. Causal attention  
- D. Cross-attention  

**Answer: C**

**Explanation:**  
Causal attention ensures each token only sees earlier tokens, preserving autoregressive structure and enabling streaming.

---

### Q9.
You are training a model with rotary positional embeddings (RoPE). What happens if you naively increase the sequence length at inference time without fine-tuning?

- A. Model fails to attend to early tokens  
- B. Positional embeddings repeat periodically  
- C. Attention degrades for longer ranges due to frequency aliasing  
- D. Output remains unchanged due to position extrapolation  

**Answer: C**

**Explanation:**  
RoPE uses complex rotations; extrapolating beyond the training range can lead to phase distortions and aliasing, degrading performance.

---

### Q10.
You observe that a Mamba model trained on next-token prediction exhibits worse calibration compared to a Transformer of similar size. What is the most likely cause?

- A. Mamba lacks recurrence  
- B. SSMs have no normalization layer  
- C. Mamba’s implicit recurrence leads to low entropy outputs  
- D. Mamba uses fewer nonlinearities, making outputs less expressive  

**Answer: C**

**Explanation:**  
The strong temporal biases in Mamba’s recurrence can cause overconfident predictions, especially on uncertain next-token distributions.

---
