# LLM Architecture Quiz 

## Section 1: Architectural Intuitions and Differences

### Q1.
Suppose you're designing a decoder-only transformer for causal language modeling. What are the *minimum* constraints you'd need to impose on the attention mechanism to prevent information leakage during training?

- A. Mask all tokens after the current one  
- B. Use absolute positional embeddings  
- C. Share key and query weights across layers  
- D. Ensure attention is only computed from left to right  

---

### Q2.
The Mamba architecture replaces attention with an SSM (state-space model) based operator. Which of the following would be a likely *limitation* of this design compared to Transformers?

- A. Lower inference-time memory footprint  
- B. Difficulty modeling long-range dependencies in practice  
- C. Inability to handle variable-length sequences  
- D. Lack of theoretical expressive power  

---

### Q3.
Which of the following architectural innovations is most directly responsible for allowing Mistral to outperform larger models like LLaMA-2 with fewer parameters?

- A. Rotary embeddings  
- B. MoE routing  
- C. Sliding Window Attention + Grouped Query Attention  
- D. Tied embeddings  

---

### Q4.
In the RWKV architecture, the model computes a weighted moving average over tokens. What behavior does this approximate in transformer attention?

- A. Value projection  
- B. Causal masking  
- C. Key-query dot product  
- D. Softmax weighting of values  

---

## Section 2: Mathematical Reasoning

### Q5.
Suppose a Transformer attention layer with dimensionality `d` has query, key, and value projections `W_Q`, `W_K`, `W_V` in `R^{d x d}`. Show that softmax attention is invariant to adding the same constant vector `c in R^d` to all key vectors.


---

### Q6.
Consider an SSM with transition matrix `A in R^{d x d}`. Under what conditions does the output of the SSM converge to a stationary distribution, and how does this relate to model stability?

---

## Section 3: Practical Implementation

### Q7.
You're pretraining a 7B decoder-only transformer. Which of the following changes will *not* reduce VRAM usage significantly during training?

- A. Switching from full attention to sliding window attention  
- B. Using flash attention  
- C. Replacing attention with an SSM layer like in Mamba  
- D. Decreasing batch size by 2× but doubling sequence length  

---

### Q8.
Consider the following toy example:

A sequence contains the tokens "The cat chased the dog". Suppose your tokenizer splits it into: ["The", "cat", "chased", "the", "dog"]. Which attention pattern would allow a decoder-only model to predict "dog" given all previous context, while still allowing efficient streaming inference?

- A. Full attention  
- B. Bidirectional attention  
- C. Causal attention  
- D. Cross-attention  

---

### Q9.
You are training a model with rotary positional embeddings (RoPE). What happens if you naively increase the sequence length at inference time without fine-tuning?

- A. Model fails to attend to early tokens  
- B. Positional embeddings repeat periodically  
- C. Attention degrades for longer ranges due to frequency aliasing  
- D. Output remains unchanged due to position extrapolation  

---

### Q10.
You observe that a Mamba model trained on next-token prediction exhibits worse calibration compared to a Transformer of similar size. What is the most likely cause?

- A. Mamba lacks recurrence  
- B. SSMs have no normalization layer  
- C. Mamba’s implicit recurrence leads to low entropy outputs  
- D. Mamba uses fewer nonlinearities, making outputs less expressive  
