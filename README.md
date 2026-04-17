# Research Project Scope: Edge-AGI & Recursive Compression

## Project Title
**Evaluating Recursive Architectures and Compression Frontiers for General Reasoning on Resource-Constrained Embedded Systems**

---

## 1. What is the question you want to answer?
Can recursive neural architectures maintain high-level abstract reasoning capabilities (specifically on the ARC and Sudoku benchmarks) when subjected to extreme model compression for deployment on <1MB SRAM embedded devices? Furthermore, does recursive depth provide a more "energy-efficient" path to AGI-like reasoning than standard feed-forward scaling in TinyML environments?

---

## 2. Why is this question important?
Currently, "Reasoning" is viewed as an emergent property of massive scale (LLMs). If we can prove that complex reasoning can be preserved in a "Tiny" recursive model on a $10 microcontroller, we break the hardware barrier for autonomous AGI. This is critical for:

- **Privacy-First AI**: Local processing of sensitive logic without cloud tethering.  
- **Extreme Environments**: Space exploration or deep-sea robotics where bandwidth is zero.  
- **Sustainability**: Reducing the carbon footprint of "thinking" by orders of magnitude.

---

## 3. What work does this question build out? What papers should one read?
This builds directly on the Tiny Recursive Model (TRM) concept, which suggests that weight-sharing through recursion allows small models to solve complex logic tasks that usually require millions of parameters.

**Key Literature:**
- *Less is More: Recursive Reasoning with Tiny Network* (The core TRM model paper)  
- *On the Measure of Intelligence* (Francois Chollet, 2019) – Introduces the ARC benchmark  
- *Deep Compression* (Han et al.) – Pruning, quantization, Huffman coding  
- *MCUNet: Tiny Deep Learning on IoT Devices* (Lin et al., MIT)

---

## 4. Is this a publishable question?
Yes. While TRM exists, its behavior under extreme edge-compression (INT4 quantization, structured pruning) and its performance on the ARC benchmark at the hardware level is unexplored territory.  

**Proposed paper title:**  
*Quantized Recursion: Pushing the Limits of Abstract Reasoning on Cortex-M Hardware*  
Target venues: NeurIPS (TinyML), CVPR, SysML.

---

## 5. Simplest experimental setting
Compare a standard TRM (FP32) against a Quantized TRM (INT8/INT4) on a Sudoku solving task. Measure **Reasoning Decay** — the point at which quantization noise breaks the model’s ability to satisfy puzzle constraints.

---

## 6. Key baselines and benchmarks
**Baselines:**
- Standard feed-forward CNNs/MLPs of equivalent parameter count.

**Benchmarks:**
- ARC (Abstraction and Reasoning Corpus)  
- Sudoku (9x9)  
- List Challenge (sequential and memory-based reasoning)

---

## 7. Datasets, models, evaluation, resources
- **Datasets:** ARC-AGI (GitHub), Sudoku-1M, synthetic List Reasoning datasets  
- **Models:** Tiny Recursive Model (TRM), Cell Model, Tiny attention-based Transformer  
- **Evaluation Metrics:**  
  - Accuracy: Pass@1 solve rate  
  - Efficiency: Inference latency (ms), energy per inference (mJ)  
  - Reliability: Correlation between recursive depth and accuracy  
- **Resources:** TensorFlow Lite Micro, TVM, Arduino Portenta H7 / ESP32-S3, Power Profiler Kit (PPK2)

---

## 8. Additional details & initial experiments
**Specialization through Similarity Loss:**  
Introduce a cosine similarity penalty between recursive steps’ internal states (attention maps).  

**Hypothesis:**  
Forcing recursive steps to be dissimilar encourages specialization (e.g., Step 1 handles row constraints, Step 2 handles column constraints). This may allow pruning redundant cycles, improving speed.

**Initial Question:**  
Does reasoning fail gracefully (e.g., 80% correct) or catastrophically (completely forgetting rules) when moving from 8-bit to 4-bit weights?

---
