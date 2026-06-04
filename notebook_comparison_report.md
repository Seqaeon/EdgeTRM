# EdgeTRM Model Evaluation & Compression Report

This report consolidates the performance scorecards, parameters, footprints, and visual plots extracted from the three core model variants:
1. **ARC-Prize 2024 (Baseline)**
2. **Maze (Hard Version)**
3. **Sudoku (Extreme Version)**

---

## ARC 2024 Evaluation Outputs

### MUST RUN FIRST — fix duplicate trm.py on Modal volume ────────────────────

### Cell 9: Sparsity Audit ────────────────────────────────────────────────────

### 3.1  High-Performance Evaluation & Adaptation Helpers ─────────────────────

### 4.0  Global imports & helpers ────────────────────────────────────────────

### 4.1  ARC DataLoader from pre-built .npy files ────────────────────────────

### 4.3  Baseline: evaluate the loaded FP32 model ────────────────────────────

### 4.4  Quantization helpers ─────────────────────────────────────────────────

### 4.7  Instantiate all model variants ──────────────────────────────────────

### 5.1  Evaluate all quantization levels ─────────────────────────────────────

### 5.2  Visualise reasoning decay ────────────────────────────────────────────

![Plot from cell 23](extracted_images/plot_arc_2024_cell_23_0.png)

### 6.1  Recursive depth sweep at inference time ──────────────────────────────

### 6.2  Plot depth × quantization grid ─────────────────────────────────────

![Plot from cell 26](extracted_images/plot_arc_2024_cell_26_1.png)

### 7.1  Carry-state similarity hooks ────────────────────────────────────────

### 7.2  Compare carry similarity across quantization variants ────────────────

![Plot from cell 29](extracted_images/plot_arc_2024_cell_29_2.png)

### 8.1  Comprehensive model footprint analysis ───────────────────────────────

### 8.2  SRAM footprint visualisation ────────────────────────────────────────

![Plot from cell 33](extracted_images/plot_arc_2024_cell_33_3.png)

### 8.3  Accuracy vs SRAM: the key publishable plot ──────────────────────────

![Plot from cell 34](extracted_images/plot_arc_2024_cell_34_4.png)

### 9.1  Inspect puzzle_emb shape & byte count ───────────────────────────────

### 9.2  Option A — INT8 quantize the embedding buffer ───────────────────────

### 9.3  Option B — SVD Low-Rank Puzzle Embedding ────────────────────────────

### 9.4  Option D — Single-Puzzle Inference (only load 1 row) ────────────────

### 9.5  Footprint summary across embedding strategies ───────────────────────

### 10.2  High-performance direct on-the-fly per-puzzle TTA evaluator ─────────

### #  10.3  Run the High-Performance Direct Source Evaluator ─────────────────────

### 10.2  Re-evaluate all variants with per-puzzle aggregation ────────────────

### 11.1  Per-channel calibration statistics ──────────────────────────────────

### 11.2  Apply calibrated INT4 and evaluate ──────────────────────────────────

### 12.1  QAT fine-tuning loop ────────────────────────────────────────────────

### 13.1  Magnitude-based row pruning ────────────────────────────────────────

### 14.1  TorchScript trace of single TRM backbone step ──────────────────────

### 14.2  Latency benchmark: traced model vs eager ────────────────────────────

### 15.1  Build train / val loaders ──────────────────────────────────────────

### 15.2  QAT loop with val-set evaluation every 10 steps ────────────────────

### 15.3  Plot train loss + val accuracy curve ────────────────────────────────

![Plot from cell 62](extracted_images/plot_arc_2024_cell_62_5.png)

### 16.1  Build fused INT8 + SinglePuzzle model ───────────────────────────────

### 16.2  Accuracy of fused model (val set, per-puzzle aggregation) ──────────

### 16.3  Save fused model (state dict + embedding table) ────────────────────

### 17.1  CPU-only latency benchmark ─────────────────────────────────────────

### 17.2  Peak SRAM usage during inference (tracemalloc) ─────────────────────

### 17.3  FLOP estimate via torch.profiler ────────────────────────────────────

```text
Estimated FLOPs per H-cycle step : 187504.0 MFLOPs
Full inference (16 steps)        : 3000.06 GFLOPs

Power estimate (Cortex-M55 @ ~1 GFLOPS/s, ~1 mW/GFLOP):
  FLOPs                  : 3000.064 GFLOPs/puzzle
  Est. energy            : ~3000.06 mJ/puzzle
  Puzzles per mAh (3.3V) : ~0
```

### 17.4  Deployment summary table ───────────────────────────────────────────

```text
=== Final Deployment Scorecard ===
                       Backbone MB Total deploy MB Puzzle Exact Cell Acc Fits 4MB Fits 8MB
Config                                                                                    
FP32 (baseline)              26.05           86.46       0.0000   0.0000        ✗        ✗
INT8 (bnb)                    6.51           66.92       0.0246   0.2003        ✗        ✗
INT4 (calibrated)             3.26            3.26       0.0148   0.1902        ✓        ✓
INT8 + Single-Puzzle ★        6.51            6.51       0.0000   0.0000        ✗        ✓

★ = recommended target for 8MB SRAM devices
```

### Utility: fix H_init / L_init device mismatch (permanent patch) ───────────

### 18.1  Evaluate all variants at n_sup_max = 1, 2, 4, 8, 16 (H_cycles = default vs 1) ──

```text
3 (Def)    Exact          0.2850     0.3600     0.3575     0.3600     0.3625
           Cell           0.8762     0.8838     0.8765     0.8763     0.8755
           Latency         7.5 ms     14.4 ms     28.1 ms     55.7 ms    111.0 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
```

```text
1          Exact          0.0025     0.0750     0.3450     0.3550     0.3575
           Cell           0.5410     0.7953     0.8825     0.8831     0.8773
           Latency         3.0 ms      5.3 ms      9.9 ms     19.2 ms     37.6 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: INT8 (bnb)                (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
```

```text
3 (Def)    Exact          0.2850     0.3575     0.3575     0.3600     0.3600
           Cell           0.8760     0.8837     0.8827     0.8842     0.8670
           Latency         5.8 ms     11.0 ms     21.3 ms     41.9 ms     83.2 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
```

```text
1          Exact          0.0025     0.0750     0.3425     0.3525     0.3625
           Cell           0.5415     0.7953     0.8827     0.8835     0.8767
           Latency         2.3 ms      4.1 ms      7.6 ms     14.6 ms     28.6 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: INT4 (fake)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
```

```text
3 (Def)    Exact          0.1575     0.2700     0.2675     0.2625     0.2575
           Cell           0.8372     0.8626     0.8473     0.8430     0.8322
           Latency         2.6 ms      4.4 ms      8.1 ms     15.5 ms     30.3 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
```

```text
1          Exact          0.0000     0.0225     0.2525     0.2650     0.2600
           Cell           0.3954     0.7148     0.8582     0.8494     0.8415
           Latency         1.3 ms      1.9 ms      3.2 ms      5.7 ms     10.7 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: FP32 (bf16)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
```

```text
3 (Def)    Exact          0.2850     0.3600     0.3575     0.3575     0.3600
           Cell           0.8762     0.8839     0.8764     0.8760     0.8757
           Latency         7.5 ms     14.3 ms     28.2 ms     55.9 ms    111.2 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
```

```text
1          Exact          0.0025     0.0750     0.3425     0.3575     0.3550
           Cell           0.5410     0.7954     0.8825     0.8832     0.8762
           Latency         3.0 ms      5.2 ms      9.9 ms     19.2 ms     37.6 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================
```

### 19.1  Zero-pad truncation at various context lengths ────────────────────

```text
900        100%   ~187 GFLOPs          0.3600      0.8755
```

```text
450         50%    ~47 GFLOPs          0.2650      0.2500
```

```text
225         25%    ~12 GFLOPs          0.1075      0.0492
```

```text
128         14%     ~4 GFLOPs          0.0825      0.0595
```

### #  20.1  QAT from calibrated INT4 (lr=1e-6, 300 steps) ────────────────────

### 21.1  Instantiate student TRM and train with KL distillation ─────────────

### 22.1  Monkey-patch softmax attention → linear attention ──────────────────

### 22.2  Retrain the linear attention model (distillation) ─────────────────

---

## Maze Hard Evaluation Outputs

### MUST RUN FIRST — fix duplicate trm.py on Modal volume ────────────────────

### Cell 9: Sparsity Audit ────────────────────────────────────────────────────

### 3.1  High-Performance Evaluation & Adaptation Helpers ─────────────────────

### 4.0  Global imports & helpers ────────────────────────────────────────────

### 4.1  Maze-Hard DataLoader from pre-built .npy files ────────────────────────

### 4.3  Baseline: evaluate the loaded FP32 model ────────────────────────────

### 4.4  Quantization helpers ─────────────────────────────────────────────────

### 4.7  Instantiate all model variants ──────────────────────────────────────

### 5.1  Evaluate all quantization levels ─────────────────────────────────────

### 5.2  Visualise reasoning decay ────────────────────────────────────────────

![Plot from cell 31](extracted_images/plot_maze_hard_cell_31_0.png)

### 6.1  Recursive depth sweep at inference time ──────────────────────────────

```text
Quant          H   n_sup    Pass@1    Pass@2  Cell Acc    Latency    GFLOPs
---------------------------------------------------------------------------
FP16           1   1        0.0000    0.0000    0.4362       2.68      62.5
FP16           1   2        0.1380    0.1380    0.9808       4.86     125.0
FP16           1   4        0.8440    0.8440    0.9949       9.25     250.0
FP16           1   6        0.8670    0.8670    0.9953      13.64     375.0
FP16           1   8        0.8700    0.8700    0.9953      18.04     500.0
FP16           1   10       0.8700    0.8700    0.9953      22.42     625.0
FP16           2   1        0.1380    0.1380    0.9808       4.86     125.0
FP16           2   2        0.8440    0.8440    0.9949       9.24     250.0
FP16           2   4        0.8700    0.8700    0.9953      18.00     500.0
FP16           2   6        0.8700    0.8700    0.9953      26.79     750.0
FP16           2   8        0.8700    0.8700    0.9953      35.51    1000.0
FP16           2   10       0.8700    0.8700    0.9953      44.21    1250.0
FP16           3   1        0.7850    0.7850    0.9940       7.05     187.5
FP16           3   2        0.8670    0.8670    0.9953      13.61     375.0
FP16           3   4        0.8700    0.8700    0.9953      26.72     750.0
FP16           3   6        0.8700    0.8700    0.9953      39.85    1125.0
FP16           3   8        0.8700    0.8700    0.9953      52.92    1500.0
FP16           3   10       0.8700    0.8700    0.9953      66.13    1875.0
FP16           4   1        0.8440    0.8440    0.9949       9.24     250.0
FP16           4   2        0.8700    0.8700    0.9953      18.00     500.0
FP16           4   4        0.8700    0.8700    0.9953      35.52    1000.0
FP16           4   6        0.8700    0.8700    0.9953      53.03    1500.0
FP16           4   8        0.8700    0.8700    0.9953      70.54    2000.0
FP16           4   10       0.8700    0.8700    0.9953      87.94    2500.1
INT8 (bnb)     1   1        0.0000    0.0000    0.4333       1.86      62.5
INT8 (bnb)     1   2        0.1380    0.1380    0.9808       3.38     125.0
INT8 (bnb)     1   4        0.8450    0.8450    0.9949       6.35     250.0
INT8 (bnb)     1   6        0.8680    0.8680    0.9953       9.21     375.0
INT8 (bnb)     1   8        0.8690    0.8690    0.9953      12.05     500.0
INT8 (bnb)     1   10       0.8700    0.8700    0.9953      14.90     625.0
INT8 (bnb)     2   1        0.1380    0.1380    0.9808       3.37     125.0
INT8 (bnb)     2   2        0.8450    0.8450    0.9949       6.33     250.0
INT8 (bnb)     2   4        0.8690    0.8690    0.9953      12.02     500.0
INT8 (bnb)     2   6        0.8710    0.8710    0.9953      17.70     750.0
INT8 (bnb)     2   8        0.8700    0.8700    0.9953      23.38    1000.0
INT8 (bnb)     2   10       0.8700    0.8700    0.9953      29.06    1250.0
INT8 (bnb)     3   1        0.7830    0.7830    0.9940       4.88     187.5
INT8 (bnb)     3   2        0.8680    0.8680    0.9953       9.17     375.0
INT8 (bnb)     3   4        0.8710    0.8710    0.9953      17.68     750.0
INT8 (bnb)     3   6        0.8710    0.8710    0.9953      26.20    1125.0
INT8 (bnb)     3   8        0.8700    0.8700    0.9953      34.71    1500.0
INT8 (bnb)     3   10       0.8700    0.8700    0.9953      43.22    1875.0
INT8 (bnb)     4   1        0.8450    0.8450    0.9949       6.32     250.0
INT8 (bnb)     4   2        0.8690    0.8690    0.9953      12.01     500.0
INT8 (bnb)     4   4        0.8700    0.8700    0.9953      23.35    1000.0
INT8 (bnb)     4   6        0.8700    0.8700    0.9953      34.69    1500.0
INT8 (bnb)     4   8        0.8700    0.8700    0.9953      46.03    2000.0
INT8 (bnb)     4   10       0.8700    0.8700    0.9953      57.38    2500.1
INT4 (fake)    1   1        0.0000    0.0000    0.4734       1.08      62.5
INT4 (fake)    1   2        0.1580    0.1580    0.9805       1.69     125.0
INT4 (fake)    1   4        0.8360    0.8360    0.9945       2.89     250.0
INT4 (fake)    1   6        0.8610    0.8610    0.9949       4.09     375.0
INT4 (fake)    1   8        0.8630    0.8630    0.9950       5.29     500.0
INT4 (fake)    1   10       0.8640    0.8640    0.9950       6.49     625.0
INT4 (fake)    2   1        0.1580    0.1580    0.9805       1.67     125.0
INT4 (fake)    2   2        0.8360    0.8360    0.9945       2.87     250.0
INT4 (fake)    2   4        0.8630    0.8630    0.9950       5.26     500.0
INT4 (fake)    2   6        0.8640    0.8640    0.9950       7.64     750.0
INT4 (fake)    2   8        0.8630    0.8630    0.9950      10.03    1000.0
INT4 (fake)    2   10       0.8640    0.8640    0.9950      12.42    1250.0
INT4 (fake)    3   1        0.7660    0.7660    0.9934       2.27     187.5
INT4 (fake)    3   2        0.8610    0.8610    0.9949       4.05     375.0
INT4 (fake)    3   4        0.8640    0.8640    0.9950       7.62     750.0
INT4 (fake)    3   6        0.8630    0.8630    0.9950      11.20    1125.0
INT4 (fake)    3   8        0.8640    0.8640    0.9950      14.77    1500.0
INT4 (fake)    3   10       0.8640    0.8640    0.9950      18.34    1875.0
INT4 (fake)    4   1        0.8360    0.8360    0.9945       2.86     250.0
INT4 (fake)    4   2        0.8630    0.8630    0.9950       5.24     500.0
INT4 (fake)    4   4        0.8630    0.8630    0.9950       9.99    1000.0
INT4 (fake)    4   6        0.8640    0.8640    0.9950      14.75    1500.0
INT4 (fake)    4   8        0.8650    0.8650    0.9950      19.51    2000.0
INT4 (fake)    4   10       0.8650    0.8650    0.9950      24.27    2500.1
FP32 (bf16)    1   1        0.0000    0.0000    0.4361       1.08      62.5
FP32 (bf16)    1   2        0.1390    0.1390    0.9808       1.68     125.0
FP32 (bf16)    1   4        0.8450    0.8450    0.9949       2.88     250.0
FP32 (bf16)    1   6        0.8660    0.8660    0.9952       4.08     375.0
FP32 (bf16)    1   8        0.8680    0.8680    0.9953       5.28     500.0
FP32 (bf16)    1   10       0.8680    0.8680    0.9952       6.48     625.0
FP32 (bf16)    2   1        0.1390    0.1390    0.9808       1.67     125.0
FP32 (bf16)    2   2        0.8450    0.8450    0.9949       2.87     250.0
FP32 (bf16)    2   4        0.8680    0.8680    0.9953       5.25     500.0
FP32 (bf16)    2   6        0.8680    0.8680    0.9952       7.64     750.0
FP32 (bf16)    2   8        0.8680    0.8680    0.9952      10.02    1000.0
FP32 (bf16)    2   10       0.8680    0.8680    0.9952      12.41    1250.0
FP32 (bf16)    3   1        0.7840    0.7840    0.9940       2.27     187.5
FP32 (bf16)    3   2        0.8660    0.8660    0.9952       4.05     375.0
FP32 (bf16)    3   4        0.8680    0.8680    0.9952       7.62     750.0
FP32 (bf16)    3   6        0.8680    0.8680    0.9952      11.19    1125.0
FP32 (bf16)    3   8        0.8680    0.8680    0.9952      14.76    1500.0
FP32 (bf16)    3   10       0.8680    0.8680    0.9952      18.33    1875.0
FP32 (bf16)    4   1        0.8450    0.8450    0.9949       2.86     250.0
FP32 (bf16)    4   2        0.8680    0.8680    0.9953       5.24     500.0
FP32 (bf16)    4   4        0.8680    0.8680    0.9952       9.99    1000.0
FP32 (bf16)    4   6        0.8680    0.8680    0.9952      14.74    1500.0
FP32 (bf16)    4   8        0.8680    0.8680    0.9952      19.50    2000.0
FP32 (bf16)    4   10       0.8680    0.8680    0.9952      24.25    2500.1
```

### 6.2  Plot depth × quantization grid ─────────────────────────────────────

![Plot from cell 34](extracted_images/plot_maze_hard_cell_34_1.png)

![Plot from cell 34](extracted_images/plot_maze_hard_cell_34_2.png)

### 7.1  Carry-state similarity hooks ────────────────────────────────────────

### 7.2  Compare carry similarity across quantization variants ────────────────

![Plot from cell 37](extracted_images/plot_maze_hard_cell_37_3.png)

### 8.1  Comprehensive model footprint analysis ───────────────────────────────

### 8.2  SRAM footprint visualisation ────────────────────────────────────────

![Plot from cell 41](extracted_images/plot_maze_hard_cell_41_4.png)

### 8.3  Accuracy vs SRAM: the key publishable plot ──────────────────────────

![Plot from cell 42](extracted_images/plot_maze_hard_cell_42_5.png)

### 9.1  Inspect puzzle_emb shape & byte count ───────────────────────────────

### 9.2  Option A — INT8 quantize the embedding buffer ───────────────────────

### 9.3  Option B — SVD Low-Rank Puzzle Embedding ────────────────────────────

### 9.4  Option D — Single-Puzzle Inference (only load 1 row) ────────────────

### 9.5  Footprint summary across embedding strategies ───────────────────────

### 10.2  High-performance direct on-the-fly per-puzzle TTA evaluator ─────────

### #  10.3  Run the High-Performance Direct Source Evaluator ─────────────────────

### 10.2  Re-evaluate all variants with per-puzzle aggregation ────────────────

### 11.1  Per-channel calibration statistics ──────────────────────────────────

### #  11.2  Apply calibrated INT4 and evaluate ──────────────────────────────────

### #  12.1  QAT fine-tuning loop ────────────────────────────────────────────────

### 13.1  Magnitude-based row pruning ────────────────────────────────────────

### 14.1  TorchScript trace of single TRM backbone step ──────────────────────

### 14.2  Latency benchmark: traced model vs eager ────────────────────────────

### 15.1  Build train / val loaders ──────────────────────────────────────────

### #  15.2  QAT loop with val-set evaluation every 10 steps ────────────────────

### 15.3  Plot train loss + val accuracy curve ────────────────────────────────

### 16.1  Build fused INT8 + SinglePuzzle model ───────────────────────────────

### 16.2  Accuracy of fused model (val set, per-puzzle aggregation) ──────────

### 16.3  Save fused model (state dict + embedding table) ────────────────────

### 17.1  CPU-only latency benchmark ─────────────────────────────────────────

### 17.2  Peak SRAM usage during inference (tracemalloc) ─────────────────────

### 17.3  FLOP estimate via torch.profiler ────────────────────────────────────

```text
Estimated FLOPs per H-cycle step : 187498.4 MFLOPs
Full inference (10 steps)        : 1874.98 GFLOPs

Power estimate (Cortex-M55 @ ~1 GFLOPS/s, ~1 mW/GFLOP):
  FLOPs                  : 1874.984 GFLOPs/puzzle
  Est. energy            : ~1874.98 mJ/puzzle
  Puzzles per mAh (3.3V) : ~0
```

### 17.4  Deployment summary table ───────────────────────────────────────────

```text
=== Final Deployment Scorecard ===
                       Backbone MB Total deploy MB Puzzle Exact Cell Acc Fits 4MB Fits 8MB
Config                                                                                    
FP32 (baseline)              26.03           26.03       0.8680   0.9952        ✗        ✗
INT8 (bnb)                    6.51            6.51       0.8700   0.9953        ✗        ✓
INT4 (calibrated)             3.25            3.26       0.8640   0.9950        ✓        ✓
INT8 + Single-Puzzle ★        6.51            6.51       1.0000   1.0000        ✗        ✓

★ = recommended target for 8MB SRAM devices
```

### #  Utility: fix H_init / L_init device mismatch (permanent patch) ───────────

### 18.1  Evaluate all variants at n_sup_max = 1, 2, 4, 8, 16 (H_cycles = default vs 1) ──

```text
======================================================================================
VARIANT: FP16                      (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.7850     0.8670     0.8700     0.8700     0.8700
           Cell           0.9940     0.9953     0.9953     0.9953     0.9953
           Latency         7.0 ms     13.6 ms     26.7 ms     53.0 ms    105.6 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
1          Exact          0.0000     0.1380     0.8440     0.8700     0.8700
           Cell           0.4362     0.9808     0.9949     0.9953     0.9953
           Latency         2.7 ms      4.9 ms      9.3 ms     18.0 ms     35.6 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: INT8 (bnb)                (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.7830     0.8680     0.8710     0.8700     0.8700
           Cell           0.9940     0.9953     0.9953     0.9953     0.9953
           Latency         4.9 ms      9.2 ms     17.7 ms     34.7 ms     68.8 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
1          Exact          0.0000     0.1380     0.8450     0.8690     0.8700
           Cell           0.4333     0.9808     0.9949     0.9953     0.9953
           Latency         1.9 ms      3.4 ms      6.3 ms     12.1 ms     23.5 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: INT4 (fake)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.7660     0.8610     0.8640     0.8640     0.8640
           Cell           0.9934     0.9949     0.9950     0.9950     0.9950
           Latency         2.3 ms      4.1 ms      7.6 ms     14.8 ms     29.1 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
1          Exact          0.0000     0.1580     0.8360     0.8630     0.8630
           Cell           0.4734     0.9805     0.9945     0.9950     0.9950
           Latency         1.1 ms      1.7 ms      2.9 ms      5.3 ms     10.1 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================

======================================================================================
VARIANT: FP32 (bf16)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.7850     0.8670     0.8700     0.8700     0.8700
           Cell           0.9940     0.9953     0.9953     0.9953     0.9953
           Latency         7.1 ms     13.7 ms     26.9 ms     53.3 ms    105.9 ms
           GFLOPs          187.5      375.0      750.0     1500.0     3000.1
--------------------------------------------------------------------------------------
1          Exact          0.0000     0.1390     0.8440     0.8690     0.8700
           Cell           0.4363     0.9808     0.9949     0.9953     0.9953
           Latency         2.7 ms      4.9 ms      9.3 ms     18.1 ms     35.8 ms
           GFLOPs           62.5      125.0      250.0      500.0     1000.0
======================================================================================
```

### 19.1  Zero-pad truncation at various context lengths ────────────────────

```text
Full seq_len = 900  (FLOPs ∝ seq_len²)
   Context  % of full   Attn FLOPs   Puzzle Exact   Cell Acc
--------------------------------------------------------------
       900        100%   ~187 GFLOPs          0.7850      0.9940
       450         50%    ~47 GFLOPs          0.0000      0.6576
       225         25%    ~12 GFLOPs          0.0000      0.5038
       128         14%     ~4 GFLOPs          0.0000      0.4436
```

### #  20.1  QAT from calibrated INT4 (lr=1e-6, 300 steps) ────────────────────

### 21.1  Instantiate student TRM and train with KL distillation ─────────────

```text
Teacher params : 6,822,914
Student params : 855,554  (0.13× teacher)
Resuming training from checkpoint: checkpoints/student_distill_latest.pt
Resumed at step 600
  Step  640/1000 | kl=2976.6737 | hard=0.3435
  Step  680/1000 | kl=2909.6919 | hard=0.3440
  Step  720/1000 | kl=2807.1469 | hard=0.3739
  Step  760/1000 | kl=2626.5060 | hard=0.3488
  Step  800/1000 | kl=2619.8455 | hard=0.3641
  [Checkpoint] Saved checkpoint at step 800 to checkpoints/student_distill_latest.pt
  Step  840/1000 | kl=2377.3866 | hard=0.3061
  Step  880/1000 | kl=2502.1829 | hard=0.4040
  Step  920/1000 | kl=2261.5046 | hard=0.2868
  Step  960/1000 | kl=2238.7907 | hard=0.3175
  Step 1000/1000 | kl=2064.3248 | hard=0.2746
  [Checkpoint] Saved checkpoint at step 1000 to checkpoints/student_distill_latest.pt
Evaluating student on test set...
  Student (hidden=256, 1L, 1-cycle): exact=0.0000  cell=0.8725  ~47 GFLOPs
```

### 22.1  Monkey-patch softmax attention → linear attention ──────────────────

```text
Patched 2 Attention modules → linear O(n) attention
Theoretical compute reduction: ~14× in attention layers

Evaluating linear attention model on test set (n_sup_max=1)...

Evaluating softmax attention (FP32 baseline, n_sup_max=1) for comparison...

Model                           Puzzle Exact   Cell Acc
--------------------------------------------------------
  FP32 softmax attn                   0.7850     0.9940
  Linear attn (ELU)                   0.0000     0.8728

Accuracy cost of linear swap: +12.12pp cell accuracy
Note: linear attn needs retraining to recover accuracy — this is a zero-shot test.
```

### 22.2  Retrain linear attention model to recover accuracy ─────────────────────

---

## Sudoku Extreme Evaluation Outputs

### MUST RUN FIRST — fix duplicate trm.py on Modal volume ────────────────────

### Cell 9: Sparsity Audit ────────────────────────────────────────────────────

### 3.1  High-Performance Evaluation & Adaptation Helpers ─────────────────────

### 4.0  Global imports & helpers ────────────────────────────────────────────

### 4.1  Sudoku Extreme DataLoader from pre-built .npy files ────────────────────────

### 4.3  Baseline: evaluate the loaded FP32 model ────────────────────────────

### 4.4  Quantization helpers ─────────────────────────────────────────────────

### 4.7  Instantiate all model variants ──────────────────────────────────────

### 5.1  Evaluate all quantization levels ─────────────────────────────────────

### 5.2  Visualise reasoning decay ────────────────────────────────────────────

![Plot from cell 31](extracted_images/plot_sudoku_extreme_cell_31_0.png)

### 6.1  Recursive depth sweep at inference time ──────────────────────────────

```text
Quant          H   n_sup    Pass@1    Pass@2  Cell Acc    Latency    GFLOPs
---------------------------------------------------------------------------
FP16           1   1        0.0060    0.0060    0.5526       0.31       8.5
FP16           1   2        0.0210    0.0210    0.6941       0.57      17.1
FP16           1   4        0.3830    0.3830    0.7856       1.08      34.2
FP16           1   6        0.4960    0.4960    0.8151       1.59      51.3
FP16           1   8        0.5480    0.5480    0.8303       2.10      68.3
FP16           1   10       0.5810    0.5810    0.8389       2.62      85.4
FP16           2   1        0.0210    0.0210    0.6941       0.56      17.1
FP16           2   2        0.3830    0.3830    0.7856       1.08      34.2
FP16           2   4        0.5480    0.5480    0.8303       2.10      68.3
FP16           2   6        0.6100    0.6100    0.8488       3.12     102.5
FP16           2   8        0.6390    0.6390    0.8574       4.14     136.7
FP16           2   10       0.6580    0.6580    0.8637       5.16     170.9
FP16           3   1        0.2830    0.2830    0.7639       0.82      25.6
FP16           3   2        0.4960    0.4960    0.8151       1.58      51.3
FP16           3   4        0.6100    0.6100    0.8488       3.12     102.5
FP16           3   6        0.6490    0.6490    0.8601       4.65     153.8
FP16           3   8        0.6700    0.6700    0.8677       6.18     205.0
FP16           3   10       0.6850    0.6850    0.8720       7.71     256.3
FP16           4   1        0.3830    0.3830    0.7856       1.07      34.2
FP16           4   2        0.5480    0.5480    0.8303       2.09      68.3
FP16           4   4        0.6390    0.6390    0.8574       4.14     136.7
FP16           4   6        0.6700    0.6700    0.8677       6.18     205.0
FP16           4   8        0.6870    0.6870    0.8751       8.22     273.4
FP16           4   10       0.6980    0.6980    0.8782      10.26     341.7
INT8 (bnb)     1   1        0.0060    0.0060    0.5528       0.26       8.5
INT8 (bnb)     1   2        0.0210    0.0210    0.6942       0.47      17.1
INT8 (bnb)     1   4        0.3820    0.3820    0.7852       0.90      34.2
INT8 (bnb)     1   6        0.5010    0.5010    0.8183       1.33      51.3
INT8 (bnb)     1   8        0.5540    0.5540    0.8306       1.75      68.3
INT8 (bnb)     1   10       0.5980    0.5980    0.8455       2.18      85.4
INT8 (bnb)     2   1        0.0210    0.0210    0.6942       0.47      17.1
INT8 (bnb)     2   2        0.3820    0.3820    0.7852       0.90      34.2
INT8 (bnb)     2   4        0.5540    0.5540    0.8306       1.75      68.3
INT8 (bnb)     2   6        0.6210    0.6210    0.8524       2.60     102.5
INT8 (bnb)     2   8        0.6560    0.6560    0.8638       3.44     136.7
INT8 (bnb)     2   10       0.6730    0.6730    0.8698       4.29     170.9
INT8 (bnb)     3   1        0.2800    0.2800    0.7643       0.68      25.6
INT8 (bnb)     3   2        0.5010    0.5010    0.8183       1.32      51.3
INT8 (bnb)     3   4        0.6210    0.6210    0.8524       2.59     102.5
INT8 (bnb)     3   6        0.6670    0.6670    0.8668       3.86     153.8
INT8 (bnb)     3   8        0.6790    0.6790    0.8721       5.13     205.0
INT8 (bnb)     3   10       0.6910    0.6910    0.8751       6.41     256.3
INT8 (bnb)     4   1        0.3820    0.3820    0.7852       0.89      34.2
INT8 (bnb)     4   2        0.5540    0.5540    0.8306       1.75      68.3
INT8 (bnb)     4   4        0.6560    0.6560    0.8638       3.44     136.7
INT8 (bnb)     4   6        0.6790    0.6790    0.8721       5.12     205.0
INT8 (bnb)     4   8        0.6940    0.6940    0.8750       6.81     273.4
INT8 (bnb)     4   10       0.7020    0.7020    0.8778       8.50     341.7
INT4 (fake)    1   1        0.0000    0.0000    0.5612       0.16       8.5
INT4 (fake)    1   2        0.0120    0.0120    0.6097       0.26      17.1
INT4 (fake)    1   4        0.0210    0.0210    0.6375       0.48      34.2
INT4 (fake)    1   6        0.0290    0.0290    0.6490       0.68      51.3
INT4 (fake)    1   8        0.0370    0.0370    0.6536       0.89      68.3
INT4 (fake)    1   10       0.0440    0.0440    0.6530       1.11      85.4
INT4 (fake)    2   1        0.0120    0.0120    0.6097       0.26      17.1
INT4 (fake)    2   2        0.0210    0.0210    0.6375       0.47      34.2
INT4 (fake)    2   4        0.0370    0.0370    0.6536       0.89      68.3
INT4 (fake)    2   6        0.0510    0.0510    0.6524       1.31     102.5
INT4 (fake)    2   8        0.0510    0.0510    0.6555       1.73     136.7
INT4 (fake)    2   10       0.0530    0.0530    0.6565       2.15     170.9
INT4 (fake)    3   1        0.0210    0.0210    0.6277       0.37      25.6
INT4 (fake)    3   2        0.0290    0.0290    0.6490       0.68      51.3
INT4 (fake)    3   4        0.0510    0.0510    0.6524       1.31     102.5
INT4 (fake)    3   6        0.0510    0.0510    0.6553       1.93     153.8
INT4 (fake)    3   8        0.0550    0.0550    0.6573       2.56     205.0
INT4 (fake)    3   10       0.0530    0.0530    0.6602       3.19     256.3
INT4 (fake)    4   1        0.0210    0.0210    0.6375       0.47      34.2
INT4 (fake)    4   2        0.0370    0.0370    0.6536       0.89      68.3
INT4 (fake)    4   4        0.0510    0.0510    0.6555       1.72     136.7
INT4 (fake)    4   6        0.0550    0.0550    0.6573       2.56     205.0
INT4 (fake)    4   8        0.0550    0.0550    0.6601       3.40     273.4
INT4 (fake)    4   10       0.0600    0.0600    0.6588       4.23     341.7
FP32 (bf16)    1   1        0.0060    0.0060    0.5524       0.16       8.5
FP32 (bf16)    1   2        0.0210    0.0210    0.6943       0.26      17.1
FP32 (bf16)    1   4        0.3810    0.3810    0.7850       0.46      34.2
FP32 (bf16)    1   6        0.4950    0.4950    0.8156       0.66      51.3
FP32 (bf16)    1   8        0.5410    0.5410    0.8292       0.87      68.3
FP32 (bf16)    1   10       0.5760    0.5760    0.8396       1.07      85.4
FP32 (bf16)    2   1        0.0210    0.0210    0.6943       0.25      17.1
FP32 (bf16)    2   2        0.3810    0.3810    0.7850       0.46      34.2
FP32 (bf16)    2   4        0.5410    0.5410    0.8292       0.86      68.3
FP32 (bf16)    2   6        0.6060    0.6060    0.8493       1.27     102.5
FP32 (bf16)    2   8        0.6370    0.6370    0.8581       1.67     136.7
FP32 (bf16)    2   10       0.6630    0.6630    0.8657       2.08     170.9
FP32 (bf16)    3   1        0.2810    0.2810    0.7642       0.36      25.6
FP32 (bf16)    3   2        0.4950    0.4950    0.8156       0.66      51.3
FP32 (bf16)    3   4        0.6060    0.6060    0.8493       1.26     102.5
FP32 (bf16)    3   6        0.6510    0.6510    0.8618       1.87     153.8
FP32 (bf16)    3   8        0.6790    0.6790    0.8714       2.47     205.0
FP32 (bf16)    3   10       0.6910    0.6910    0.8747       3.08     256.3
FP32 (bf16)    4   1        0.3810    0.3810    0.7850       0.46      34.2
FP32 (bf16)    4   2        0.5410    0.5410    0.8292       0.86      68.3
FP32 (bf16)    4   4        0.6370    0.6370    0.8581       1.67     136.7
FP32 (bf16)    4   6        0.6790    0.6790    0.8714       2.47     205.0
FP32 (bf16)    4   8        0.6930    0.6930    0.8745       3.28     273.4
FP32 (bf16)    4   10       0.7030    0.7030    0.8787       4.08     341.7
```

### 6.2  Plot depth × quantization grid ─────────────────────────────────────

![Plot from cell 34](extracted_images/plot_sudoku_extreme_cell_34_1.png)

![Plot from cell 34](extracted_images/plot_sudoku_extreme_cell_34_2.png)

### 7.1  Carry-state similarity hooks ────────────────────────────────────────

### 7.2  Compare carry similarity across quantization variants ────────────────

![Plot from cell 37](extracted_images/plot_sudoku_extreme_cell_37_3.png)

### 8.1  Comprehensive model footprint analysis ───────────────────────────────

### 8.2  SRAM footprint visualisation ────────────────────────────────────────

![Plot from cell 41](extracted_images/plot_sudoku_extreme_cell_41_4.png)

### 8.3  Accuracy vs SRAM: the key publishable plot ──────────────────────────

![Plot from cell 42](extracted_images/plot_sudoku_extreme_cell_42_5.png)

### 9.1  Inspect puzzle_emb shape & byte count ───────────────────────────────

### 9.2  Option A — INT8 quantize the embedding buffer ───────────────────────

### 9.3  Option B — SVD Low-Rank Puzzle Embedding ────────────────────────────

### 9.4  Option D — Single-Puzzle Inference (only load 1 row) ────────────────

### 9.5  Footprint summary across embedding strategies ───────────────────────

### 10.2  High-performance direct on-the-fly per-puzzle TTA evaluator ─────────

### #  10.3  Run the High-Performance Direct Source Evaluator ─────────────────────

### 10.2  Re-evaluate all variants with per-puzzle aggregation ────────────────

### 11.1  Per-channel calibration statistics ──────────────────────────────────

### #  11.2  Apply calibrated INT4 and evaluate ──────────────────────────────────

### #  12.1  QAT fine-tuning loop ────────────────────────────────────────────────

### 13.1  Magnitude-based row pruning ────────────────────────────────────────

### 14.1  TorchScript trace of single TRM backbone step ──────────────────────

### 14.2  Latency benchmark: traced model vs eager ────────────────────────────

### 15.1  Build train / val loaders ──────────────────────────────────────────

### #  15.2  QAT loop with val-set evaluation every 10 steps ────────────────────

### 15.3  Plot train loss + val accuracy curve ────────────────────────────────

### 16.1  Build fused INT8 + SinglePuzzle model ───────────────────────────────

### 16.2  Accuracy of fused model (val set, per-puzzle aggregation) ──────────

### 16.3  Save fused model (state dict + embedding table) ────────────────────

### 17.1  CPU-only latency benchmark ─────────────────────────────────────────

### 17.2  Peak SRAM usage during inference (tracemalloc) ─────────────────────

### 17.3  FLOP estimate via torch.profiler ────────────────────────────────────

```text
Estimated FLOPs per H-cycle step : 25660.1 MFLOPs
Full inference (10 steps)        : 256.60 GFLOPs

Power estimate (Cortex-M55 @ ~1 GFLOPS/s, ~1 mW/GFLOP):
  FLOPs                  : 256.601 GFLOPs/puzzle
  Est. energy            : ~256.60 mJ/puzzle
  Puzzles per mAh (3.3V) : ~0
```

### 17.4  Deployment summary table ───────────────────────────────────────────

```text
=== Final Deployment Scorecard ===
                       Backbone MB Total deploy MB Puzzle Exact Cell Acc Fits 4MB Fits 8MB
Config                                                                                    
FP32 (baseline)              19.18           19.19       0.6910   0.8747        ✗        ✗
INT8 (bnb)                    4.80            4.80       0.6910   0.8751        ✗        ✓
INT4 (calibrated)             2.40            2.40       0.0530   0.6602        ✓        ✓
INT8 + Single-Puzzle ★        4.80            4.80       0.8600   0.9464        ✗        ✓

★ = recommended target for 8MB SRAM devices
```

### #  Utility: fix H_init / L_init device mismatch (permanent patch) ───────────

### 18.1  Evaluate all variants at n_sup_max = 1, 2, 4, 8, 16 (H_cycles = default vs 1) ──

```text
======================================================================================
VARIANT: FP16                      (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.2830     0.4960     0.6100     0.6700     0.7160
           Cell           0.7639     0.8151     0.8488     0.8677     0.8851
           Latency         0.8 ms      1.6 ms      3.1 ms      6.2 ms     12.3 ms
           GFLOPs           25.6       51.3      102.5      205.0      410.1
--------------------------------------------------------------------------------------
1          Exact          0.0060     0.0210     0.3830     0.5480     0.6390
           Cell           0.5526     0.6941     0.7856     0.8303     0.8574
           Latency         0.3 ms      0.6 ms      1.1 ms      2.1 ms      4.2 ms
           GFLOPs            8.5       17.1       34.2       68.4      136.7
======================================================================================

======================================================================================
VARIANT: INT8 (bnb)                (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.2800     0.5010     0.6210     0.6790     0.7120
           Cell           0.7643     0.8183     0.8524     0.8721     0.8816
           Latency         0.7 ms      1.3 ms      2.6 ms      5.2 ms     10.2 ms
           GFLOPs           25.6       51.3      102.5      205.0      410.1
--------------------------------------------------------------------------------------
1          Exact          0.0060     0.0210     0.3820     0.5540     0.6560
           Cell           0.5528     0.6942     0.7852     0.8306     0.8638
           Latency         0.3 ms      0.5 ms      0.9 ms      1.8 ms      3.5 ms
           GFLOPs            8.5       17.1       34.2       68.4      136.7
======================================================================================

======================================================================================
VARIANT: INT4 (fake)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.0210     0.0290     0.0510     0.0550     0.0610
           Cell           0.6277     0.6490     0.6524     0.6573     0.6597
           Latency         0.4 ms      0.7 ms      1.3 ms      2.6 ms      5.1 ms
           GFLOPs           25.6       51.3      102.5      205.0      410.1
--------------------------------------------------------------------------------------
1          Exact          0.0000     0.0120     0.0210     0.0370     0.0510
           Cell           0.5612     0.6097     0.6375     0.6536     0.6555
           Latency         0.2 ms      0.3 ms      0.5 ms      0.9 ms      1.7 ms
           GFLOPs            8.5       17.1       34.2       68.4      136.7
======================================================================================

======================================================================================
VARIANT: FP32 (bf16)               (device: cuda)
======================================================================================
H_cycles   Metric            n=1        n=2        n=4        n=8       n=16
--------------------------------------------------------------------------------------
3 (Def)    Exact          0.2840     0.4960     0.6040     0.6690     0.7090
           Cell           0.7640     0.8147     0.8474     0.8680     0.8810
           Latency         0.8 ms      1.6 ms      3.2 ms      6.3 ms     12.7 ms
           GFLOPs           25.6       51.3      102.5      205.0      410.1
--------------------------------------------------------------------------------------
1          Exact          0.0060     0.0210     0.3830     0.5470     0.6370
           Cell           0.5527     0.6943     0.7856     0.8293     0.8563
           Latency         0.3 ms      0.6 ms      1.1 ms      2.2 ms      4.3 ms
           GFLOPs            8.5       17.1       34.2       68.4      136.7
======================================================================================
```

### 19.1  Zero-pad truncation at various context lengths ────────────────────

```text
Full seq_len = 81  (FLOPs ∝ seq_len due to MLP architecture)
   Context  % of full    MLP FLOPs   Puzzle Exact   Cell Acc
--------------------------------------------------------------
        81        100%  ~25.63 GFLOPs          0.2840      0.7640
        40         49%  ~14.80 GFLOPs          0.0000      0.3276
        20         25%  ~9.51 GFLOPs          0.0000      0.2035
        16         20%  ~8.46 GFLOPs          0.0000      0.1767
```

### #  20.1  QAT from calibrated INT4 (lr=1e-6, 300 steps) ────────────────────

### 21.1  Instantiate student TRM and train with KL distillation ─────────────

```text
Teacher params : 5,028,866
Student params : 744,962  (0.15× teacher)
  Step   40/100 | kl=1802.0414 | hard=2.1362
  Step   80/100 | kl=1125.4269 | hard=2.1942
Evaluating student on test set...
  Student (hidden=256, 1L, 1-cycle): exact=0.0000  cell=0.5477  ~1 GFLOPs
```

### 22.1  Monkey-patch softmax attention → linear attention ──────────────────

### 22.2  Retrain linear attention model to recover accuracy ─────────────────────

---
