"""
Injects Sections 15, 16, 17 into edge-trm-copy.ipynb after cell index 66 (last summary).
Run with:  python add_sections_15_17.py
"""
import json, copy

NB_PATH = "edge-trm-copy.ipynb"

def code_cell(source):
    return {
        "cell_type": "code",
        "metadata": {"collapsed": False},
        "source": source if isinstance(source, list) else [source],
        "execution_count": None,
        "outputs": [],
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }

# ── Section 15 ─────────────────────────────────────────────────────────────
S15_MD = """\
---
## Section 15 — QAT with Proper Train / Val Split

The §12 QAT evaluated on the **same** data it trained on → loss 632→1.2 in 50 steps looks
like overfitting. This section re-runs QAT with an 80/20 split and evaluates on the held-out
val set to get an honest accuracy number.
"""

S15_1 = """\
# ── 15.1  Build train / val loaders ──────────────────────────────────────────
import math
from torch.utils.data import Subset

if train_loader is not None:
    full_ds   = train_loader.dataset
    n_total   = len(full_ds)
    n_val     = max(1, math.floor(n_total * 0.20))
    n_train   = n_total - n_val

    # Deterministic split (no shuffle of indices so puzzle groups stay intact)
    all_idx   = list(range(n_total))
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:]

    train_sub = Subset(full_ds, train_idx)
    val_sub   = Subset(full_ds, val_idx)

    BATCH = train_loader.batch_size
    qat_train_loader = torch.utils.data.DataLoader(
        train_sub, batch_size=BATCH, shuffle=True,
        collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None,
        drop_last=True,
    )
    qat_val_loader = torch.utils.data.DataLoader(
        val_sub, batch_size=BATCH, shuffle=False,
        collate_fn=train_loader.collate_fn if hasattr(train_loader, 'collate_fn') else None,
        drop_last=False,
    )
    print(f"Train split : {len(train_sub):,} samples")
    print(f"Val   split : {len(val_sub):,}  samples")
else:
    print("[SKIP] train_loader not available.")
"""

S15_2 = """\
# ── 15.2  QAT loop with val-set evaluation every 10 steps ────────────────────
import copy, numpy as np

def run_qat_validated(base_mdl, tr_loader, v_loader,
                      n_steps=100, lr=1e-5, device="cuda",
                      eval_every=10, max_eval_batches=20):
    \"\"\"
    QAT fine-tune on tr_loader, evaluate on v_loader every eval_every steps.
    Returns (trained_model, history_dict).
    \"\"\"
    from models.layers import CastedLinear

    # Build INT4 fake-quant copy
    m = copy.deepcopy(base_mdl)
    inner_m = get_inner(m)
    replaced = 0
    for name, mod in list(inner_m.named_modules()):
        if isinstance(mod, (nn.Linear, CastedLinear)):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = inner_m if parent_name == "" else dict(inner_m.named_modules())[parent_name]
            bias = mod.bias if (hasattr(mod, 'bias') and mod.bias is not None) else None
            fq = FakeQuantINT4(mod.weight, bias)
            setattr(parent, child_name, fq)
            replaced += 1
    print(f"  Replaced {replaced} layers → FakeQuantINT4")

    m = m.to(device).train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    history   = {"step": [], "train_loss": [], "val_exact": [], "val_cell": []}

    tr_iter = iter(tr_loader)
    for step in range(n_steps):
        # ── training step ──────────────────────────────────────────────────
        try:
            batch = next(tr_iter)
        except StopIteration:
            tr_iter = iter(tr_loader)
            batch = next(tr_iter)

        batch_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
        carry = m.initial_carry(batch_d)
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(4):
            carry, outputs = m(carry, batch_d)
            logits = outputs["logits"]
            labels = batch_d["labels"]
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1), ignore_index=-100)
            total_loss = total_loss + loss
            if carry.halted.all(): break

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()

        # ── val evaluation ─────────────────────────────────────────────────
        if (step + 1) % eval_every == 0:
            pexact, cacc, _, _ = evaluate_arc_per_puzzle(
                m, v_loader, device=device, n_sup_max=16,
                max_batches=max_eval_batches)
            history["step"].append(step + 1)
            history["train_loss"].append(total_loss.item())
            history["val_exact"].append(pexact)
            history["val_cell"].append(cacc)
            print(f"  Step {step+1:4d}/{n_steps} | "
                  f"loss={total_loss.item():.4f} | "
                  f"val_exact={pexact:.4f} | val_cell={cacc:.4f}")

    m.eval()
    return m, history

if 'model' in dir() and 'qat_train_loader' in dir() and 'FakeQuantINT4' in dir():
    print("Starting validated QAT (100 steps, lr=1e-5)...")
    qat_model_v, qat_history = run_qat_validated(
        model, qat_train_loader, qat_val_loader,
        n_steps=100, lr=1e-5, device=str(DEVICE),
        eval_every=10, max_eval_batches=20)
else:
    print("[SKIP] Prerequisites missing — run §12 cell first for FakeQuantINT4.")
"""

S15_3 = """\
# ── 15.3  Plot train loss + val accuracy curve ────────────────────────────────
if 'qat_history' in dir() and qat_history["step"]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(qat_history["step"], qat_history["train_loss"], "b-o", ms=4)
    ax1.set_xlabel("QAT Step"); ax1.set_ylabel("Train Loss")
    ax1.set_title("QAT Training Loss"); ax1.grid(True, alpha=0.3)

    ax2.plot(qat_history["step"], qat_history["val_exact"], "g-o", ms=4, label="Puzzle Exact")
    ax2.plot(qat_history["step"], qat_history["val_cell"],  "r-o", ms=4, label="Cell Acc")
    ax2.set_xlabel("QAT Step"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Val Accuracy During QAT"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("qat_val_curve.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: qat_val_curve.png")

    best_idx = max(range(len(qat_history["val_exact"])),
                   key=lambda i: qat_history["val_exact"][i])
    print(f"\\nBest val result at step {qat_history['step'][best_idx]}:")
    print(f"  Puzzle exact : {qat_history['val_exact'][best_idx]:.4f}")
    print(f"  Cell acc     : {qat_history['val_cell'][best_idx]:.4f}")
else:
    print("[SKIP] No QAT history — run §15.2 first.")
"""

# ── Section 16 ─────────────────────────────────────────────────────────────
S16_MD = """\
---
## Section 16 — INT8 Backbone + Single-Puzzle Fused Artifact

Best accuracy/size combo identified in the analysis:
- **INT8 backbone** (bitsandbytes): 6.7 MB, preserves ~19-20% cell accuracy
- **Single-puzzle embedding**: 2 KB SRAM, full table on flash

This section fuses both into a single deployment-ready TorchScript `.pt` file
and measures accuracy on the held-out val set.
"""

S16_1 = """\
# ── 16.1  Build fused INT8 + SinglePuzzle model ───────────────────────────────
import copy

if 'model' in dir() and 'INT8PuzzleEmbedding' in dir() and 'SinglePuzzleEmbedding' in dir():
    print("Building INT8-backbone + single-puzzle model...")

    # Step 1: Start from INT8 (bnb) backbone variant.
    # variants stores (model, device_str) tuples — unpack accordingly.
    int8_backbone, _dev = variants["INT8 (bnb)"]
    fused_model = copy.deepcopy(int8_backbone)

    # Step 2: Get the puzzle_emb from the *original* model's inner (int8 model
    # is already the unwrapped inner, but puzzle_emb lives on .inner.puzzle_emb).
    orig_inner  = get_inner(model)           # original FP32 model
    orig_emb    = orig_inner.inner.puzzle_emb  # the real embedding module

    # Step 3: Replace puzzle_emb on the INT8 model with SinglePuzzleEmbedding.
    fused_model.inner.puzzle_emb = SinglePuzzleEmbedding(orig_emb)

    fused_model = fused_model.to(DEVICE).eval()

    # Size accounting
    import io
    def state_dict_kb(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return buf.tell() / 1024

    backbone_kb  = state_dict_kb(fused_model)
    emb_full_kb  = orig_emb.weights.numel() * 1 / 1024  # INT8 table (1 byte/element)
    emb_row_kb   = orig_emb.weights.shape[1] * 4 / 1024 # 1 FP32 row in SRAM

    print(f"\\nFused model breakdown:")
    print(f"  INT8 backbone .pt size : {backbone_kb:>10.1f} KB  ({backbone_kb/1024:.2f} MB)")
    print(f"  INT8 emb table (flash) : {emb_full_kb:>10.1f} KB  ({emb_full_kb/1024:.2f} MB)")
    print(f"  Active emb row (SRAM)  : {emb_row_kb:>10.2f} KB")
    print(f"\\n  ✓ SRAM during inference: {backbone_kb + emb_row_kb:.1f} KB  ({(backbone_kb+emb_row_kb)/1024:.2f} MB)")
    print(f"  Fits 4MB SRAM?  {'✓' if backbone_kb + emb_row_kb < 4096 else '✗'}")
    print(f"  Fits 8MB SRAM?  {'✓' if backbone_kb + emb_row_kb < 8192 else '✗'}")
else:
    print("[SKIP] model or helpers not available — run §9 cells first.")
"""

S16_2 = """\
# ── 16.2  Accuracy of fused model (val set, per-puzzle aggregation) ──────────
if 'fused_model' in dir() and 'qat_val_loader' in dir():
    loader_to_use = qat_val_loader
    label = "val"
elif 'fused_model' in dir() and train_loader is not None:
    loader_to_use = train_loader
    label = "train (no val split available)"
else:
    loader_to_use = None

if loader_to_use is not None:
    print(f"Evaluating fused INT8+SinglePuzzle model on {label} set...")
    fp32_exact, fp32_cell, _, _ = evaluate_arc_per_puzzle(
        model, loader_to_use, device=str(DEVICE), n_sup_max=16, max_batches=30)
    fused_exact, fused_cell, fused_ms, fused_n = evaluate_arc_per_puzzle(
        fused_model, loader_to_use, device=str(DEVICE), n_sup_max=16, max_batches=30)

    print(f"\\n{'Variant':<30} {'Puzzle Exact':>12} {'Cell Acc':>10} {'ms/puzzle':>10}")
    print("-" * 65)
    print(f"  {'FP32 (bf16) baseline':<28} {fp32_exact:>12.4f} {fp32_cell:>10.4f}")
    print(f"  {'INT8 + Single-Puzzle (fused)':<28} {fused_exact:>12.4f} {fused_cell:>10.4f} {fused_ms:>10.2f}")
    delta_exact = fused_exact - fp32_exact
    delta_cell  = fused_cell  - fp32_cell
    sign_e = '+' if delta_exact >= 0 else ''
    sign_c = '+' if delta_cell  >= 0 else ''
    print(f"  {'Δ vs FP32':<28} {sign_e}{delta_exact:>12.4f} {sign_c}{delta_cell:>10.4f}")
else:
    print("[SKIP] fused_model not available.")
"""

S16_3 = """\
# ── 16.3  Export fused model as TorchScript ───────────────────────────────────
import os, time

if 'fused_model' in dir() and 'TRMBackboneStep' in dir():
    try:
        inner_fused = get_inner(fused_model)
        dev         = str(DEVICE)
        seq_len     = inner_fused.config.seq_len
        emb_len     = inner_fused.inner.puzzle_emb_len
        hidden      = inner_fused.config.hidden_size
        # SinglePuzzleEmbedding uses one row → emb_dim from original weights
        orig_weights = inner_fused.inner.puzzle_emb.orig_weights
        emb_dim     = orig_weights.shape[1]

        wrapper_fused = TRMBackboneStep(inner_fused).eval().to(dev).float()

        x_d   = torch.zeros(1, seq_len,              dtype=torch.int64,  device=dev)
        emb_d = torch.zeros(1, emb_dim,              dtype=torch.float32, device=dev)
        zH_d  = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32, device=dev)
        zL_d  = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32, device=dev)

        print("Tracing INT8+SinglePuzzle model...")
        t0 = time.perf_counter()
        with torch.no_grad():
            traced_fused = torch.jit.trace(wrapper_fused, (x_d, emb_d, zH_d, zL_d), strict=False)
        dt = time.perf_counter() - t0

        fused_pt_path = "trm_int8_singlepuzzle.pt"
        traced_fused.save(fused_pt_path)
        size_mb = os.path.getsize(fused_pt_path) / 1024 / 1024
        print(f"  Trace done in {dt:.1f}s")
        print(f"  Saved: {fused_pt_path}  ({size_mb:.2f} MB)")

        # Save the full embedding table separately (simulates flash storage)
        emb_table_path = "puzzle_emb_int8_table.bin"
        emb_int8 = orig_weights.to(torch.int8).cpu()
        emb_int8.numpy().tofile(emb_table_path)
        emb_mb = os.path.getsize(emb_table_path) / 1024 / 1024
        print(f"  Saved: {emb_table_path} ({emb_mb:.2f} MB)  ← flash storage")

        print(f"\\n  Deployment package:")
        print(f"    {fused_pt_path:<40} {size_mb:.2f} MB  (SRAM)")
        print(f"    {emb_table_path:<40} {emb_mb:.2f} MB  (flash, seek-and-load per puzzle)")
        print(f"    Total package size: {size_mb + emb_mb:.2f} MB")
    except Exception as e:
        import traceback; traceback.print_exc()
else:
    print("[SKIP] fused_model or TRMBackboneStep not available.")
"""

# ── Section 17 ─────────────────────────────────────────────────────────────
S17_MD = """\
---
## Section 17 — Simulated Edge Deployment Profile

No physical edge hardware available, so we simulate the deployment environment:

- **CPU-only latency**: move the INT8+SinglePuzzle model to CPU and benchmark there
  (representative of ARM Cortex-M55 / ESP32-S3 class devices)
- **Peak SRAM usage**: track memory high-watermark during inference with `tracemalloc`
- **FLOP count**: estimate via `torch.profiler`
- **Estimated power**: rough estimate using published Cortex-M55 efficiency figures

*Note: CPU latency on a server CPU is not the same as on-device, but it gives a useful
order-of-magnitude estimate and validates that inference is architecturally feasible.*
"""

S17_1 = """\
# ── 17.1  CPU-only latency benchmark ─────────────────────────────────────────
import time, gc

if 'fused_model' in dir():
    print("Moving fused model to CPU for edge-device latency simulation...")
    cpu_model = copy.deepcopy(fused_model).cpu().eval()

    inner_cpu = get_inner(cpu_model)
    seq_len   = inner_cpu.config.seq_len
    emb_len   = inner_cpu.inner.puzzle_emb_len
    hidden    = inner_cpu.config.hidden_size
    emb_dim   = inner_cpu.inner.puzzle_emb.orig_weights.shape[1]

    x_cpu   = torch.zeros(1, seq_len,               dtype=torch.int64)
    emb_cpu = torch.zeros(1, emb_dim,               dtype=torch.float32)
    zH_cpu  = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32)
    zL_cpu  = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32)

    wrapper_cpu = TRMBackboneStep(inner_cpu).eval().cpu().float()

    N_WARMUP, N_BENCH = 2, 5   # fewer reps — CPU is slow
    for _ in range(N_WARMUP):
        with torch.no_grad():
            wrapper_cpu(x_cpu, emb_cpu, zH_cpu, zL_cpu)

    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        with torch.no_grad():
            wrapper_cpu(x_cpu, emb_cpu, zH_cpu, zL_cpu)
    ms_per_step = (time.perf_counter() - t0) / N_BENCH * 1000

    n_sup_max = 16
    ms_per_puzzle = ms_per_step * n_sup_max

    print(f"\\nCPU latency (server CPU, 1 H-cycle step)  : {ms_per_step:.1f} ms/step")
    print(f"Full inference ({n_sup_max} H-cycles)             : {ms_per_puzzle:.0f} ms/puzzle")
    print(f"\\nEstimated on-device scaling (relative to server CPU):")
    for label, factor in [("Cortex-A55 (mobile)", 3), ("Cortex-M55 (MCU)", 20), ("ESP32-S3 (LX7)", 30)]:
        est_ms = ms_per_puzzle * factor
        print(f"  {label:<35}: ~{est_ms/1000:.1f}s per puzzle  (×{factor} slowdown estimate)")
else:
    print("[SKIP] fused_model not available.")
"""

S17_2 = """\
# ── 17.2  Peak SRAM usage during inference (tracemalloc) ─────────────────────
import tracemalloc

if 'wrapper_cpu' in dir():
    gc.collect()
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()

    with torch.no_grad():
        for _ in range(3):
            wrapper_cpu(x_cpu, emb_cpu, zH_cpu, zL_cpu)

    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Peak delta
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    peak_kb = sum(s.size_diff for s in stats if s.size_diff > 0) / 1024
    top_stats = sorted(stats, key=lambda s: s.size_diff, reverse=True)[:5]

    print(f"Peak SRAM delta during inference : {peak_kb:,.0f} KB  ({peak_kb/1024:.2f} MB)")
    print(f"\\nTop allocations:")
    for s in top_stats:
        if s.size_diff > 0:
            print(f"  {s.size_diff/1024:>8.1f} KB  {str(s.traceback[0])}")
else:
    print("[SKIP] Run §17.1 first.")
"""

S17_3 = """\
# ── 17.3  FLOP estimate via torch.profiler ────────────────────────────────────
try:
    from torch.profiler import profile, ProfilerActivity, record_function

    if 'wrapper_cpu' in dir():
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True,
                     with_flops=True) as prof:
            with record_function("trm_step"):
                with torch.no_grad():
                    wrapper_cpu(x_cpu, emb_cpu, zH_cpu, zL_cpu)

        total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
        print(f"Estimated FLOPs per H-cycle step : {total_flops/1e6:.1f} MFLOPs")
        print(f"Full inference ({16} steps)        : {total_flops*16/1e9:.2f} GFLOPs")

        print(f"\\nPower estimate (Cortex-M55 @ ~1 GFLOPS/s, ~1 mW/GFLOP):")
        gflops_total = total_flops * 16 / 1e9
        print(f"  FLOPs                  : {gflops_total:.3f} GFLOPs/puzzle")
        print(f"  Est. energy            : ~{gflops_total:.2f} mJ/puzzle")
        print(f"  Puzzles per mAh (3.3V) : ~{3600*3.3/1000/max(gflops_total,0.001):.0f}")
    else:
        print("[SKIP] Run §17.1 first.")
except Exception as e:
    print(f"[WARN] torch.profiler FLOP counting: {e}")
    print("FLOPs unavailable on this PyTorch build — skipping.")
"""

S17_4 = """\
# ── 17.4  Deployment summary table ───────────────────────────────────────────
if 'fused_exact' in dir() and 'fused_cell' in dir():
    import pandas as pd

    rows = [
        {"Config":           "FP32 (baseline)",
         "Backbone MB":      26676/1024,
         "Total deploy MB":  88538/1024,
         "Puzzle Exact":     fp32_exact,
         "Cell Acc":         fp32_cell,
         "Fits 4MB":         "✗",
         "Fits 8MB":         "✗"},
        {"Config":           "INT8 (bnb)",
         "Backbone MB":      6669/1024,
         "Total deploy MB":  68531/1024,
         "Puzzle Exact":     0.0246,
         "Cell Acc":         0.2003,
         "Fits 4MB":         "✗",
         "Fits 8MB":         "✗"},
        {"Config":           "INT4 (calibrated)",
         "Backbone MB":      3334.5/1024,
         "Total deploy MB":  3334.5/1024 + 2/1024,
         "Puzzle Exact":     0.0148,
         "Cell Acc":         0.1902,
         "Fits 4MB":         "✓",
         "Fits 8MB":         "✓"},
        {"Config":           "INT8 + Single-Puzzle ★",
         "Backbone MB":      6669/1024,
         "Total deploy MB":  6669/1024 + 2/1024,
         "Puzzle Exact":     fused_exact,
         "Cell Acc":         fused_cell,
         "Fits 4MB":         "✗",
         "Fits 8MB":         "✓"},
    ]

    df = pd.DataFrame(rows).set_index("Config")
    df["Backbone MB"] = df["Backbone MB"].map("{:.2f}".format)
    df["Total deploy MB"] = df["Total deploy MB"].map("{:.2f}".format)
    df["Puzzle Exact"] = df["Puzzle Exact"].map("{:.4f}".format)
    df["Cell Acc"] = df["Cell Acc"].map("{:.4f}".format)
    print("\\n=== Final Deployment Scorecard ===")
    print(df.to_string())
    print("\\n★ = recommended target for 8MB SRAM devices")
else:
    print("[SKIP] Run §16.2 first.")
"""

# ── Inject into notebook ─────────────────────────────────────────────────────
with open(NB_PATH) as f:
    nb = json.load(f)

new_cells = [
    md_cell(S15_MD),
    code_cell(S15_1),
    code_cell(S15_2),
    code_cell(S15_3),
    md_cell(S16_MD),
    code_cell(S16_1),
    code_cell(S16_2),
    code_cell(S16_3),
    md_cell(S17_MD),
    code_cell(S17_1),
    code_cell(S17_2),
    code_cell(S17_3),
    code_cell(S17_4),
]

# Insert after cell 66 (the last summary markdown)
INSERT_AFTER = 66
nb["cells"] = nb["cells"][:INSERT_AFTER + 1] + new_cells + nb["cells"][INSERT_AFTER + 1:]

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"✓ Injected {len(new_cells)} new cells after cell {INSERT_AFTER}")
print(f"  Total cells now: {len(nb['cells'])}")
print()
print("New sections added:")
print("  §15 — QAT with proper train/val split (cells 67–70)")
print("  §16 — INT8 + Single-Puzzle fused artifact (cells 71–74)")
print("  §17 — Simulated edge deployment profile (cells 75–79)")
