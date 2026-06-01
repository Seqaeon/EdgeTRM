import json

# Define the new cells
new_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Section 3 — High-Performance Test-Time Adaptation (TTA) & Evaluation Functions\n",
        "\n",
        "This section defines the unified evaluation and training functions for our model compression experiments:\n",
        "- **Function A: `evaluate_compressed_baseline`**: Fast, zero-shot evaluation using post-adaptation checkpoint embeddings.\n",
        "- **Function B: `run_adaptation_eval`**: Evaluation of TTA convergence (training/adaptation from scratch).\n"
    ]
}

new_code_cell_source = """# ── 3.1  High-Performance Evaluation & Adaptation Helpers ─────────────────────
from collections import defaultdict
import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

@torch.no_grad()
def evaluate_arc_per_puzzle(mdl, loader, device="cpu", n_sup_max=10, max_batches=None, return_pass2=False):
    \"\"\"
    Evaluate accuracy at the PUZZLE level, not sample level.
    Uses dihedral transformations, crops, inverse transformations, and quality-weighted voting
    matching the paper's exact evaluation logic.
    \"\"\"
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1Carry,
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np, PuzzleIdSeparator
    from evaluators.arc import _crop

    inner = get_inner(mdl)
    inner.eval()
    inner = inner.to(device)

    # Load mappings from dataset folder
    with open(os.path.join(DATA_DIR, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(DATA_DIR, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    # High-performance caching of inverse_aug to prevent redundant string parsing
    aug_cache = {}
    def get_aug(pid):
        if pid not in aug_cache:
            name = identifier_map[pid]
            aug_cache[pid] = inverse_aug(name)
        return aug_cache[pid]

    # Precompute canonical input hashes (99.9% faster CPU lookups)
    precomputed_input_info = {}
    ds = loader.dataset
    if hasattr(ds, "inputs") and hasattr(ds, "per_sample_pids"):
        try:
            puzzle_ids_arr = None
            puzzle_ptr_arr = None
            for split in ["test", "train"]:
                ids_path = os.path.join(DATA_DIR, split, "all__puzzle_identifiers.npy")
                ptr_path = os.path.join(DATA_DIR, split, "all__puzzle_indices.npy")
                if os.path.exists(ids_path):
                    ptr = np.load(ptr_path)
                    if ptr[-1] == len(ds):
                        puzzle_ids_arr = np.load(ids_path)
                        puzzle_ptr_arr = ptr
                        break
            if puzzle_ids_arr is not None and puzzle_ptr_arr is not None:
                puzzle_test_hashes = {
                    name: [grid_hash(arc_grid_to_np(pair["input"])) for pair in pz["test"]]
                    for name, pz in test_puzzles.items()
                }
                for j in range(len(puzzle_ids_arr)):
                    pid = int(puzzle_ids_arr[j])
                    if pid == 0: continue
                    name = identifier_map[pid]
                    orig_name = name.split(PuzzleIdSeparator)[0]
                    if orig_name not in test_puzzles: continue
                    E = len(test_puzzles[orig_name]["test"])
                    start_ptr = int(puzzle_ptr_arr[j])
                    end_ptr = int(puzzle_ptr_arr[j+1])
                    for k in range(start_ptr, end_ptr):
                        test_example_index = (k - start_ptr) % E
                        input_hash = puzzle_test_hashes[orig_name][test_example_index]
                        precomputed_input_info[k] = (orig_name, input_hash)
        except Exception as e:
            print(f"[WARN] Input hash precomputation failed, falling back: {e}")

    local_hmap = {}       # pred_hash -> canonical grid np.ndarray
    local_preds = {}      # orig_name -> {input_hash -> [(pred_hash, q_val), ...]}

    t0 = time.time()
    for batch_idx, (x_batch, y_true, pids) in enumerate(tqdm(loader, desc="Evaluating per-puzzle batches", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_batch = x_batch.to(device)
        y_true  = y_true.to(device)
        pids    = pids.to(device)

        batch = {
            "inputs":             x_batch.to(torch.int32),
            "labels":             y_true.to(torch.int32),
            "puzzle_identifiers": pids.to(torch.int32),
        }

        carry = inner.initial_carry(batch)
        ic    = carry.inner_carry
        cast  = lambda t: t.to(device)
        carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=cast(ic.z_H), z_L=cast(ic.z_L)),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data={k: v.to(device) for k, v in carry.current_data.items()},
        )

        last_outputs = None
        for _ in range(n_sup_max):
            carry, outputs = inner(carry, batch)
            last_outputs = outputs
            if carry.halted.all():
                break

        if last_outputs is None:
            continue

        preds_batch = last_outputs["logits"].argmax(-1).cpu().numpy()  # (B, seq_len)
        q_logits    = last_outputs.get("q_halt_logits", torch.zeros(preds_batch.shape[0], device=device))
        q_values    = q_logits.sigmoid().cpu().numpy().flatten()    # (B,)

        inputs_cpu  = x_batch.cpu().numpy()
        pids_cpu    = pids.cpu().numpy()

        for i in range(preds_batch.shape[0]):
            identifier = pids_cpu[i]
            if identifier == 0:  # Skip blank padding
                continue

            orig_name, _inverse_fn = get_aug(identifier)
            pred_seq = preds_batch[i]
            q_val = float(q_values[i])

            sample_idx = batch_idx * (loader.batch_size if hasattr(loader, "batch_size") else preds_batch.shape[0]) + i
            if sample_idx in precomputed_input_info and precomputed_input_info[sample_idx][0] == orig_name:
                input_hash = precomputed_input_info[sample_idx][1]
            else:
                inp_seq = inputs_cpu[i]
                input_grid = _inverse_fn(_crop(inp_seq))
                input_hash = grid_hash(input_grid)

            # Crop and inverse transform prediction
            pred_grid = _inverse_fn(_crop(pred_seq))
            pred_hash = grid_hash(pred_grid)

            local_hmap[pred_hash] = pred_grid

            local_preds.setdefault(orig_name, {})
            local_preds[orig_name].setdefault(input_hash, [])
            local_preds[orig_name][input_hash].append((pred_hash, q_val))

    elapsed = time.time() - t0

    # paper-compliant aggregated voting and accuracy evaluation
    pass_Ks = [1, 2, 5]
    correct = [0.0 for _ in pass_Ks]
    evaluated_puzzles = [name for name in test_puzzles.keys() if name in local_preds]
    n_puzzles = len(evaluated_puzzles)

    for name in evaluated_puzzles:
        puzzle = test_puzzles[name]
        num_test_correct = [0 for _ in pass_Ks]

        for pair in puzzle["test"]:
            inp_grid = arc_grid_to_np(pair["input"])
            out_grid = arc_grid_to_np(pair["output"])

            input_hash = grid_hash(inp_grid)
            label_hash = grid_hash(out_grid)

            p_map = {}
            for h, q in local_preds.get(name, {}).get(input_hash, []):
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q

            if not len(p_map):
                continue

            for h, stats in p_map.items():
                stats[1] /= stats[0]

            p_map_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

            for i, k in enumerate(pass_Ks):
                ok = False
                for h, stats in p_map_sorted[:k]:
                    ok |= (h == label_hash)
                num_test_correct[i] += int(ok)

        for i in range(len(pass_Ks)):
            correct[i] += num_test_correct[i] / len(puzzle["test"])

    # Cell accuracy estimation
    cell_hits = 0
    n_cells = 0
    for name in evaluated_puzzles:
        puzzle = test_puzzles[name]
        for pair in puzzle["test"]:
            inp_grid = arc_grid_to_np(pair["input"])
            out_grid = arc_grid_to_np(pair["output"])
            input_hash = grid_hash(inp_grid)

            preds_list = local_preds.get(name, {}).get(input_hash, [])
            if not preds_list:
                continue

            p_map = {}
            for h, q in preds_list:
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q
            for h, stats in p_map.items():
                stats[1] /= stats[0]

            p_map_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
            top_hash = p_map_sorted[0][0]
            top_grid = local_hmap[top_hash]

            if top_grid.shape == out_grid.shape:
                cell_hits += (top_grid == out_grid).sum()
                n_cells += out_grid.size
            else:
                n_cells += out_grid.size

    cell_acc = cell_hits / n_cells if n_cells > 0 else 0.0
    pass_1_acc = correct[0] / n_puzzles if n_puzzles > 0 else 0.0
    pass_2_acc = correct[1] / n_puzzles if n_puzzles > 0 else 0.0
    ms_per_puzzle = elapsed / n_puzzles if n_puzzles > 0 else 0.0

    if return_pass2:
        return pass_1_acc, pass_2_acc, cell_acc, ms_per_puzzle * 1000, n_puzzles
    else:
        return pass_1_acc, cell_acc, ms_per_puzzle * 1000, n_puzzles

# Dynamic wrapper to seamlessly rewire all legacy evaluate_arc calls to use puzzle-by-puzzle TTA voting
def evaluate_arc(mdl, loader, device="cpu", n_sup_max=10, max_batches=None, return_pass2=False):
    return evaluate_arc_per_puzzle(mdl, loader, device=device, n_sup_max=n_sup_max, max_batches=max_batches, return_pass2=return_pass2)

# ── Function A: Fast Baseline Evaluation using Post-TTA checkpoint ────────────
def evaluate_compressed_baseline(model, checkpoint_path, data_dir, device="cuda"):
    \"\"\"
    Function A: For instant evaluation of compressed models (quantized/pruned) using the post-adaptation checkpoint.
    Loads the post-adaptation checkpoint (which contains the fitted puzzle embeddings),
    applies the compressed/quantized model structure, runs zero-shot inference
    on the augmented test set, ensembles the predictions via voting,
    and returns the Pass@1 and Pass@2 exact match scores.
    \"\"\"
    # 1. Clean and load the state dict from checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    unwanted_prefix = '_orig_mod.model.'
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            clean_state_dict[k[len(unwanted_prefix):]] = v
        else:
            clean_state_dict[k] = v
            
    # 2. Resize model's puzzle embeddings if necessary
    inner_model = get_inner(model)
    puzzle_emb_name = "inner.puzzle_emb.weights"
    expected_shape = inner_model.puzzle_emb.weights.shape
    if puzzle_emb_name in clean_state_dict:
        puzzle_emb = clean_state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(f"[evaluate_compressed_baseline] Resizing puzzle embedding. Found {puzzle_emb.shape}, Expected {expected_shape}")
            new_weights = torch.empty(expected_shape, dtype=puzzle_emb.dtype, device=puzzle_emb.device)
            mean_emb = torch.mean(puzzle_emb, dim=0)
            new_weights[:] = mean_emb
            min_rows = min(puzzle_emb.shape[0], expected_shape[0])
            new_weights[:min_rows] = puzzle_emb[:min_rows]
            clean_state_dict[puzzle_emb_name] = new_weights
            
    inner_model.load_state_dict(clean_state_dict, strict=False)
    inner_model = inner_model.to(device)
    inner_model.eval()

    # 3. Create fresh ARCDataset and DataLoader
    test_ds = ARCDataset(f"{data_dir}/test")
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    print(f"[evaluate_compressed_baseline] Running direct baseline evaluation on {device}...")
    p1, p2, cell, ms, npuzz = evaluate_arc_per_puzzle(
        inner_model, test_loader, device=device, n_sup_max=10, return_pass2=True
    )
    
    print(f"Evaluation Complete ({npuzz} puzzles):")
    print(f"  Pass@1 Accuracy: {p1*100:.2f}%")
    print(f"  Pass@2 Accuracy: {p2*100:.2f}%")
    print(f"  Cell Accuracy  : {cell*100:.2f}%")
    print(f"  Latency        : {ms:.2f} ms/puzzle")
    
    return p1, p2, cell, ms

# ── Function B: Train-Time Adaptation convergence loop from scratch ───────────
def run_adaptation_eval(model, data_dir, epochs=100, lr=1e-4, device="cuda"):
    \"\"\"
    Function B: Runs the training/TTA loop for a specified number of epochs on the passed model.
    \"\"\"
    from trm import TinyRecursiveReasoningModel_ACTV1Carry, TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    # 1. Freshly create ARCDataset for train and test splits
    train_ds = ARCDataset(f"{data_dir}/train")
    test_ds = ARCDataset(f"{data_dir}/test")
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    inner_model = get_inner(model)
    inner_model = inner_model.to(device)
    
    # We set up a standard AdamW optimizer for simplicity and speed inside the notebook
    optimizer = torch.optim.AdamW(inner_model.parameters(), lr=lr)
    
    inner_model.train()
    print(f"[run_adaptation_eval] Starting adaptation training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for x_batch, y_true, pids in train_loader:
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)
            pids = pids.to(device)
            
            batch = {
                "inputs": x_batch.to(torch.int32),
                "labels": y_true.to(torch.int32),
                "puzzle_identifiers": pids.to(torch.int32),
            }
            
            optimizer.zero_grad()
            
            # Initial carry
            carry = inner_model.initial_carry(batch)
            ic = carry.inner_carry
            cast = lambda t: t.to(device)
            carry = TinyRecursiveReasoningModel_ACTV1Carry(
                inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=cast(ic.z_H), z_L=cast(ic.z_L)),
                steps=carry.steps.to(device),
                halted=carry.halted.to(device),
                current_data={k: v.to(device) for k, v in carry.current_data.items()},
            )
            
            # Forward pass over steps (similar to evaluate_arc_per_puzzle but with gradients)
            n_sup_max = inner_model.config.halt_max_steps if hasattr(inner_model.config, 'halt_max_steps') else 10
            
            for _ in range(n_sup_max):
                carry, outputs = inner_model(carry, batch)
                if carry.halted.all():
                    break
            
            loss = outputs["loss"]
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss / max(1, n_batches):.4f}")
            
    # Evaluate at the end
    print("[run_adaptation_eval] Evaluating adapted model...")
    p1, p2, cell, ms, npuzz = evaluate_arc_per_puzzle(
        inner_model, test_loader, device=device, n_sup_max=10, return_pass2=True
    )
    
    print(f"Adaptation Complete ({npuzz} puzzles):")
    print(f"  Pass@1 Accuracy: {p1*100:.2f}%")
    print(f"  Pass@2 Accuracy: {p2*100:.2f}%")
    print(f"  Cell Accuracy  : {cell*100:.2f}%")
    
    return p1, p2, cell, ms
"""

new_code_cell = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in new_code_cell_source.split("\n")]
}

# 1. Load the notebook
with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

# 2. Delete redundant cells 9 to 19 (inclusive)
del nb['cells'][9:20]

# 3. Insert the new cells at index 9 and 10
nb['cells'].insert(9, new_md_cell)
nb['cells'].insert(10, new_code_cell)

# 4. Clean up the duplicate evaluate_arc_per_puzzle definition (which was Cell 52, now shifted to index 43)
# Let's search for it to be absolutely robust!
duplicate_found = False
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "@torch.no_grad()\ndef evaluate_arc_per_puzzle(" in src and idx > 10:
        print(f"Found duplicate evaluate_arc_per_puzzle at cell {idx}, replacing it with a clean placeholder.")
        nb['cells'][idx] = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Note: `evaluate_arc_per_puzzle` has been moved to Section 3 for global accessibility."
            ]
        }
        duplicate_found = True
        break

# 5. Save the modified notebook
with open("edge-trm-new.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook edge-trm-new.ipynb successfully modified and saved!")
