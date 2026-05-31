# diagnose_modal.py
# ─────────────────────────────────────────────────────────────────────────────
# This script isolates the model loading and high-performance puzzle-by-puzzle
# evaluation to verify that the FP32 baseline accuracy is successfully restored.
# Run this on Modal!
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import torch
import numpy as np
import json

# Add repo to path
sys.path.append(os.getcwd())

# Ensure correct data dir and devices
DATA_DIR = "./data1/arc2test-aug-1000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "step_723914"

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Data Dir: {DATA_DIR}")

# 1. Import model components
from trm import TinyRecursiveReasoningModel_ACTV1
from torch.utils.data import Dataset, DataLoader

class ARCDataset(Dataset):
    def __init__(self, split_dir: str):
        self.inputs = np.load(f"{split_dir}/all__inputs.npy")
        self.labels = np.load(f"{split_dir}/all__labels.npy")

        puzzle_ids  = np.load(f"{split_dir}/all__puzzle_identifiers.npy")
        puzzle_ptr  = np.load(f"{split_dir}/all__puzzle_indices.npy")

        counts = np.diff(puzzle_ptr).astype(np.int64)
        self.per_sample_pids = np.repeat(puzzle_ids, counts)

        assert len(self.inputs) == len(self.per_sample_pids), (
            f"Shape mismatch: inputs={len(self.inputs)}, pids={len(self.per_sample_pids)}"
        )

        with open(f"{split_dir}/../train/dataset.json") as fj:
            meta = json.load(fj)
        self.seq_len               = meta["seq_len"]
        self.vocab_size            = meta["vocab_size"]
        self.num_puzzle_identifiers = meta["num_puzzle_identifiers"]

    def __len__(self): return len(self.inputs)

    def __getitem__(self, i):
        return (
            torch.tensor(self.inputs[i],          dtype=torch.long),
            torch.tensor(self.labels[i],           dtype=torch.long),
            torch.tensor(self.per_sample_pids[i],  dtype=torch.long),
        )

# 2. Re-create the load_arc_model function with aligned H_cycles=3, L_cycles=4, halt_max_steps=16
config_data = """
arch:
  H_cycles: 3
  H_layers: 0
  L_cycles: 4
  L_layers: 2
  expansion: 4
  forward_dtype: bfloat16
  halt_exploration_prob: 0.1
  halt_max_steps: 16
  hidden_size: 512
  num_heads: 8
  pos_encodings: rope
  puzzle_emb_len: 16
  puzzle_emb_ndim: 512
global_batch_size: 512
"""

# Re-implement notebook's cell 7 load_arc_model helper
import yaml

def load_arc_model(checkpoint_path, config_text):
    raw_config = yaml.safe_load(config_text)
    arch = raw_config['arch']
    
    final_config = {
        "batch_size": 32,
        "seq_len": 900,
        "num_puzzle_identifiers": 1191730,
        "vocab_size": 12,
        "H_cycles": arch['H_cycles'],
        "L_cycles": arch['L_cycles'],
        "H_layers": arch['H_layers'],
        "L_layers": arch['L_layers'],
        "hidden_size": arch['hidden_size'],
        "expansion": arch['expansion'],
        "num_heads": arch['num_heads'],
        "pos_encodings": arch['pos_encodings'],
        "halt_max_steps": arch['halt_max_steps'],
        "halt_exploration_prob": arch['halt_exploration_prob'],
        "forward_dtype": arch.get('forward_dtype', 'bfloat16'),
        "mlp_t": arch.get('mlp_t', False),
        "puzzle_emb_ndim": arch.get('puzzle_emb_ndim', 512),
        "puzzle_emb_len": arch.get('puzzle_emb_len', 16),
        "no_ACT_continue": arch.get('no_ACT_continue', True)
    }

    model = TinyRecursiveReasoningModel_ACTV1(config_dict=final_config)
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
            
    puzzle_emb_name = "inner.puzzle_emb.weights"
    expected_shape = model.inner.puzzle_emb.weights.shape
    if puzzle_emb_name in clean_state_dict:
        puzzle_emb = clean_state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(f"Resizing puzzle embedding. Found {puzzle_emb.shape}, Expected {expected_shape}")
            new_weights = torch.empty(expected_shape, dtype=puzzle_emb.dtype, device=puzzle_emb.device)
            mean_emb = torch.mean(puzzle_emb, dim=0)
            new_weights[:] = mean_emb
            min_rows = min(puzzle_emb.shape[0], expected_shape[0])
            new_weights[:min_rows] = puzzle_emb[:min_rows]
            clean_state_dict[puzzle_emb_name] = new_weights
            
    model.load_state_dict(clean_state_dict)
    model.__dict__['model'] = model
    model.eval()
    print("Model loaded successfully!")
    return model

print("Loading model...")
model = load_arc_model(CHECKPOINT_PATH, config_data)

# Helper function to unwrap
def get_inner(m):
    m2 = m.module if hasattr(m, 'module') else m
    return m2._orig_mod if hasattr(m2, '_orig_mod') else m2

# 3. Load ARC dataset
print("Loading dataset...")
test_ds = ARCDataset(f"{DATA_DIR}/test")
print(f"Test dataset has {len(test_ds)} samples.")@torch.no_grad()
def evaluate_arc_per_puzzle(mdl, loader, device="cpu", n_sup_max=16, max_batches=None, return_pass2=False):
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1Carry,
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np, PuzzleIdSeparator
    from evaluators.arc import _crop
    import json
    import os
    import numpy as np
    import time
    from tqdm import tqdm

    inner = get_inner(mdl)
    inner.eval()
    inner = inner.to(device)

    # Load mappings
    with open(os.path.join(DATA_DIR, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(DATA_DIR, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    # High-performance caching of inverse_aug
    aug_cache = {}
    def get_aug(pid):
        if pid not in aug_cache:
            name = identifier_map[pid]
            aug_cache[pid] = inverse_aug(name)
        return aug_cache[pid]

    # Precompute canonical input hashes
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
    
    # We wrap the loader with a standard batch loop
    pbar = tqdm(loader, desc="Evaluating per-puzzle batches", leave=False)
    for batch_idx, (x_batch, y_true, pids) in enumerate(pbar):
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

            bs = loader.batch_size if hasattr(loader, "batch_size") else preds_batch.shape[0]
            sample_idx = batch_idx * bs + i
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

# 5. Run evaluation
print("Running FP32 baseline evaluation...")
# Configure DataLoader with highly optimized batch size for GPU
BATCH_SIZE = 512
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

p1, p2, cell, ms, npuzz = evaluate_arc_per_puzzle(
    model, test_loader, device=DEVICE, n_sup_max=16, return_pass2=True
)
print("\n[Evaluation Complete]")
print(f"Pass@1 Exact: {p1*100:.2f}%")
print(f"Pass@2 Exact: {p2*100:.2f}%")
print(f"Cell Accuracy: {cell*100:.2f}%")
print(f"Latency: {ms:.2f} ms/puzzle")
print(f"Evaluated puzzles: {npuzz}")
