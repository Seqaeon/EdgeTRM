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
print(f"Test dataset has {len(test_ds)} samples.")

# 4. Implement high-performance evaluation
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from evaluators.arc import _crop
import json

@torch.no_grad()
def evaluate_arc_per_puzzle(mdl, dataset, device="cpu", n_sup_max=16, max_puzzles=None, return_pass2=False):
    inner = get_inner(mdl)
    inner.eval()
    inner = inner.to(device)

    # Store puzzle pointers on dataset if not present
    if not hasattr(dataset, "puzzle_ids") or not hasattr(dataset, "puzzle_ptr"):
        split_dir = f"{DATA_DIR}/test"
        if not os.path.exists(f"{split_dir}/all__puzzle_identifiers.npy"):
            split_dir = f"{DATA_DIR}/train"
        dataset.puzzle_ids = np.load(f"{split_dir}/all__puzzle_identifiers.npy")
        dataset.puzzle_ptr = np.load(f"{split_dir}/all__puzzle_indices.npy")

    with open(os.path.join(DATA_DIR, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(DATA_DIR, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    aug_cache = {}
    def get_aug(pid):
        if pid not in aug_cache:
            name = identifier_map[pid]
            aug_cache[pid] = inverse_aug(name)
        return aug_cache[pid]

    n_puzzles = len(dataset.puzzle_ids)
    if max_puzzles is not None:
        n_puzzles = min(n_puzzles, max_puzzles)

    correct_pass1 = 0.0
    correct_pass2 = 0.0
    cell_hits = 0
    n_cells = 0
    evaluated_count = 0

    import time
    t0 = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(n_puzzles), desc="Evaluating puzzles")
    for j in pbar:
        start_idx = int(dataset.puzzle_ptr[j])
        end_idx = int(dataset.puzzle_ptr[j+1])
        if start_idx == end_idx:
            continue

        first_pid = int(dataset.per_sample_pids[start_idx])
        orig_name, _inverse_fn = get_aug(first_pid)
        if orig_name not in test_puzzles:
            continue

        x_puzzle = torch.tensor(dataset.inputs[start_idx:end_idx], dtype=torch.long, device=device)
        y_puzzle = torch.tensor(dataset.labels[start_idx:end_idx], dtype=torch.long, device=device)
        pids_puzzle = torch.tensor(dataset.per_sample_pids[start_idx:end_idx], dtype=torch.long, device=device)

        batch = {
            "inputs": x_puzzle.to(torch.int32),
            "labels": y_puzzle.to(torch.int32),
            "puzzle_identifiers": pids_puzzle.to(torch.int32),
        }

        carry = inner.initial_carry(batch)
        ic = carry.inner_carry
        cast = lambda t: t.to(device)
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

        preds_batch = last_outputs["logits"].argmax(-1).cpu().numpy()
        q_logits = last_outputs.get("q_halt_logits", torch.zeros(preds_batch.shape[0], device=device))
        q_values = q_logits.sigmoid().cpu().numpy().flatten()

        inputs_cpu = x_puzzle.cpu().numpy()

        puzzle_preds = {}
        pred_hash_to_grid = {}

        for i in range(preds_batch.shape[0]):
            pred_seq = preds_batch[i]
            q_val = float(q_values[i])

            inp_seq = inputs_cpu[i]
            input_grid = _inverse_fn(_crop(inp_seq))
            input_hash = grid_hash(input_grid)

            pred_grid = _inverse_fn(_crop(pred_seq))
            pred_hash = grid_hash(pred_grid)

            pred_hash_to_grid[pred_hash] = pred_grid
            puzzle_preds.setdefault(input_hash, [])
            puzzle_preds[input_hash].append((pred_hash, q_val))

        puzzle = test_puzzles[orig_name]
        num_test_correct_p1 = 0
        num_test_correct_p2 = 0

        for pair in puzzle["test"]:
            inp_grid = arc_grid_to_np(pair["input"])
            out_grid = arc_grid_to_np(pair["output"])

            input_hash = grid_hash(inp_grid)
            label_hash = grid_hash(out_grid)

            p_map = {}
            for h, q in puzzle_preds.get(input_hash, []):
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q

            if not len(p_map):
                n_cells += out_grid.size
                continue

            for h, stats in p_map.items():
                stats[1] /= stats[0]

            p_map_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

            top1_hash = p_map_sorted[0][0]
            if top1_hash == label_hash:
                num_test_correct_p1 += 1

            top2_hashes = [kv[0] for kv in p_map_sorted[:2]]
            if label_hash in top2_hashes:
                num_test_correct_p2 += 1

            top_grid = pred_hash_to_grid[top1_hash]
            if top_grid.shape == out_grid.shape:
                cell_hits += (top_grid == out_grid).sum()
            n_cells += out_grid.size

        correct_pass1 += num_test_correct_p1 / len(puzzle["test"])
        correct_pass2 += num_test_correct_p2 / len(puzzle["test"])
        evaluated_count += 1

        current_p1 = correct_pass1 / evaluated_count
        current_cell = cell_hits / n_cells if n_cells > 0 else 0.0
        pbar.set_postfix({"p1": f"{current_p1:.4f}", "cell": f"{current_cell:.4f}"})

    elapsed = time.time() - t0
    pass_1_acc = correct_pass1 / evaluated_count if evaluated_count > 0 else 0.0
    pass_2_acc = correct_pass2 / evaluated_count if evaluated_count > 0 else 0.0
    cell_acc = cell_hits / n_cells if n_cells > 0 else 0.0
    sec_per_puzzle = elapsed / evaluated_count if evaluated_count > 0 else 0.0

    if return_pass2:
        return pass_1_acc, pass_2_acc, cell_acc, sec_per_puzzle
    else:
        return pass_1_acc, cell_acc, sec_per_puzzle

# 5. Run evaluation
print("Running FP32 baseline evaluation...")
p1, p2, cell, ms = evaluate_arc_per_puzzle(model, test_ds, device=DEVICE, n_sup_max=16, return_pass2=True)
print("\n[Evaluation Complete]")
print(f"Pass@1 Exact: {p1*100:.2f}%")
print(f"Pass@2 Exact: {p2*100:.2f}%")
print(f"Cell Accuracy: {cell*100:.2f}%")
print(f"Latency: {ms*1000:.2f} ms/puzzle")
