import sys
sys.path.append('TinyRecursiveModels')
import torch
import yaml
import json
from torch.utils.data import DataLoader
from dataset.build_arc_dataset import ARCDataset
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# 1. Load config
with open("trm_maze_hard/all_config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

# 2. Build model and load checkpoint
checkpoint_path = "checkpoints/trm_scratch_maze/step_34500.pt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint['model']

# Clean prefixes
unwanted_prefix = '_orig_mod.model.'
clean_state_dict = {}
for k, v in state_dict.items():
    if k.startswith(unwanted_prefix):
        clean_state_dict[k[len(unwanted_prefix):]] = v
    else:
        clean_state_dict[k] = v

model = TinyRecursiveReasoningModel_ACTV1(config_dict['arch'])
model.inner.load_state_dict(clean_state_dict, strict=False)
model = model.to('cuda')
model.eval()

# 3. Create dataset
test_ds = ARCDataset("data/maze-30x30-hard-1k/test")
test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

# 4. Helper to get inner
def get_inner(m):
    m2 = m.module if hasattr(m, 'module') else m
    return m2._orig_mod if hasattr(m2, '_orig_mod') else m2

# 5. Define evaluate helper
@torch.no_grad()
def eval_func(mdl, loader, h_cycles):
    inner = get_inner(mdl)
    inner.config.H_cycles = h_cycles
    inner.inner.config.H_cycles = h_cycles
    
    total_samples = 0
    total_correct = 0
    for inputs, labels, pids in loader:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        pids = pids.to('cuda')
        
        batch = {
            "inputs": inputs.to(torch.int32),
            "labels": labels.to(torch.int32),
            "puzzle_identifiers": pids.to(torch.int32),
        }
        
        # Forward step
        carry = inner.initial_carry(batch)
        # Force H_cycles to check
        for _ in range(10): # n_sup_max
            carry, outputs = inner(carry, batch)
            
        preds = torch.argmax(outputs["logits"], dim=-1)
        mask = (labels != 0)
        is_correct = mask & (preds == labels)
        loss_counts = mask.sum(-1)
        seq_is_correct = (is_correct.sum(-1) == loss_counts) & (loss_counts > 0)
        total_correct += seq_is_correct.sum().item()
        total_samples += inputs.shape[0]
        
    print(f"H_cycles={h_cycles} | Uncompiled Accuracy: {total_correct/total_samples*100:.2f}%")

eval_func(model, test_loader, 1)
eval_func(model, test_loader, 2)
eval_func(model, test_loader, 3)
eval_func(model, test_loader, 4)
