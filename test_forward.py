
import os
import torch
import sys
from pathlib import Path
sys.path.insert(0, '/root/EdgeTRM/TinyRecursiveModels')
from trm import TinyRecursiveReasoningModel_ACTV1

import yaml
def load_arc_model(checkpoint_path, config_text):
    raw_config = yaml.safe_load(config_text)
    arch = raw_config['arch']
    final_config = {
        "batch_size": 32,
        "seq_len": 900,
        "num_puzzle_identifiers": 30670,
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
    # create a mock state dict instead of loading
    # model.eval()
    return model

config_data = """
arch:
  H_cycles: 3
  H_layers: 0
  L_cycles: 4
  L_layers: 2
  expansion: 4
  forward_dtype: float32
  halt_exploration_prob: 0.1
  halt_max_steps: 16
  hidden_size: 512
  num_heads: 8
  pos_encodings: rope
  puzzle_emb_len: 16
  puzzle_emb_ndim: 512
global_batch_size: 512
"""

model = load_arc_model("", config_data)
batch = {
    "inputs": torch.zeros(2, 900, dtype=torch.int32),
    "labels": torch.zeros(2, 900, dtype=torch.int32),
    "puzzle_identifiers": torch.zeros(2, dtype=torch.int32)
}
carry = model.initial_carry(batch)
print("initial_carry shapes:")
print(" z_H:", carry.inner_carry.z_H.shape)
print(" z_L:", carry.inner_carry.z_L.shape)

try:
    carry, outputs = model(carry, batch)
    print("forward succeeded!")
    print("logits shape:", outputs["logits"].shape)
except Exception as e:
    import traceback
    traceback.print_exc()
