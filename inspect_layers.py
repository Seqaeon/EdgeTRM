import sys
sys.path.append('TinyRecursiveModels')
import torch
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

# Define model config matching the Maze model
config = type('Config', (), {
    'd_model': 512,
    'L_cycles': 4,
    'H_cycles': 3,
    'vocab_size': 6,
    'num_puzzle_identifiers': 1,
    'seq_len': 900
})()

model = TinyRecursiveReasoningModel_ACTV1(config)
for name, module in model.named_modules():
    if hasattr(module, 'weight') and module.weight is not None:
        print(f"Layer: {name:<40} Class: {module.__class__.__name__:<25} Shape: {list(module.weight.shape)}")
