import os
import torch
import json
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from trm_backbone_step import TRMBackboneStep
from torch.profiler import profile, ProfilerActivity, record_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("eval_checkpoint/config.json", "r") as f:
    config_data = json.load(f)

# Initialize model
model = TinyRecursiveReasoningModel_ACTV1(config_data)

# Load checkpoint
checkpoint = "eval_checkpoint/step_2790"
if os.path.exists(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    unwanted_prefix = '_orig_mod.model.'
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(unwanted_prefix):
            clean_state_dict[k[len(unwanted_prefix):]] = v
        else:
            clean_state_dict[k] = v
    # Resize puzzle_emb
    inner_model = model.inner
    puzzle_emb_name = "inner.puzzle_emb.weights"
    if puzzle_emb_name in clean_state_dict:
        expected_shape = inner_model.puzzle_emb.weights.shape
        ckpt_shape = clean_state_dict[puzzle_emb_name].shape
        if ckpt_shape[0] != expected_shape[0]:
            print(f"Resizing puzzle_emb from {expected_shape[0]} to {ckpt_shape[0]}")
            from models.sparse_embedding import CastedSparseEmbedding
            inner_model.puzzle_emb = CastedSparseEmbedding(
                num_embeddings=ckpt_shape[0],
                embedding_dim=inner_model.config.puzzle_emb_ndim,
                batch_size=inner_model.config.batch_size,
                init_std=0.0,
                cast_to=getattr(torch, inner_model.config.forward_dtype)
            )
    model.load_state_dict(clean_state_dict, strict=False)
    print("Model loaded successfully!")

# Run profiling
inner = model.inner
seq_len = inner.config.seq_len
emb_len = inner.inner.puzzle_emb_len
hidden = inner.config.hidden_size
emb_dim = inner.inner.puzzle_emb.weights.shape[1]

wrapper_cpu = TRMBackboneStep(inner).eval().cpu().float()
x_cpu = torch.zeros(1, seq_len, dtype=torch.int64)
emb_cpu = torch.zeros(1, emb_dim, dtype=torch.float32)
zH_cpu = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32)
zL_cpu = torch.zeros(1, seq_len + emb_len, hidden, dtype=torch.float32)

with profile(activities=[ProfilerActivity.CPU],
             record_shapes=True,
             with_flops=True) as prof:
    with record_function("trm_step"):
        with torch.no_grad():
            wrapper_cpu(x_cpu, emb_cpu, zH_cpu, zL_cpu)

total_flops = sum(e.flops for e in prof.key_averages() if e.flops)
print(f"ACTUAL FLOPS per H-cycle step: {total_flops} FLOPS")
print(f"MFLOPs: {total_flops / 1e6:.3f} MFLOPs")
