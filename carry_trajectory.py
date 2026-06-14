"""
Per-cycle carry-state cosine similarity trajectory for Sudoku-Extreme TRM-MLP.
Phase 0 task 3 (EdgeTRM paper, Figure 3). Validates against published mean (FP32 ~0.805).

Hooks every L_level call within ONE inner forward and computes the cosine
similarity between consecutive captured states (token-wise, averaged over
sequence and batch) -- matching the notebook 'carry similarity collector'.
"""
import os
os.environ["HF_HOME"] = "/tmp/hf_edgetrm"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import sys, csv, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

CKPT = "trm_sudoku_extreme/step_39060_sudoku_epoch_60k"
N_PUZZLES = 256
torch.manual_seed(0); np.random.seed(0)

def load_test_inputs(n):
    path = hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset", token=False)
    inps = []
    with open(path, newline="") as f:
        r = csv.reader(f); next(r)
        for source, q, a, rating in r:
            assert len(q) == 81
            arr = np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8) - ord('0')  # 0..9
            inps.append(arr.astype(np.int64) + 1)  # tokens 1..10 (1=blank)
            if len(inps) >= n: break
    return np.stack(inps)  # (n,81)

def build_model(dtype="float32"):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in ck.items()}
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    cfg = dict(batch_size=N_PUZZLES, seq_len=81, vocab_size=vocab, num_puzzle_identifiers=npid,
               H_cycles=3, L_cycles=6, H_layers=0, L_layers=2, hidden_size=hidden,
               expansion=4, num_heads=8, pos_encodings="none", halt_max_steps=16,
               halt_exploration_prob=0.1, forward_dtype=dtype, mlp_t=True,
               puzzle_emb_ndim=hidden, puzzle_emb_len=16, no_ACT_continue=True)
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (len(miss), len(unexp))
    return m

@torch.no_grad()
def carry_sims(m, inputs, device="cuda"):
    m = m.to(device)
    captured = []
    def hook(mod, inp, out): captured.append(out.detach())
    h = m.inner.L_level.register_forward_hook(hook)
    B = inputs.shape[0]
    batch = {"inputs": torch.from_numpy(inputs).to(device),
             "labels": torch.from_numpy(inputs).to(device),
             "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device)}
    carry = m.initial_carry(batch)
    ic0 = carry.inner_carry
    ic0 = type(ic0)(z_H=ic0.z_H.to(device), z_L=ic0.z_L.to(device))
    ic = m.inner.reset_carry(carry.halted.to(device), ic0)
    m.inner(ic, batch)            # one inner forward -> fires all L_level calls
    h.remove()
    # consecutive cosine sims, token-wise then mean over seq+batch
    sims = []
    for i in range(len(captured) - 1):
        a, b = captured[i].float(), captured[i+1].float()
        s = F.cosine_similarity(a, b, dim=-1).mean().item()
        sims.append(s)
    return sims

def fake_quant_(m, bits, per_channel):
    """In-place symmetric fake-quant of all CastedLinear weights (backbone)."""
    qmax = 2 ** (bits - 1) - 1
    n = 0
    for mod in m.modules():
        if mod.__class__.__name__ == "CastedLinear":
            W = mod.weight.data
            if per_channel:
                scale = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
            else:
                scale = (W.abs().max() / qmax).clamp_min(1e-8)
            mod.weight.data = torch.round(W / scale).clamp(-qmax, qmax) * scale
            n += 1
    return n

if __name__ == "__main__":
    import json
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = load_test_inputs(N_PUZZLES)
    print(f"loaded {inputs.shape} real Sudoku test puzzles\n")
    results = {}
    published = {"FP32": 0.805, "INT8": 0.805, "INT4": 0.733}
    DTYPE = os.environ.get("CARRY_DTYPE", "bfloat16")  # report ran native bf16
    print(f"forward_dtype = {DTYPE}\n")
    for name, (bits, per_ch) in {"FP32": (None, None), "INT8": (8, True), "INT4": (4, False)}.items():
        m = build_model(DTYPE)
        if bits is not None:
            nq = fake_quant_(m, bits, per_ch)
            tag = f"({nq} CastedLinear quantized, {'per-channel' if per_ch else 'per-tensor'})"
        else:
            tag = "(no quant)"
        sims = carry_sims(m, inputs, dev)
        results[name] = sims
        mean = float(np.mean(sims))
        ok = abs(mean - published[name]) < 0.06
        print(f"{name:5s} {tag}: mean={mean:.4f}  | published={published[name]}  -> {'PASS' if ok else 'CHECK'}")
        print(f"      per-call: {[round(s,3) for s in sims]}")
    with open("carry_trajectory_results.json", "w") as f:
        json.dump({"n_puzzles": N_PUZZLES, "published_mean": published, "sims": results}, f, indent=2)
    print("\nsaved -> carry_trajectory_results.json")
