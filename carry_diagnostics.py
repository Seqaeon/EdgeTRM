"""
Option B: redesign the carry diagnostic into a real label-free detector with a
large margin. Searches several candidate signals on Sudoku (INT4 BREAKS: 69->5%)
and Maze (INT4 FINE: ~86%). A good detector flags Sudoku-INT4 strongly while NOT
false-alarming on Maze-INT4.

Captured signal per run = ordered list of L_level outputs within one inner forward.
"""
import os
os.environ["HF_HOME"] = "/tmp/hf_edgetrm"; os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import sys, csv, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MAZE_CHARSET = "# SGo"
TASKS = {
  "Sudoku": dict(ckpt="trm_sudoku_extreme/step_39060_sudoku_epoch_60k", repo="sapientinc/sudoku-extreme",
                 seq=81, mlp_t=True, pos="none", L_cycles=6, N=256),
  "Maze":   dict(ckpt="trm_maze_hard/model.pt", repo="sapientinc/maze-30x30-hard-1k",
                 seq=900, mlp_t=False, pos="rope", L_cycles=4, N=64),
}

def load_inputs(task, n):
    p = hf_hub_download(TASKS[task]["repo"], "test.csv", repo_type="dataset", token=False)
    out = []
    with open(p, newline="") as f:
        r = csv.reader(f); next(r)
        for source, q, a, rating in r:
            if task == "Sudoku":
                out.append((np.frombuffer(q.replace('.', '0').encode(), np.uint8) - ord('0')).astype(np.int64) + 1)
            else:
                out.append(np.array([MAZE_CHARSET.index(chr(c)) + 1 for c in q.encode()], dtype=np.int64))
            if len(out) >= n: break
    return np.stack(out)

def build(task, dtype="float32"):
    t = TASKS[task]
    ck = torch.load(t["ckpt"], map_location="cpu", weights_only=False)
    sd = {}
    for k, v in ck.items():
        for pre in ("_orig_mod.model.", "model."):
            if k.startswith(pre): k = k[len(pre):]; break
        sd[k] = v
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    cfg = dict(batch_size=t["N"], seq_len=t["seq"], vocab_size=vocab, num_puzzle_identifiers=npid,
               H_cycles=3, L_cycles=t["L_cycles"], H_layers=0, L_layers=2, hidden_size=hidden,
               expansion=4, num_heads=8, pos_encodings=t["pos"], halt_max_steps=16,
               halt_exploration_prob=0.1, forward_dtype=dtype, mlp_t=t["mlp_t"],
               puzzle_emb_ndim=hidden, puzzle_emb_len=16, no_ACT_continue=True)
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (task, len(miss), len(unexp))
    return m

def fake_quant_(m, bits, per_ch):
    qmax = 2 ** (bits - 1) - 1
    for mod in m.modules():
        if mod.__class__.__name__ == "CastedLinear":
            W = mod.weight.data
            s = (W.abs().amax(dim=1, keepdim=True) if per_ch else W.abs().max()) / qmax
            mod.weight.data = torch.round(W / s.clamp_min(1e-8)).clamp(-qmax, qmax) * s.clamp_min(1e-8)

@torch.no_grad()
def capture(m, inputs):
    m = m.to(DEV); states = []
    h = m.inner.L_level.register_forward_hook(lambda mod, i, o: states.append(o.detach().float().cpu()))
    B = inputs.shape[0]
    batch = {"inputs": torch.from_numpy(inputs).to(DEV),
             "labels": torch.from_numpy(inputs).to(DEV),
             "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
    c = m.initial_carry(batch); ic0 = c.inner_carry
    ic0 = type(ic0)(z_H=ic0.z_H.to(DEV), z_L=ic0.z_L.to(DEV))
    ic = m.inner.reset_carry(c.halted.to(DEV), ic0)
    m.inner(ic, batch); h.remove()
    return states  # list of (B,T,D)

def signals(states, ref=None):
    consec = [F.cosine_similarity(states[i], states[i-1], dim=-1).mean().item() for i in range(1, len(states))]
    norms = [s.norm(dim=-1).mean().item() for s in states]
    out = {
      "consec_mean":  float(np.mean(consec)),                 # current paper signal
      "consec_final": consec[-1],                             # last transition
      "rough":        float(np.std(consec)),                  # trajectory roughness
      "norm_ratio":   norms[-1] / (norms[0] + 1e-8),          # carry norm growth/decay
      "final_vs_init": F.cosine_similarity(states[-1], states[0], dim=-1).mean().item(),
    }
    if ref is not None:  # cross-precision fidelity vs FP32 trajectory (label-free; FP32 avail pre-deploy)
        fid = [F.cosine_similarity(states[i], ref[i], dim=-1).mean().item() for i in range(len(states))]
        out["fidelity_mean"]  = float(np.mean(fid))
        out["fidelity_final"] = fid[-1]
        out["fidelity_min"]   = float(np.min(fid))
    return out

if __name__ == "__main__":
    results = {}
    for task in TASKS:
        inp = load_inputs(task, TASKS[task]["N"])
        print(f"\n===== {task}  ({inp.shape[0]} puzzles, seq={inp.shape[1]}) =====")
        ref_states = capture(build(task, "float32"), inp)  # FP32 reference
        results[task] = {}
        for name, (bits, per) in {"FP32": (None, None), "INT8": (8, True), "INT4": (4, False)}.items():
            m = build(task, "float32")
            if bits: fake_quant_(m, bits, per)
            st = capture(m, inp)
            results[task][name] = signals(st, ref=ref_states)
        # print table
        sig_names = list(results[task]["INT4"].keys())
        print(f"{'signal':14s} {'FP32':>9s} {'INT8':>9s} {'INT4':>9s} {'INT4-INT8':>10s}")
        for s in sig_names:
            fp = results[task]["FP32"].get(s, float('nan'))
            i8 = results[task]["INT8"].get(s, float('nan'))
            i4 = results[task]["INT4"].get(s, float('nan'))
            print(f"{s:14s} {fp:9.4f} {i8:9.4f} {i4:9.4f} {i4-i8:10.4f}")

    # Detector verdict: want |INT4-INT8| LARGE on Sudoku, SMALL on Maze
    print("\n===== DETECTOR SEARCH: |INT4 - INT8| separation (Sudoku should be >> Maze) =====")
    print(f"{'signal':14s} {'Sudoku sep':>11s} {'Maze sep':>10s} {'ratio S/M':>10s}")
    for s in results["Sudoku"]["INT4"].keys():
        ss = abs(results["Sudoku"]["INT4"].get(s,0) - results["Sudoku"]["INT8"].get(s,0))
        ms = abs(results["Maze"]["INT4"].get(s,0) - results["Maze"]["INT8"].get(s,0))
        ratio = ss / (ms + 1e-6)
        print(f"{s:14s} {ss:11.4f} {ms:10.4f} {ratio:10.1f}")
    import json; json.dump(results, open("carry_diagnostics_results.json","w"), indent=2)
    print("\nsaved -> carry_diagnostics_results.json")
