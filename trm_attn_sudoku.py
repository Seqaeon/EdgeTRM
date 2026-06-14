"""Within-family ablation: TRM-ATTENTION on Sudoku (same model family + task as the
collapsing TRM-MLP-Mixing, only attention vs MLP-mixing swapped). If this survives INT4
while TRM-MLP collapses, the fragility is ARCHITECTURAL, not task-intrinsic."""
import os; os.environ["HF_HOME"]="/tmp/hf_edgetrm"; os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
import sys, csv, numpy as np, torch
sys.path.insert(0,"TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as TRM
from carry_diagnostics import fake_quant_, signals, capture
from sudoku_calibrated import calib_quant_, accuracy
np.random.seed(0); torch.manual_seed(0)
N=256; CK="trm_sudoku_extreme/step_39060_sudoku_60k_epoch_attn_type"

def load_qa(n):
    p=hf_hub_download("sapientinc/sudoku-extreme","test.csv",repo_type="dataset",token=False)
    q,a=[],[]
    with open(p,newline="") as f:
        r=csv.reader(f); next(r)
        for s,qq,aa,rt in r:
            q.append((np.frombuffer(qq.replace('.','0').encode(),np.uint8)-ord('0')).astype(np.int64)+1)
            a.append((np.frombuffer(aa.encode(),np.uint8)-ord('0')).astype(np.int64)+1)
            if len(q)>=n: break
    return np.stack(q), np.stack(a)

def build(dtype="float32"):
    sd=torch.load(CK,map_location="cpu",weights_only=False)
    if isinstance(sd,dict) and "model" in sd: sd=sd["model"]
    sd={k.replace("_orig_mod.model.","").replace("model.",""):v for k,v in sd.items()}
    cfg=dict(batch_size=N,seq_len=81,vocab_size=11,num_puzzle_identifiers=sd["inner.puzzle_emb.weights"].shape[0],
             H_cycles=3,L_cycles=6,H_layers=0,L_layers=2,hidden_size=512,expansion=4,num_heads=8,
             pos_encodings="rope",halt_max_steps=16,halt_exploration_prob=0.1,forward_dtype=dtype,
             mlp_t=False,puzzle_emb_ndim=512,puzzle_emb_len=16,no_ACT_continue=True)
    m=TRM(cfg).eval(); miss,unexp=m.load_state_dict(sd,strict=False)
    assert len(miss)==0 and len(unexp)==0,(len(miss),len(unexp)); return m

if __name__=="__main__":
    inp,lab=load_qa(N)
    print(f"TRM-ATTENTION on {N} Sudoku-Extreme puzzles (within-family ablation)\n")
    ref=capture(build("float32"),inp)
    print(f"{'variant':16s} {'PExact':>8s} {'Cell':>7s} {'fidelity':>9s}")
    rows=[]
    for name,q in [("FP32",None),("INT8",lambda m:fake_quant_(m,8,True)),
                   ("INT4-naive",lambda m:fake_quant_(m,4,False)),("INT4-calib",lambda m:calib_quant_(m,4))]:
        m=build("float32")
        if q: q(m)
        pex,cell=accuracy(m,inp,lab); fid=signals(capture(m,inp),ref=ref)["fidelity_final"]
        rows.append((name,pex,cell,fid)); print(f"{name:16s} {pex*100:7.2f}% {cell*100:6.2f}% {fid:9.4f}")
    import json; json.dump([{"variant":r[0],"pexact":r[1],"cell":r[2],"fidelity":r[3]} for r in rows],
                            open("trm_attn_sudoku_results.json","w"),indent=2)
    print("\nCompare — TRM-MLP-Sudoku: INT4-naive 10.2% (fid 0.35) | HRM-Sudoku(attn): INT4-naive 48.4% (fid 0.98)")
