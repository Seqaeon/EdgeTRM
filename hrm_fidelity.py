"""Second recursive architecture: HRM (Hierarchical Reasoning Model) on Sudoku-Extreme.
Does the compositional collapse + carry-fidelity detector + calibration recovery
GENERALIZE beyond TRM? Same harness, same task, different recursion (H_level+L_level,
H_layers=L_layers=4, H_cycles=L_cycles=2)."""
import os, glob
os.environ["HF_HOME"]="/tmp/hf_edgetrm"; os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
import sys, csv, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from models.recursive_reasoning.hrm import HierarchicalReasoningModel_ACTV1 as HRM
from carry_diagnostics import fake_quant_, signals
from sudoku_calibrated import calib_quant_
np.random.seed(0); torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"; N=256
CK=glob.glob("/tmp/hf_edgetrm/hub/models--sapientinc--HRM-checkpoint-sudoku-extreme/snapshots/*/checkpoint")[0]

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
    sd={k.replace("_orig_mod.model.","").replace("model.",""):v for k,v in sd.items()}
    npid=sd["inner.puzzle_emb.weights"].shape[0]
    cfg=dict(batch_size=N,seq_len=81,vocab_size=11,num_puzzle_identifiers=npid,
             H_cycles=2,L_cycles=2,H_layers=4,L_layers=4,hidden_size=512,expansion=4,
             num_heads=8,pos_encodings="rope",halt_max_steps=16,halt_exploration_prob=0.1,
             forward_dtype=dtype,mlp_t=False,puzzle_emb_ndim=512)
    m=HRM(cfg).eval()
    miss,unexp=m.load_state_dict(sd,strict=False); assert len(miss)==0 and len(unexp)==0,(len(miss),len(unexp))
    return m

@torch.no_grad()
def capture(m, inp):
    m=m.to(DEV); states=[]
    h=m.inner.H_level.register_forward_hook(lambda mo,i,o: states.append(o.detach().float().cpu()))
    B=inp.shape[0]
    batch={"inputs":torch.from_numpy(inp).to(DEV),"labels":torch.from_numpy(inp).to(DEV),
           "puzzle_identifiers":torch.zeros(B,dtype=torch.int32,device=DEV)}
    c=m.initial_carry(batch); ic=c.inner_carry
    ic=type(ic)(z_H=ic.z_H.to(DEV),z_L=ic.z_L.to(DEV))
    m.inner(m.inner.reset_carry(c.halted.to(DEV),ic),batch); h.remove()
    del m; torch.cuda.empty_cache(); return states

@torch.no_grad()
def accuracy(m, inp, lab):
    from models.recursive_reasoning.hrm import (HierarchicalReasoningModel_ACTV1Carry as C,
        HierarchicalReasoningModel_ACTV1InnerCarry as IC)
    m=m.to(DEV); B=inp.shape[0]
    batch={"inputs":torch.from_numpy(inp).to(DEV),"labels":torch.from_numpy(lab).to(DEV),
           "puzzle_identifiers":torch.zeros(B,dtype=torch.int32,device=DEV)}
    c=m.initial_carry(batch); ic=c.inner_carry
    carry=C(inner_carry=IC(z_H=ic.z_H.to(DEV),z_L=ic.z_L.to(DEV)),steps=c.steps.to(DEV),
            halted=c.halted.to(DEV),current_data={k:v.to(DEV) for k,v in c.current_data.items()})
    out=None
    for _ in range(16):
        carry,out=m(carry,batch)
        if carry.halted.all(): break
    preds=out["logits"].argmax(-1); L=batch["labels"]; mask=L!=0; corr=(preds==L)&mask
    return ((corr.sum(-1)==mask.sum(-1)).float().mean().item(), corr.sum().item()/mask.sum().item())

if __name__=="__main__":
    inp,lab=load_qa(N)
    print(f"HRM on {N} Sudoku-Extreme test puzzles (2nd recursive architecture)\n")
    ref=capture(build("float32"),inp)
    print(f"{'variant':16s} {'PExact':>8s} {'Cell':>7s} {'fidelity':>9s}")
    rows=[]
    for name,q in [("FP32",None),("INT8",lambda m:fake_quant_(m,8,True)),
                   ("INT4-naive",lambda m:fake_quant_(m,4,False)),
                   ("INT4-calib",lambda m:calib_quant_(m,4))]:
        m=build("float32")
        if q: q(m)
        pex,cell=accuracy(m,inp,lab)
        fid=signals(capture(m,inp),ref=ref)["fidelity_final"]
        rows.append((name,pex,cell,fid)); print(f"{name:16s} {pex*100:7.2f}% {cell*100:6.2f}% {fid:9.4f}")
    import json; json.dump([{"variant":r[0],"pexact":r[1],"cell":r[2],"fidelity":r[3]} for r in rows],
                            open("hrm_fidelity_results.json","w"),indent=2)
    print("\nTRM-Sudoku for comparison: FP32 73.8/INT4-naive 10.2 (fid 0.35)/INT4-calib 71.9 (fid 0.89)")
