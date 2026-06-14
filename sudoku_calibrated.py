"""Does calibrated INT4 recover BOTH accuracy and carry-fidelity? (detector tracks recovery?)
Computes Sudoku puzzle-exact/cell accuracy AND fidelity-vs-FP32 for FP32/INT8/INT4-naive/INT4-calibrated."""
import os
os.environ["HF_HOME"]="/tmp/hf_edgetrm"; os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
import sys, csv, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from carry_diagnostics import build, fake_quant_, capture, signals
from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1Carry as Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry as IC)
np.random.seed(0); torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"; N=256

def load_qa(n):
    p=hf_hub_download("sapientinc/sudoku-extreme","test.csv",repo_type="dataset",token=False)
    q,a=[],[]
    with open(p,newline="") as f:
        r=csv.reader(f); next(r)
        for source,qq,aa,rating in r:
            q.append((np.frombuffer(qq.replace('.','0').encode(),np.uint8)-ord('0')).astype(np.int64)+1)
            a.append((np.frombuffer(aa.encode(),np.uint8)-ord('0')).astype(np.int64)+1)
            if len(q)>=n: break
    return np.stack(q), np.stack(a)

def calib_quant_(m, bits=4):  # per-channel ASYMMETRIC (report's "calibrated INT4")
    qmax=2**bits-1
    for mod in m.modules():
        if mod.__class__.__name__=="CastedLinear":
            W=mod.weight.data
            wmin=W.amin(dim=1,keepdim=True); wmax=W.amax(dim=1,keepdim=True)
            scale=((wmax-wmin)/qmax).clamp_min(1e-8); zp=torch.round(-wmin/scale)
            mod.weight.data=(torch.clamp(torch.round(W/scale)+zp,0,qmax)-zp)*scale

@torch.no_grad()
def accuracy(m, inp, lab):
    m=m.to(DEV); B=inp.shape[0]
    batch={"inputs":torch.from_numpy(inp).to(DEV),"labels":torch.from_numpy(lab).to(DEV),
           "puzzle_identifiers":torch.zeros(B,dtype=torch.int32,device=DEV)}
    c=m.initial_carry(batch); ic=c.inner_carry
    carry=Carry(inner_carry=IC(z_H=ic.z_H.to(DEV),z_L=ic.z_L.to(DEV)),
                steps=c.steps.to(DEV),halted=c.halted.to(DEV),
                current_data={k:v.to(DEV) for k,v in c.current_data.items()})
    out=None
    for _ in range(16):
        carry,out=m(carry,batch)
        if carry.halted.all(): break
    preds=out["logits"].argmax(-1); L=batch["labels"]
    mask=L!=0; corr=(preds==L)&mask
    cell=corr.sum().item()/mask.sum().item()
    pex=((corr.sum(-1)==mask.sum(-1)).float().mean()).item()
    return pex, cell

if __name__=="__main__":
    inp,lab=load_qa(N)
    print(f"{N} Sudoku test puzzles (q+a)\n")
    ref=capture(build("Sudoku","float32"), inp)
    rows=[]
    for name,quant in [("FP32",None),("INT8",lambda m:fake_quant_(m,8,True)),
                       ("INT4-naive",lambda m:fake_quant_(m,4,False)),
                       ("INT4-calibrated",lambda m:calib_quant_(m,4))]:
        m=build("Sudoku","float32")
        if quant: quant(m)
        pex,cell=accuracy(m,inp,lab)
        fid=signals(capture(m,inp),ref=ref)["fidelity_final"]
        rows.append((name,pex,cell,fid))
        print(f"{name:16s} | PuzzleExact={pex*100:5.2f}%  Cell={cell*100:5.2f}%  carry-fidelity={fid:.4f}")
    print("\nPublished: FP32 69.10/87.47, INT4-naive 5.30/66.02")
    print("Detector check: does fidelity track accuracy across all 4 variants?")
    import json; json.dump({"N":N,"rows":[{"variant":r[0],"pexact":r[1],"cell":r[2],"fidelity_final":r[3]} for r in rows]},
                            open("sudoku_calibrated_results.json","w"),indent=2)
