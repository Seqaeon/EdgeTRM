"""ARC-2 carry-fidelity: the INTERMEDIATE case (INT4 loses ~10pp, not catastrophic).
Detector should land BETWEEN Maze (0.99, fine) and Sudoku (0.35, broken) -> graded, not binary."""
import os, glob
os.environ["HF_HOME"]="/tmp/hf_edgetrm"
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"TinyRecursiveModels")
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from carry_diagnostics import fake_quant_, signals
from sudoku_calibrated import calib_quant_
np.random.seed(0); torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"; B=32
CK=glob.glob("/tmp/hf_edgetrm/hub/models--arcprize--trm_arc_prize_verification/snapshots/*/arc_v2_public/step_723914")[0]
D="/tmp/arc_data/arc2test-aug-128/test"

def load_batch():
    inp=np.load(D+"/all__inputs.npy"); pid=np.load(D+"/all__puzzle_identifiers.npy"); pidx=np.load(D+"/all__puzzle_indices.npy")
    persample=np.repeat(pid, np.diff(pidx).astype(np.int64))
    sel=np.where(persample!=0)[0][:B]                         # skip blank pid=0
    return inp[sel].astype(np.int64), persample[sel].astype(np.int64)

def build():
    sd=torch.load(CK,map_location="cpu",weights_only=False)
    if "model" in sd: sd=sd["model"]
    sd={k.replace("_orig_mod.model.","").replace("model.",""):v for k,v in sd.items()}
    cfg=dict(batch_size=B,seq_len=900,vocab_size=12,num_puzzle_identifiers=1191730,
             H_cycles=3,L_cycles=4,H_layers=0,L_layers=2,hidden_size=512,expansion=4,num_heads=8,
             pos_encodings="rope",halt_max_steps=16,halt_exploration_prob=0.1,forward_dtype="float32",
             mlp_t=False,puzzle_emb_ndim=512,puzzle_emb_len=16,no_ACT_continue=True)
    m=TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss,unexp=m.load_state_dict(sd,strict=False); assert len(miss)==0 and len(unexp)==0,(len(miss),len(unexp))
    return m

@torch.no_grad()
def capture(m, inp, pids):
    m=m.to(DEV); states=[]
    h=m.inner.L_level.register_forward_hook(lambda mo,i,o: states.append(o.detach().float().cpu()))
    batch={"inputs":torch.from_numpy(inp).to(DEV),"labels":torch.from_numpy(inp).to(DEV),
           "puzzle_identifiers":torch.from_numpy(pids).to(torch.int32).to(DEV)}
    c=m.initial_carry(batch); ic=c.inner_carry
    ic=type(ic)(z_H=ic.z_H.to(DEV),z_L=ic.z_L.to(DEV))
    m.inner(m.inner.reset_carry(c.halted.to(DEV),ic), batch); h.remove()
    del m; torch.cuda.empty_cache()
    return states

if __name__=="__main__":
    inp,pids=load_batch()
    print(f"ARC: {B} real test puzzles, seq=900, pids {pids.min()}..{pids.max()}\n")
    ref=capture(build(),inp,pids)
    print(f"{'variant':18s} {'fidelity_final':>14s}")
    print(f"{'FP32':18s} {1.0:14.4f}")
    out={"FP32":1.0}
    for name,q in [("INT8",lambda m:fake_quant_(m,8,True)),
                   ("INT4-naive",lambda m:fake_quant_(m,4,False)),
                   ("INT4-calibrated",lambda m:calib_quant_(m,4))]:
        m=build(); q(m)
        fid=signals(capture(m,inp,pids),ref=ref)["fidelity_final"]
        out[name]=fid; print(f"{name:18s} {fid:14.4f}")
    print(f"\nContext: Sudoku INT4-naive=0.35 (broken,5%) | Maze INT4=0.99 (fine,86%)")
    print(f"ARC INT4-naive should be INTERMEDIATE (ARC INT4 loses ~10pp, cell stays 85%)")
    import json; json.dump({"B":B,"fidelity_final":out},open("arc_fidelity_results.json","w"),indent=2)
