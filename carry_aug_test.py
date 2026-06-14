"""Reconciliation test: does INT4 carry drop to ~0.733 on AUGMENTED puzzles
(as the notebook's train_loader used) vs raw test puzzles?"""
import os
os.environ["HF_HOME"] = "/tmp/hf_edgetrm"; os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import sys, csv, numpy as np, torch
sys.path.insert(0, "TinyRecursiveModels")
from huggingface_hub import hf_hub_download
from carry_trajectory import build_model, carry_sims, fake_quant_

def shuffle_sudoku(board, solution):  # verbatim from dataset/build_sudoku_dataset.py
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])
    def apply(x):
        if transpose_flag: x = x.T
        return digit_map[x.flatten()[mapping].reshape(9, 9).copy()]
    return apply(board), apply(solution)

np.random.seed(0); torch.manual_seed(0)
N = 256

def load_raw_boards(n):
    path = hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset", token=False)
    qs, as_ = [], []
    with open(path, newline="") as f:
        r = csv.reader(f); next(r)
        for source, q, a, rating in r:
            qs.append((np.frombuffer(q.replace('.', '0').encode(), np.uint8) - ord('0')).reshape(9, 9))
            as_.append((np.frombuffer(a.encode(), np.uint8) - ord('0')).reshape(9, 9))
            if len(qs) >= n: break
    return qs, as_

def to_tokens(boards):  # digit 0..9 board -> tokens 1..10 (1=blank)
    return np.stack([b.reshape(-1).astype(np.int64) + 1 for b in boards])

qs, as_ = load_raw_boards(N)
raw_inp = to_tokens(qs)
aug_boards = [shuffle_sudoku(q, a)[0] for q, a in zip(qs, as_)]  # augmented input boards
aug_inp = to_tokens(aug_boards)
print(f"loaded {N} test puzzles; built raw + augmented token sets")

dev = "cuda" if torch.cuda.is_available() else "cpu"
for label, inp in [("RAW test", raw_inp), ("AUGMENTED test", aug_inp)]:
    fp = carry_sims(build_model("bfloat16"), inp, dev)
    m4 = build_model("bfloat16"); fake_quant_(m4, 4, False)
    i4 = carry_sims(m4, inp, dev)
    print(f"\n{label}:  FP32 mean={np.mean(fp):.4f}   INT4 mean={np.mean(i4):.4f}   (published FP32=0.805, INT4=0.733)")
