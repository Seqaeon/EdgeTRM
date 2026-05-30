import numpy as np
import os
import json

print("--- Checking dataset pids ---")
for split in ["test", "train"]:
    p_path = f"data1/arc2test-aug-128/{split}/all__puzzle_identifiers.npy"
    if os.path.exists(p_path):
        pids = np.load(p_path)
        print(f"{split} pids shape: {pids.shape}")
        print(f"{split} pids min: {pids.min()}, max: {pids.max()}")
        print(f"First 10 {split} pids: {pids[:10]}")

print("\n--- Checking identifiers.json ---")
with open("data1/arc2test-aug-128/identifiers.json") as f:
    ids = json.load(f)
print(f"Number of identifiers in identifiers.json: {len(ids)}")
