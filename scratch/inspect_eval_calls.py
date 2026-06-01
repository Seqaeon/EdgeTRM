import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "evaluate_arc" in src or "evaluate_arc_per_puzzle" in src:
        print(f"Cell {idx} calls evaluation:")
        lines = src.split('\n')
        for line in lines:
            if "evaluate_arc" in line:
                print(f"  {line[:120]}")
