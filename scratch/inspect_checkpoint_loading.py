import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "load_arc_model" in src or "torch.load" in src:
        print(f"================ CELL {idx} ================")
        print(src[:2000])
