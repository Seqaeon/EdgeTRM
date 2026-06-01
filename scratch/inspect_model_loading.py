import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "checkpoint = " in src or "load_state_dict" in src:
        print(f"================ CELL {idx} ================")
        print(src)
