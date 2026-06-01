import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx in range(21):
    cell = nb['cells'][idx]
    src = "".join(cell['source'])
    print(f"================ CELL {idx} ({cell['cell_type']}) ================")
    print(src[:500])
