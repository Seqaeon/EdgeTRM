import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx in [21, 22, 23, 24, 25]:
    cell = nb['cells'][idx]
    src = "".join(cell['source'])
    print(f"================ CELL {idx} ================")
    print(src[:2000])
