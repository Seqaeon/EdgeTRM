import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

for idx in range(35):
    cell = nb['cells'][idx]
    src = "".join(cell['source'])
    first_line = src.split('\n')[0] if src else ""
    print(f"Cell {idx} [{cell['cell_type']}]: {first_line[:80]}")
