import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = "".join(cell['source'])
        first_line = src.split('\n')[0] if src else ""
        print(f"Cell {idx} [Markdown]: {first_line[:80]}")
    elif cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        first_line = src.split('\n')[0] if src else ""
        print(f"Cell {idx} [Code]: {first_line[:80]}")
