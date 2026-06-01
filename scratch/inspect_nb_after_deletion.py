import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

# Delete cells 9 to 19 (11 cells total)
# Note: list slicing/deletion
del nb['cells'][9:20]

print(f"Number of cells after deletion: {len(nb['cells'])}")
for idx in range(25):
    cell = nb['cells'][idx]
    src = "".join(cell['source'])
    print(f"Cell {idx} [{cell['cell_type']}]: {src[:80].replace('\n', ' ')}")
