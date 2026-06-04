import json

with open("edge-trm.ipynb", "r") as f:
    nb = json.load(f)

cell = nb['cells'][4]
src = "".join(cell['source'])
for line in src.split("\n"):
    if "def evaluate_arc_per_puzzle" in line:
        print(line)
