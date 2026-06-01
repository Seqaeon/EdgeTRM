import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

print("".join(nb['cells'][52]['source'])[:3000])
