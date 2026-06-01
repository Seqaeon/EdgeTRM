import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

print(nb['cells'][6]['source'])
