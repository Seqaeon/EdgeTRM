import json

try:
    with open("edge-trm-new.ipynb", "r") as f:
        nb = json.load(f)
    print(f"JSON validation successful! Total cells: {len(nb['cells'])}")
except Exception as e:
    print(f"JSON error: {e}")
