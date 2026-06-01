import json

with open("edge-trm-new.ipynb", "r") as f:
    nb = json.load(f)

print("Original Cell 14 source:")
print("".join(nb['cells'][14]['source'])[:300])

# Replace Cell 14 with a clean placeholder
nb['cells'][14] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Note: `evaluate_arc` has been consolidated and moved to Section 3 for global accessibility."
    ]
}

with open("edge-trm-new.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Cell 14 successfully updated with placeholder!")
