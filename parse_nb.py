import json
import ast

with open("trm_quantization_experiments.ipynb", "r") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        # Some cells have shell commands like `!pip install ...`
        code_lines = []
        for line in code.split("\n"):
            if not line.strip().startswith("!"):
                code_lines.append(line)
            else:
                code_lines.append("#" + line)
        try:
            ast.parse("\n".join(code_lines))
        except SyntaxError as e:
            print(f"Syntax error in cell {i}: {e}")

print("Syntax check completed.")
