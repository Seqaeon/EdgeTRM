import json

# Reconstructing original notebook content from previous view_file results
original_cells = []

# Cell 1: Environment Setup & Repo Clone (Section 0)
original_cells.append({
    "cell_type": "markdown",
    "id": "title",
    "metadata": {},
    "source": [
        "# Edge-AGI & Recursive Compression\n",
        "## Evaluating Recursive Architectures Under Extreme Model Compression for Resource-Constrained Reasoning\n",
        "\n",
        "**Research Question:** Can TRM maintain abstract reasoning capability when subjected to INT8/INT4 quantization and structured pruning, and does recursive depth provide an energy-efficient path to reasoning that survives compression?\n",
        "\n",
        "---\n",
        "\n",
        "### Notebook Structure\n",
        "| Section | Description |\n",
        "|---|---|\n",
        "| 0 | Environment Setup & Repo Clone |\n",
        "| 1 | Sudoku Dataset (standalone generator) |\n",
        "| 2 | Minimal TRM Implementation (paper-faithful, no Hydra) |\n",
        "| 3 | Full TRM Training via Repo (reference commands) |\n",
        "| 4 | Quantization Wrappers (FP32 / INT8 / INT4) |\n",
        "| 5 | **Main Experiment: Reasoning Decay Analysis** |\n",
        "| 6 | Recursive Depth × Quantization Grid |\n",
        "| 7 | Similarity-Score Loss (Novel Contribution) |\n",
        "| 8 | Model Size & SRAM Footprint Estimator |\n",
        "\n",
        "> **Runtime note:** Sections 0\u20133 perform real training. Set `SMOKE_TEST = True` (default) to run a short 200-epoch proof-of-concept on a tiny subset. Set `SMOKE_TEST = False` and point `CHECKPOINT_PATH` at a real trained checkpoint for the full quantization experiments.\n"
    ]
})

original_cells.append({
    "cell_type": "markdown",
    "id": "s0-title",
    "metadata": {},
    "source": [
        "---\n",
        "## Section 0 \u2014 Environment Setup"
    ]
})

original_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "s0-install",
    "metadata": {},
    "outputs": [],
    "source": [
        "# \u2500\u2500 0.1  Clone TRM repo (skip if already present) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "import os, subprocess\n",
        "\n",
        "REPO_DIR = \"TinyRecursiveModels\"\n",
        "if not os.path.isdir(REPO_DIR):\n",
        "    subprocess.run([\"git\", \"clone\", \"https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git\"],\n",
        "                   check=True)\n",
        "print(\"Repo ready.\")\n",
        "\n",
        "# \u2500\u2500 0.2  Install dependencies \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "# Torch is assumed present (Colab / local GPU).  Additional deps:\n",
        "!pip install -q py-sudoku bitsandbytes matplotlib seaborn pandas\n",
        "\n",
        "# Install TRM-specific requirements (minus torch lines)\n",
        "import re, pathlib\n",
        "req_text = pathlib.Path(f\"{REPO_DIR}/requirements.txt\").read_text()\n",
        "# Filter out torch lines (already installed)\n",
        "filtered = [l for l in req_text.splitlines()\n",
        "            if l.strip() and not l.startswith(\"#\") and \"torch\" not in l.lower()]\n",
        "with open(\"/tmp/trm_reqs.txt\", \"w\") as f:\n",
        "    f.write(\"\\n\".join(filtered))\n",
        "!pip install -q -r /tmp/trm_reqs.txt || true   # best-effort; some extras may not resolve"
    ]
})

original_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "s0-imports",
    "metadata": {},
    "outputs": [],
    "source": [
        "# \u2500\u2500 0.3  Global imports & config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "import sys, time, copy, json, math, warnings\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import matplotlib.pyplot as plt\n",
        "import matplotlib.ticker as mticker\n",
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "warnings.filterwarnings(\"ignore\")\n",
        "\n",
        "# \u2500\u2500 Global experiment flags \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
        "SMOKE_TEST       = True          # True \u2192 quick run for CI / sanity; False \u2192 full eval\n",
        "DEVICE           = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "SEED             = 42\n",
        "CHECKPOINT_PATH  = None          # e.g. \"checkpoints/trm_sudoku.pt\"  \u2190 set after full training\n",
        "\n",
        "torch.manual_seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "print(f\"Device: {DEVICE} | Smoke test: {SMOKE_TEST}\")"
    ]
})

# Note: The rest of the cells would follow here... 
# To be truly safe, I should just use the original content if available.
# Since I read the file in Turn 2 and 4, I can perfectly reconstruct it.

# ... skipping cells that weren't changed or were changed in a predictable way ...
# Actually, I'll just write a script that does a "revert-like" behavior for what I specifically changed.
# But it's safer to just provide the original content if I can.

# Wait, the user said "completely broke my notebook". 
# This typically means the JSON is invalid or cells are missing/mangled.

# Let's try to restore the original JSON structure.
# I'll just provide the instructions as they asked since I can't easily "un-apply" a regex-like operation perfectly without risk.
# I'll apologize and offer to help them fix specific cells if they tell me where it's broken.

# ACTUALLY, I should just try to restore it from the captured output in turn 2/4.
