import json

with open("trm_quantization_experiments.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    new_source = []
    for line in source:
        # Section 0
        if line == 'DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n':
            new_source.append('NUM_GPUS         = torch.cuda.device_count()\n')
            new_source.append(line)
        elif line == 'print(f"Device: {DEVICE} | Smoke test: {SMOKE_TEST}")':
            new_source.append('print(f"Device: {DEVICE} ({NUM_GPUS} GPUs available) | Smoke test: {SMOKE_TEST}")')
        elif 'CHECKPOINT_PATH  = None' in line:
            new_source.append(line)
            new_source.append('\n')
            new_source.append('def get_model(m):\n')
            new_source.append('    return m.module if isinstance(m, nn.DataParallel) else m\n')

        # Section 2.1
        elif line == 'trm = TinyRecursiveModel(hidden_size=H).to(DEVICE)\n':
            new_source.append('trm = TinyRecursiveModel(hidden_size=H)\n')
            new_source.append('if NUM_GPUS > 1:\n')
            new_source.append('    trm = nn.DataParallel(trm)\n')
            new_source.append('trm = trm.to(DEVICE)\n')
        elif line == 'n_p = count_params(trm)\n':
            new_source.append('n_p = count_params(get_model(trm))\n')

        # Section 2.3
        elif line == '            y_carry = torch.zeros(B, model.seq_len, dtype=torch.long, device=DEVICE)\n':
            new_source.append('            y_carry = torch.zeros(B, get_model(model).seq_len, dtype=torch.long, device=DEVICE)\n')
        elif line == '            z_carry = model.init_carry(B).to(DEVICE)\n':
            new_source.append('            z_carry = get_model(model).init_carry(B).to(DEVICE)\n')
        elif 'ce_loss = F.cross_entropy(logits.reshape(-1, model.vocab_size)' in line:
            new_source.append('                ce_loss = F.cross_entropy(logits.reshape(-1, get_model(model).vocab_size),\n')
        elif line == '        total_cells    += B * model.seq_len\n':
            new_source.append('        total_cells    += B * get_model(model).seq_len\n')

        # Section 5.1
        elif line == '        y_carry = torch.zeros(B, trm_ref.seq_len, dtype=torch.long)\n':
             new_source.append('        y_carry = torch.zeros(B, get_model(trm_ref).seq_len, dtype=torch.long)\n')
        elif line == '        z_carry = trm_ref.z_init.expand(B, -1, -1).clone().cpu()\n':
             new_source.append('        z_carry = get_model(trm_ref).z_init.expand(B, -1, -1).clone().cpu()\n')
        elif line == '        total_cells    += B * trm_ref.seq_len\n':
             new_source.append('        total_cells    += B * get_model(trm_ref).seq_len\n')

        # Section 6.1
        elif line == '            x_emb = model.x_embed(x_batch)\n':
             new_source.append('            x_emb = get_model(model).x_embed(x_batch)\n')
        elif line == '            y_emb = model.y_embed(y_carry)\n':
             new_source.append('            y_emb = get_model(model).y_embed(y_carry)\n')
        elif line == '            for _ in range(model.T_cycles - 1):\n':
             new_source.append('            for _ in range(get_model(model).T_cycles - 1):\n')
        elif line == '                    z_carry = model._latent_step(x_emb, y_emb, z_carry)\n':
             new_source.append('                    z_carry = get_model(model)._latent_step(x_emb, y_emb, z_carry)\n')
        elif line == '                y_emb = model._answer_step(y_emb, z_carry)\n':
             new_source.append('                y_emb = get_model(model)._answer_step(y_emb, z_carry)\n')
        elif line == '            logits = model.lm_head(y_emb)\n':
             new_source.append('            logits = get_model(model).lm_head(y_emb)\n')
        elif line == '            q      = model.halt_prob(y_emb)\n':
             new_source.append('            q      = get_model(model).halt_prob(y_emb)\n')

        # Section 7.1
        elif 'ce_loss  = F.cross_entropy(logits.reshape(-1, model.vocab_size)' in line:
             new_source.append('                ce_loss  = F.cross_entropy(logits.reshape(-1, get_model(model).vocab_size),\n')
        elif line == '                        for _ in range(model.n_cycles):\n':
             new_source.append('                        for _ in range(get_model(model).n_cycles):\n')
        elif line == '                for _ in range(model.n_cycles):\n':
             new_source.append('                for _ in range(get_model(model).n_cycles):\n')
        
        # Section 7.2
        elif line == 'trm_sim = TinyRecursiveModel(hidden_size=H).to(DEVICE)\n':
             new_source.append('trm_sim = TinyRecursiveModel(hidden_size=H)\n')
             new_source.append('if NUM_GPUS > 1:\n')
             new_source.append('    trm_sim = nn.DataParallel(trm_sim)\n')
             new_source.append('trm_sim = trm_sim.to(DEVICE)\n')

        # Section 8.1
        elif 'activations = 2 * model.seq_len * model.hidden_size' in line:
             new_source.append('    activations = 2 * get_model(model).seq_len * get_model(model).hidden_size * batch_size * bytes_per_val\n')

        else:
            new_source.append(line)
    cell["source"] = new_source

with open("trm_quantization_experiments.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
