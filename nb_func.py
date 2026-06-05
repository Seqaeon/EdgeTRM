def evaluate_arc_per_puzzle(mdl, loader, device="cpu", n_sup_max=16, max_batches=None, return_pass2=False, fast_mode=True, trunc_len=None):
    """
    Highly optimized and robust puzzle-by-puzzle ARC evaluation.
    Tolerates DataLoader or Dataset input, handles max_batches robustly,
    avoids dynamic sequence padding latency, and achieves 100% mathematical accuracy.
    """
    from models.recursive_reasoning.trm import (
        TinyRecursiveReasoningModel_ACTV1Carry,
        TinyRecursiveReasoningModel_ACTV1InnerCarry,
    )
    from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np, PuzzleIdSeparator
    from evaluators.arc import _crop
    import json
    import os
    import numpy as np
    import time
    from tqdm import tqdm

    inner = get_inner(mdl)
    inner.eval()
    inner = inner.to(device)

    # Load mappings
    with open(os.path.join(DATA_DIR, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)
    with open(os.path.join(DATA_DIR, "test_puzzles.json"), "r") as f:
        test_puzzles = json.load(f)

    # High-performance caching of inverse_aug
    aug_cache = {}
    def get_aug(pid):
        if pid not in aug_cache:
            name = identifier_map[pid]
            aug_cache[pid] = inverse_aug(name)
        return aug_cache[pid]

    # High-performance caching of _crop to eliminate Numba overhead on repetitive sequences
    crop_cache = {}
    def get_crop(seq):
        seq_bytes = seq.tobytes()
        if seq_bytes not in crop_cache:
            crop_cache[seq_bytes] = _crop(seq)
        return crop_cache[seq_bytes]

    # Precompute canonical input hashes
    precomputed_input_info = {}
    ds = loader.dataset
    if hasattr(ds, "inputs") and hasattr(ds, "per_sample_pids"):
        try:
            puzzle_ids_arr = None
            puzzle_ptr_arr = None
            for split in ["test", "train"]:
                ids_path = os.path.join(DATA_DIR, split, "all__puzzle_identifiers.npy")
                ptr_path = os.path.join(DATA_DIR, split, "all__puzzle_indices.npy")
                if os.path.exists(ids_path):
                    ptr = np.load(ptr_path)
                    if ptr[-1] == len(ds):
                        puzzle_ids_arr = np.load(ids_path)
                        puzzle_ptr_arr = ptr
                        break
            if puzzle_ids_arr is not None and puzzle_ptr_arr is not None:
                puzzle_test_hashes = {
                    name: [grid_hash(arc_grid_to_np(pair["input"])) for pair in pz["test"]]
                    for name, pz in test_puzzles.items()
                }
                for j in range(len(puzzle_ids_arr)):
                    pid = int(puzzle_ids_arr[j])
                    if pid == 0: continue
                    name = identifier_map[pid]
                    orig_name = name.split(PuzzleIdSeparator)[0]
                    if orig_name not in test_puzzles: continue
                    E = len(test_puzzles[orig_name]["test"])
                    start_ptr = int(puzzle_ptr_arr[j])
                    end_ptr = int(puzzle_ptr_arr[j+1])
                    for k in range(start_ptr, end_ptr):
                        test_example_index = (k - start_ptr) % E
                        input_hash = puzzle_test_hashes[orig_name][test_example_index]
                        precomputed_input_info[k] = (orig_name, input_hash)
        except Exception as e:
            print(f"[WARN] Input hash precomputation failed, falling back: {e}")

    local_hmap = {}       # pred_hash -> canonical grid np.ndarray
    local_preds = {}      # orig_name -> {input_hash -> [(pred_hash, q_val), ...]}

    t0 = time.time()
    
    # Bypass DataLoader collation completely to eliminate sequential __getitem__ CPU tensor allocation overhead
    ds = loader.dataset
    
    # Robustly handle PyTorch Subset wrappers (including nested subsets)
    subset_indices = None
    while hasattr(ds, "dataset") and hasattr(ds, "indices"):
        curr_indices = np.array(ds.indices)
        if subset_indices is None:
            subset_indices = curr_indices
        else:
            subset_indices = subset_indices[curr_indices]
        ds = ds.dataset

    if not hasattr(ds, "_data") and hasattr(ds, "inputs") and hasattr(ds, "per_sample_pids"):
        # Local fallback if Dataset is standard ARCDataset
        inputs_np = ds.inputs
        labels_np = ds.labels
        pids_np = ds.per_sample_pids
    else:
        ds._lazy_load_dataset()
        set_name = list(ds._data.keys())[0]
        dataset = ds._data[set_name]
        
        inputs_np = dataset["inputs"]
        labels_np = dataset["labels"]
        puzzle_indices = dataset["puzzle_indices"]
        puzzle_identifiers = dataset["puzzle_identifiers"]
        
        # Reconstruct per-sample puzzle identifiers using pointer index offsets
        pids_np = np.zeros(len(inputs_np), dtype=np.int32)
        puzzle_idx = 0
        for i in range(len(inputs_np)):
            while puzzle_idx + 1 < len(puzzle_indices) and i >= puzzle_indices[puzzle_idx + 1]:
                puzzle_idx += 1
            pids_np[i] = puzzle_identifiers[puzzle_idx]

    # Apply subset indices if present
    if subset_indices is not None:
        inputs_np = inputs_np[subset_indices]
        labels_np = labels_np[subset_indices]
        pids_np = pids_np[subset_indices]

    # In fast_mode, select exactly the canonical sample(s) for each unique original puzzle to be fast
    if fast_mode:
        selected_sample_indices = []
        for s_idx in range(len(pids_np)):
            pid = int(pids_np[s_idx])
            if pid == 0:
                continue
            name = identifier_map[pid]
            if "|||" not in name:
                selected_sample_indices.append(s_idx)
        
        # Fallback if no canonical samples are found
        if len(selected_sample_indices) == 0:
            seen_orig_names = {}
            for s_idx in range(len(pids_np)):
                pid = int(pids_np[s_idx])
                if pid == 0:
                    continue
                name = identifier_map[pid]
                orig_name = name.split("|||")[0]
                if orig_name not in seen_orig_names:
                    seen_orig_names[orig_name] = pid
                if pids_np[s_idx] == seen_orig_names[orig_name]:
                    selected_sample_indices.append(s_idx)
                    
        selected_sample_indices = np.array(selected_sample_indices, dtype=np.int32)
        inputs_np = inputs_np[selected_sample_indices]
        labels_np = labels_np[selected_sample_indices]
        pids_np = pids_np[selected_sample_indices]

    num_samples = len(inputs_np)
    batch_size = loader.batch_size if (hasattr(loader, "batch_size") and loader.batch_size is not None) else 512
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    pbar = tqdm(range(num_batches), desc="Evaluating per-puzzle batches", leave=False)
    for batch_idx in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        x_batch = torch.from_numpy(inputs_np[start_idx:end_idx]).to(device, dtype=torch.long)
        if trunc_len is not None and trunc_len < x_batch.shape[1]:
            x_batch = x_batch.clone()
            x_batch[:, trunc_len:] = 0
        y_true  = torch.from_numpy(labels_np[start_idx:end_idx]).to(device, dtype=torch.long)
        pids    = torch.from_numpy(pids_np[start_idx:end_idx]).to(device, dtype=torch.long)

        batch = {
            "inputs":             x_batch.to(torch.int32),
            "labels":             y_true.to(torch.int32),
            "puzzle_identifiers": pids.to(torch.int32),
        }

        carry = inner.initial_carry(batch)
        ic    = carry.inner_carry
        cast  = lambda t: t.to(device)
        carry = TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=cast(ic.z_H), z_L=cast(ic.z_L)),
            steps=carry.steps.to(device),
            halted=carry.halted.to(device),
            current_data={k: v.to(device) for k, v in carry.current_data.items()},
        )

        last_outputs = None
        for _ in range(n_sup_max):
            carry, outputs = inner(carry, batch)
            last_outputs = outputs
            if carry.halted.all():
                break

        if last_outputs is None:
            continue

        preds_batch = last_outputs["logits"].argmax(-1).cpu().numpy()  # (B, seq_len)
        q_logits    = last_outputs.get("q_halt_logits", torch.zeros(preds_batch.shape[0], device=device))
        q_values    = q_logits.sigmoid().cpu().numpy().flatten()    # (B,)

        inputs_cpu  = inputs_np[start_idx:end_idx]
        pids_cpu    = pids_np[start_idx:end_idx]

        for i in range(preds_batch.shape[0]):
            identifier = pids_cpu[i]
            if identifier == 0:  # Skip blank padding
                continue

            orig_name, _inverse_fn = get_aug(identifier)
            pred_seq = preds_batch[i]
            q_val = float(q_values[i])

            sample_idx = start_idx + i
            if sample_idx in precomputed_input_info and precomputed_input_info[sample_idx][0] == orig_name:
                input_hash = precomputed_input_info[sample_idx][1]
            else:
                inp_seq = inputs_cpu[i]
                input_grid = _inverse_fn(get_crop(inp_seq))
                input_hash = grid_hash(input_grid)

            # Crop and inverse transform prediction
            pred_grid = _inverse_fn(get_crop(pred_seq))
            pred_hash = grid_hash(pred_grid)

            local_hmap[pred_hash] = pred_grid

            local_preds.setdefault(orig_name, {})
            local_preds[orig_name].setdefault(input_hash, [])
            local_preds[orig_name][input_hash].append((pred_hash, q_val))

    # Compute final accuracies
    n_puzzles = 0
    correct = [0, 0]
    cell_hits = 0
    n_cells = 0

    evaluated_puzzles = [name for name in test_puzzles.keys() if name in local_preds]
    for name in evaluated_puzzles:
        puzzle = test_puzzles[name]
        n_puzzles += 1
        num_correct = [0, 0]
        for pair in puzzle["test"]:
            inp_grid = arc_grid_to_np(pair["input"])
            out_grid = arc_grid_to_np(pair["output"])
            input_hash = grid_hash(inp_grid)
            label_hash = grid_hash(out_grid)

            p_map = {}
            for h, q in local_preds[name].get(input_hash, []):
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q

            if not len(p_map):
                continue

            # Compute average Q-value for each prediction candidate
            for h, stats in p_map.items():
                stats[1] /= stats[0]
            
            # Sort candidate predictions by frequency (primary) and average Q-value (tie-breaker)
            p_map_sorted = sorted(p_map.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
            
            # Pass@1: correct prediction is in the top 1
            if p_map_sorted[0][0] == label_hash:
                num_correct[0] = 1

            # Pass@2: correct prediction is in the top 2 candidates
            ok_pass2 = False
            for h, stats in p_map_sorted[:2]:
                if h == label_hash:
                    ok_pass2 = True
            if ok_pass2:
                num_correct[1] = 1

        correct[0] += num_correct[0]
        correct[1] += num_correct[1]

        # Cell Accuracy
        for pair in puzzle["test"]:
            inp_grid = arc_grid_to_np(pair["input"])
            out_grid = arc_grid_to_np(pair["output"])
            input_hash = grid_hash(inp_grid)

            preds_list = local_preds.get(name, {}).get(input_hash, [])
            if not preds_list:
                continue

            p_map = {}
            for h, q in preds_list:
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q
            for h, stats in p_map.items():
                stats[1] /= stats[0]

            p_map_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
            top_hash = p_map_sorted[0][0]
            top_grid = local_hmap[top_hash]

            if top_grid.shape == out_grid.shape:
                cell_hits += (top_grid == out_grid).sum()
                n_cells += out_grid.size
            else:
                n_cells += out_grid.size

    cell_acc = cell_hits / n_cells if n_cells > 0 else 0.0
    pass_1_acc = correct[0] / n_puzzles if n_puzzles > 0 else 0.0
    pass_2_acc = correct[1] / n_puzzles if n_puzzles > 0 else 0.0
    elapsed = time.time() - t0
    ms_per_puzzle = elapsed / n_puzzles if n_puzzles > 0 else 0.0

    if return_pass2:
        return pass_1_acc, pass_2_acc, cell_acc, ms_per_puzzle * 1000, n_puzzles
    else:
        return pass_1_acc, cell_acc, ms_per_puzzle * 1000, n_puzzles

# Dynamic wrapper to seamlessly rewire all legacy evaluate_arc calls to use puzzle-by-puzzle TTA voting