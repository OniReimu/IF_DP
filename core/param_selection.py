"""Shared utility for selecting parameters under DP parameter budget constraints.

This ensures all DP training methods (Vanilla DP-SGD, DP-SAT, Fisher DP) select
the same parameters when using --dp-param-count, enabling fair comparison.
"""

from typing import List, Tuple, Optional
import torch.nn as nn


def select_parameters_by_budget(
    model: nn.Module,
    dp_param_count: int,
    target_layer: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[str], List[nn.Parameter], dict]:
    """
    Select parameters for DP training under a budget constraint.
    
    Uses head-first selection: prioritizes classifier/head parameters first,
    then fills remaining budget with backbone parameters in model order.
    
    Args:
        model: PyTorch model
        dp_param_count: Maximum number of parameters to select
        target_layer: If provided and dp_param_count is None, select by layer prefix
        verbose: Whether to print selection details
    
    Returns:
        Tuple of:
        - names: List of selected parameter names
        - params: List of selected parameter tensors
        - stats: Dictionary with selection statistics
    """
    all_params = list(model.named_parameters())
    
    if dp_param_count is not None:
        # DP parameter budget mode: prioritize classifier/head parameters first
        if verbose:
            print(f"   🎯 DP Parameter Budget Mode: selecting up to {dp_param_count} parameters (head-first)")
        
        def _is_head_param(param_name: str) -> bool:
            parts = param_name.split(".")
            head_parts = {"classifier", "fc", "head", "lm_head", "score", "output"}
            return any(p in head_parts for p in parts)
        
        # Build list with head-first priority
        param_info = []
        for idx, (name, param) in enumerate(all_params):
            size = int(param.numel())
            priority = 1 if _is_head_param(name) else 0
            param_info.append((priority, idx, name, param, size))
        param_info.sort(key=lambda x: (-x[0], x[1]))  # Head params first, then by original order
        
        # Greedy knapsack: select parameters that fit within budget
        selected_indices = []
        total_selected = 0
        head_selected = 0
        head_tensors = [(name, size) for prio, _, name, _, size in param_info if prio == 1]
        head_total_params = sum(size for _, size in head_tensors)
        head_min_tensor = min([size for _, size in head_tensors] or [0])
        
        for priority, idx, name, param, size in param_info:
            if total_selected + size <= dp_param_count:
                selected_indices.append(idx)
                total_selected += size
                if priority == 1:
                    head_selected += 1
            elif total_selected < dp_param_count:
                # Would exceed budget - silently skip for cleaner logs
                continue
        
        # Extract selected parameters
        names = []
        params = []
        for idx in sorted(selected_indices):
            name, param = all_params[idx]
            names.append(name)
            params.append(param)

        selected_name_set = set(names)
        head_total_tensors = len(head_tensors)
        head_skipped = [(name, size) for name, size in head_tensors if name not in selected_name_set]
        head_oversize = [(name, size) for name, size in head_skipped if size > dp_param_count]

        unused = dp_param_count - total_selected
        efficiency = (total_selected / dp_param_count) * 100
        
        stats = {
            'total_selected': total_selected,
            'unused': unused,
            'efficiency': efficiency,
            'head_selected': head_selected,
            'head_total_params': head_total_params,
            'head_min_tensor': head_min_tensor,
            'head_total_tensors': head_total_tensors,
            'head_skipped': head_skipped,
            'head_oversize': head_oversize,
        }
        
        if verbose:
            print(f"   ✅ Selected {len(names)} complete parameters")
            print(f"      Budget: {dp_param_count} | Used: {total_selected} | Unused: {unused} ({efficiency:.1f}% efficiency)")
            if head_total_params > 0:
                print(f"      Head params: selected {head_selected} tensors (head scalars available: {head_total_params})")
                if head_selected == 0 and head_min_tensor > 0 and dp_param_count < head_min_tensor:
                    print(f"   ⚠️  Budget too small to include even the smallest head tensor (min head tensor size={head_min_tensor}).")
                    print(f"      Increase --dp-param-count or use --dp-layer to target the classifier/head directly.")
                elif head_skipped:
                    skipped_preview = ", ".join(f"{n}({s})" for n, s in head_skipped[:3])
                    extra = "" if len(head_skipped) <= 3 else f", +{len(head_skipped)-3} more"
                    print(f"   ⚠️  Some head tensors were skipped under this budget: {skipped_preview}{extra}")
                    if head_oversize:
                        biggest = max(head_oversize, key=lambda x: x[1])
                        print(f"      Budget {dp_param_count} cannot include {biggest[0]} (size={biggest[1]}), so it will be frozen.")
                        print(f"      If you need the classifier/head to adapt (e.g., excluded classes), increase --dp-param-count or use --dp-layer backbone.classifier.")
        
        return names, params, stats
    
    else:
        # Layer-based selection (legacy mode)
        def _match(name: str, layer: str) -> bool:
            return name.startswith(layer)
        
        if target_layer == "all":
            names = [n for n, _ in all_params]
        elif target_layer and "," in target_layer:
            layers = [s.strip() for s in target_layer.split(",")]
            names = [n for n, _ in all_params
                    if any(_match(n, l) for l in layers)]
        elif target_layer:
            names = [n for n, _ in all_params if _match(n, target_layer)]
        else:
            names = [n for n, _ in all_params]
        
        params = [dict(all_params)[n] for n in names]
        
        stats = {
            'total_selected': sum(p.numel() for p in params),
            'unused': 0,
            'efficiency': 100.0,
            'head_selected': 0,
            'head_total_params': 0,
            'head_min_tensor': 0
        }
        
        return names, params, stats
