# =================================================================
# Configuration file for Fisher DP-SGD Project
# =================================================================

import os
import torch
import numpy as np

# Default random seed for reproducible experiments
DEFAULT_SEED = 400

# Rehearsal buffer threshold: maximum allowed ratio of excluded classes in private set
# When using --public-pretrain-exclude-classes, we add rehearsal samples from other classes
# to private until excluded classes are no longer the majority (≤ 50%).
REHEARSAL_MAX_EXCLUDED_CLASS_RATIO = 0.5

def get_random_seed():
    """
    Get the random seed from environment variable or use default.
    
    Returns:
        int: Random seed value
    """
    return int(os.getenv('RANDOM_SEED', DEFAULT_SEED))

def set_random_seeds(seed=None):
    """
    Set random seeds for torch and numpy for reproducible experiments.
    
    Args:
        seed (int, optional): Random seed. If None, uses get_random_seed()
    """
    if seed is None:
        seed = get_random_seed()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Also set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"🎲 Random seeds set to {seed}")

def get_dataset_location(default='./data', dataset_key=None, required_subdir=None):
    """
    Return the dataset root directory and whether downloading is allowed.
    
    Prefers the HF_DATASETS_CACHE environment variable (local data already
    present) and falls back to the default path otherwise.
    
    Args:
        default (str): Path used when HF_DATASETS_CACHE is not set.
        dataset_key (str, optional): Subdirectory inside HF cache that holds
            the dataset (e.g., "cifar10"). Only applied when HF_DATASETS_CACHE
            is defined to avoid breaking local ./data layouts.
        required_subdir (str, optional): Folder that must exist under the
            dataset root to consider the dataset already present (e.g.,
            "cifar-10-batches-py").
    
    Returns:
        tuple[str, bool]: (dataset_root, allow_download)
    """
    cache_root = os.getenv('HF_DATASETS_CACHE')
    if cache_root:
        cache_root = os.path.abspath(cache_root)
        os.makedirs(cache_root, exist_ok=True)
        dataset_root = cache_root
        if dataset_key:
            dataset_root = os.path.join(cache_root, dataset_key)
        os.makedirs(dataset_root, exist_ok=True)
        
        required_path = os.path.join(dataset_root, required_subdir) if required_subdir else None
        data_present = required_path and os.path.isdir(required_path)
        allow_download = not bool(data_present)
        location_note = f"{dataset_root} (HF_DATASETS_CACHE)"
        if data_present:
            print(f"📁 Using local dataset cache: {location_note}")
        else:
            print(f"📁 Preparing local dataset cache: {location_note} (download if needed)")
        return dataset_root, allow_download
    
    dataset_root = os.path.abspath(default)
    os.makedirs(dataset_root, exist_ok=True)
    print(f"📁 Using default dataset path: {dataset_root} (downloads enabled)")
    return dataset_root, True

# For backward compatibility, expose the seed value directly
RANDOM_SEED = get_random_seed()
