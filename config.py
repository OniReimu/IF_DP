# =================================================================
# Configuration file for Fisher DP-SGD Project
# =================================================================

import os
import torch
import numpy as np

# Default random seed for reproducible experiments
DEFAULT_SEED = 40

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

# For backward compatibility, expose the seed value directly
RANDOM_SEED = get_random_seed() 