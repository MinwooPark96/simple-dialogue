import random
import torch
import numpy as np
import os

def set_seed(seed: int = 42):
    # Set seed for python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy random number generator
    np.random.seed(seed)
    
    # Set seed for PyTorch on the CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch on the GPU (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you have multi-GPU setup
    
    # Make sure that CUDA operations are deterministic (may slow down performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
