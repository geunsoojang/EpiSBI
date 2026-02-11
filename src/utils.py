from sbi.utils.metrics import c2st
import torch
import numpy as np
import random
import os


def set_seed(seed: int = 0):
    """
    Fixes the seed for reproducibility across numpy, torch, and python random.
    Also ensures deterministic behavior in CuDNN for GPU operations.
    """
    # 1. Base Python & OS
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # 2. Numpy
    np.random.seed(seed)
    # 3. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # 4. CuDNN Determinism
    # These flags ensure that GPU operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[#] Seed has been fixed to: {seed}")
    
    
def add_poisson_noise(data):
    """
    Applies Poisson noise to the input data.
    Suitable for cases where variance equals the mean.
    """
    return np.random.poisson(np.maximum(data, 1e-6))


def add_nb_noise(data, alpha=0.1):
    """
    Applies Negative Binomial (NB) noise to the input data.
    Formula: variance = mean + alpha * mean^2
    
    Args:
        data (ndarray): The mean values (mu) for the distribution.
        alpha (float): Overdispersion parameter. Large alpha means 
        more variance.
    """
    mu = np.maximum(data, 1e-6)
    # Convert mu and alpha to n (number of successes) and 
    # p (probability of success)
    n = 1.0 / alpha
    p = n / (n + mu)
    return np.random.negative_binomial(n, p)


def apply_noise(data, noise_type='poisson', **kwargs):
    """
    Wrapper function to apply different types of noise.
    """
    if noise_type == 'poisson':
        return add_poisson_noise(data)
    elif noise_type == 'nb':
        alpha = kwargs.get('alpha', 0.1)
        return add_nb_noise(data, alpha=alpha)
    else:
        return data
    

def calculate_c2st(samples_a, samples_b):
    """
    Computes the Classifier 2-Sample Test (C2ST) accuracy.
    A value of 0.5 means the distributions are identical.
    """
    return c2st(samples_a, samples_b)
