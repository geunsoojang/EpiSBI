from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sim_to_array(sim_output):
    if isinstance(sim_output, dict):
        key = "data" if "data" in sim_output else list(sim_output.keys())[0]
        sim_output = sim_output[key]
    arr = _to_numpy(sim_output).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Simulator output must have shape (time,) or (time, n_outputs), got {arr.shape}.")
    return arr


def _theta_to_input(theta, param_names: Optional[Sequence[str]] = None):
    if param_names is None:
        return theta
    return {name: float(theta[i]) for i, name in enumerate(param_names)}


def sample_prior(prior, num_simulations: int, seed: int = 0):
    if num_simulations <= 0:
        raise ValueError("num_simulations must be positive.")
    rng = np.random.default_rng(seed)
    low = _to_numpy(prior.low).astype(np.float32)
    high = _to_numpy(prior.high).astype(np.float32)
    return rng.uniform(low, high, size=(num_simulations, low.size)).astype(np.float32)


def simulate_for_sbi(
    prior,
    simulator: Callable,
    num_simulations: int,
    total_days: int,
    seed: int = 0,
    param_names: Optional[Sequence[str]] = None,
    simulator_kwargs: Optional[dict] = None,
):

    if total_days <= 0:
        raise ValueError("total_days must be positive.")
    simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
    if param_names is None and hasattr(prior, "names"):
        param_names = prior.names

    thetas = sample_prior(prior, num_simulations=num_simulations, seed=seed)
    xs = []
    for sim_id, theta in enumerate(thetas):
        theta_input = _theta_to_input(theta, param_names)
        sim = simulator(theta_input, total_days=total_days, **simulator_kwargs)
        arr = _sim_to_array(sim)
        if arr.shape[0] < total_days:
            raise ValueError(f"Simulator output must contain at least {total_days} time points, got {arr.shape[0]}.")
        xs.append(arr[:total_days])
    return thetas, np.stack(xs).astype(np.float32)
