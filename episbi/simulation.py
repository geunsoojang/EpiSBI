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

def _simulate_one(
    theta,
    simulator: Callable,
    param_names: Optional[Sequence[str]] = None,
    simulator_kwargs: Optional[dict] = None,
    seed: Optional[int] = None,
):
    theta_input = _theta_to_input(theta, param_names)
    kwargs = {} if simulator_kwargs is None else dict(simulator_kwargs)
    if seed is not None and "seed" not in kwargs:
        kwargs["seed"] = int(seed)
    sim = simulator(theta_input, **kwargs)
    return _sim_to_array(sim)

def simulate_for_sbi(
    prior,
    simulator: Callable,
    num_simulations: int,
    seed: int = 0,
    param_names: Optional[Sequence[str]] = None,
    simulator_kwargs: Optional[dict] = None,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
):

    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero.")
        
    simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
    if param_names is None and hasattr(prior, "names"):
        param_names = prior.names

    thetas = sample_prior(prior, num_simulations=num_simulations, seed=seed)
    if n_jobs == 1:
        xs = [_simulate_one(theta, simulator, param_names, simulator_kwargs) for theta in thetas]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError("simulate_for_sbi(n_jobs != 1) requires joblib.") from exc
        xs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(_simulate_one)(theta, simulator, param_names, simulator_kwargs) for theta in thetas)
    return thetas, np.stack(xs).astype(np.float32)
