from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _dict_to_array(values: dict):
    columns = []
    for value in values.values():
        arr = _to_numpy(value).astype(np.float32)
        if arr.ndim == 0:
            arr = arr[None]
        if arr.ndim > 1 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim != 1:
            raise ValueError(f"Named simulator output values must be one-dimensional, got {arr.shape}.")
        columns.append(arr)
    return np.stack(columns, axis=1)


def _sim_to_array(sim_output, output_key: str = "observed"):
    if isinstance(sim_output, dict):
        if output_key in sim_output:
            sim_output = sim_output[output_key]
        elif "observed" in sim_output:
            sim_output = sim_output["observed"]
        elif "data" in sim_output:
            sim_output = sim_output["data"]
        elif "transitions" in sim_output:
            sim_output = sim_output["transitions"]
        elif "compartments" in sim_output:
            sim_output = sim_output["compartments"]
        else:
            sim_output = _dict_to_array(sim_output)
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
    total_days: Optional[int] = None,
    pass_seed: bool = True,
    output_key: str = "observed",
):
    theta_input = _theta_to_input(theta, param_names)
    kwargs = {} if simulator_kwargs is None else dict(simulator_kwargs)
    if total_days is not None and "total_days" not in kwargs:
        kwargs["total_days"] = total_days
    if pass_seed and seed is not None and "seed" not in kwargs:
        kwargs["seed"] = int(seed)
    arr = _sim_to_array(simulator(theta_input, **kwargs), output_key=output_key)
    if total_days is not None and arr.shape[0] < total_days:
        raise ValueError(f"Simulator output must contain at least {total_days} time points, got {arr.shape[0]}.")
    return arr[:total_days] if total_days is not None else arr


def simulate_for_sbi(
    prior,
    simulator: Callable,
    num_simulations: int,
    total_days: int,
    seed: int = 0,
    param_names: Optional[Sequence[str]] = None,
    simulator_kwargs: Optional[dict] = None,
    pass_seed: bool = True,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 0,
    output_key: str = "observed",
):
    if total_days <= 0:
        raise ValueError("total_days must be positive.")
    if n_jobs == 0:
        raise ValueError("n_jobs must be nonzero.")

    simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
    if param_names is None and hasattr(prior, "names"):
        param_names = prior.names

    thetas = sample_prior(prior, num_simulations=num_simulations, seed=seed)
    rng = np.random.default_rng(seed)
    sim_seeds = rng.integers(0, 2**32 - 1, size=num_simulations, dtype=np.uint32)

    if n_jobs == 1:
        xs = [
            _simulate_one(
                theta,
                simulator,
                param_names,
                simulator_kwargs,
                seed=sim_seed,
                total_days=total_days,
                pass_seed=pass_seed,
                output_key=output_key,
            )
            for theta, sim_seed in zip(thetas, sim_seeds)
        ]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError("simulate_for_sbi(n_jobs != 1) requires joblib.") from exc

        xs = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
            delayed(_simulate_one)(
                theta,
                simulator,
                param_names,
                simulator_kwargs,
                seed=sim_seed,
                total_days=total_days,
                pass_seed=pass_seed,
                output_key=output_key,
            )
            for theta, sim_seed in zip(thetas, sim_seeds)
        )
    return thetas, np.stack(xs).astype(np.float32)
