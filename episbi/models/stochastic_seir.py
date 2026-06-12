from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def _theta_value(theta, name: str, index: int, default: Optional[float] = None) -> float:
    if isinstance(theta, Mapping):
        if name in theta:
            return float(theta[name])
        if default is not None:
            return float(default)
        raise KeyError(f"Missing parameter {name!r}.")
    values = theta.detach().cpu().numpy() if hasattr(theta, "detach") else theta
    values = np.asarray(values, dtype=float)
    if index < values.size:
        return float(values[index])
    if default is not None:
        return float(default)
    raise IndexError(f"Missing parameter {name!r} at index {index}.")


def _poisson_event(rng: np.random.Generator, mean: float, available: int) -> int:
    mean = max(0.0, float(mean))
    available = max(0, int(available))
    return min(available, int(rng.poisson(mean)))

def _daily_bins(values: np.ndarray, dt: float, total_days: int) -> np.ndarray:
    daily = np.zeros(total_days, dtype=np.float32)
    for step, value in enumerate(values):
        day = min(total_days - 1, int(np.floor(step * dt)))
        daily[day] += value
    return daily


def stochastic_seir(
    theta,
    total_days: int = 100,
    population: int = 100_000,
    dt: float = 0.01,
    initial_conditions: Optional[Sequence[float]] = None,
    initial_exposed: int = 0,
    initial_infectious: Optional[int] = None,
    initial_recovered: int = 0,
    seed: Optional[int] = None,
    return_compartments: bool = False,
    
):
    if total_days <= 0:
        raise ValueError("total_days must be positive.")
    if population <= 0:
        raise ValueError("population must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

        
    beta = _theta_value(theta, "beta", 0)
    kappa = _theta_value(theta, "kappa", 1)
    gamma = _theta_value(theta, "gamma", 2)


    rng = np.random.default_rng(seed)
    if initial_conditions is None:
        if initial_infectious is None:
            initial_infectious = 1.0
        e = max(0, int(round(initial_exposed)))
        i = max(0, int(round(initial_infectious)))
        r = max(0, int(round(initial_recovered)))
        s = int(population) - e - i - r
        if s < 0:
            raise ValueError("Initial compartments exceed population.")
    initial_conditions = np.asarray(initial_conditions, dtype=float)
    s,e,i,r = initial_conditions
    
    steps = int(np.ceil(total_days / dt))
    transitions = np.zeros((steps, 3), dtype=np.float32)
    compartments = np.zeros((total_days, 4), dtype=np.float32)

    for step in range(steps):
        n = max(1, s + e + i + r)
        new_exposed = _poisson_event(rng, beta * s * i / n * dt, s)
        new_infectious = _poisson_event(rng, kappa * e * dt, e)
        new_recovered = _poisson_event(rng, gamma * i * dt, i)

        s -= new_exposed
        e += new_exposed - new_infectious
        i += new_infectious - new_recovered
        r += new_recovered

        transitions[step] = (new_exposed, new_infectious, new_recovered)
        day = min(total_days - 1, int(np.floor(step * dt)))
        compartments[day] = (s, e, i, r)
    
    daily_cases = _daily_bins(transitions[:, 1], dt=dt, total_days=total_days)
    result = {"data": daily_cases[:, None]}
    if return_compartments:
        result["compartments"] = compartments
    return result