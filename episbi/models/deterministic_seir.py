from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence

import numpy as np
from scipy.integrate import odeint

Observation = Literal["incidence", "compartments"]


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


def ode_system(y, t, beta: float, kappa: float, gamma: float, population: float):
    s, e, i, r = y
    infection = beta * s * i / population
    return [
        -infection,
        infection - kappa * e,
        kappa * e - gamma * i,
        gamma * i,
    ]

def seir_ode_simulator(
    parameters: Sequence[float],
    initial_conditions: Sequence[float],
    total_days: int):

    beta, kappa, gamma = parameters
    s0, e0, i0, r0 = initial_conditions
    population = s0 + e0 + i0 + r0
    t = np.arange(total_days + 1)
    trajectory = odeint(ode_system, initial_conditions, t, args=(beta, kappa, gamma, population))
    return trajectory[1:]
    

def deterministic_seir(
    theta,
    total_days: int = 100,
    population: int = 100_000,
    initial_conditions: Optional[Sequence[float]] = None,
    initial_exposed: int = 0,
    initial_infectious: Optional[int] = None,
    initial_recovered: int = 0,
    observation: Observation = "incidence",
    observation_noise: Optional[Literal["poisson"]] = None,
    seed: Optional[int] = None,
    return_compartments: bool = False,
):
    beta = _theta_value(theta, "beta", 0)
    kappa = _theta_value(theta, "kappa", 1)
    gamma = _theta_value(theta, "gamma", 2)

    if initial_conditions is None:
        if initial_infectious is None:
            initial_infectious = 1.0
        e0 = max(0.0, float(initial_exposed))
        i0 = max(0.0, float(initial_infectious))
        r0 = max(0.0, float(initial_recovered))
        s0 = float(population) - e0 - i0 - r0
        initial_conditions = [s0, e0, i0, r0]
    
    initial_conditions = np.asarray(initial_conditions, dtype=float)
    if initial_conditions.shape != (4,):
        raise ValueError("initial_conditions must contain S0, E0, I0, R0.")
    if np.any(initial_conditions < 0):
        raise ValueError("Initial compartments must be non-negative.")
    if not np.isclose(initial_conditions.sum(), population):
        population = float(initial_conditions.sum())
    if population <= 0:
        raise ValueError("Initial compartments must sum to a positive population.")

    compartments = seir_ode_simulator(parameters=[beta, kappa, gamma], initial_conditions=initial_conditions, total_days=total_days).astype(np.float32)

    if observation == "incidence":
        data = (kappa * compartments[:, 1])[:, None]
    elif observation == "compartments":
        data = compartments
    else:
        raise ValueError("observation must be 'incidence' or 'compartments'.")

    if observation_noise == "poisson":
        rng = np.random.default_rng(seed)
        data = rng.poisson(np.clip(data, a_min=0, a_max=None)).astype(np.float32)
    elif observation_noise is not None:
        raise ValueError("observation_noise must be None or 'poisson'.")

    result = {"data": data.astype(np.float32)}
    if return_compartments:
        result["compartments"] = compartments
    return result