from __future__ import annotations

from typing import Literal, Mapping, Optional

import numpy as np


Observation = Literal["cumulative_cases", "daily_cases", "cases_deaths"]


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


def _poisson_transition(rng: np.random.Generator, mean: float, available: int) -> int:
    mean = max(0.0, float(mean))
    available = max(0, int(available))
    return min(available, int(rng.poisson(mean)))


def stochastic_seird(theta, total_days: int = 37, population: int = 1_000,
    initial_exposed: Optional[int] = None, initial_infectious: int = 1, initial_recovered: int = 0, initial_dead: int = 0,
    seed: Optional[int] = None, observation: Observation = "cumulative_cases",
    return_states: bool = False, return_transitions: bool = False):

    if total_days <= 0:
        raise ValueError("total_days must be positive.")
    if population <= 0:
        raise ValueError("population must be positive.")

    beta = _theta_value(theta, "beta", 0)
    z_period = _theta_value(theta, "Z", 1)
    d_period = _theta_value(theta, "D", 2)
    delta = _theta_value(theta, "delta", 3)
    if initial_exposed is None:
        initial_exposed = int(round(_theta_value(theta, "E0", 4, 1.0)))

    if z_period <= 0 or d_period <= 0:
        raise ValueError("Z and D must be positive.")
    if not 0 <= delta <= 1:
        raise ValueError("delta must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    e = max(0, int(round(initial_exposed)))
    i = max(0, int(round(initial_infectious)))
    r = max(0, int(round(initial_recovered)))
    dead = max(0, int(round(initial_dead)))
    s = int(population) - e - i - r - dead
    if s < 0:
        raise ValueError("Initial compartments exceed population.")

    states = np.zeros((total_days, 5), dtype=np.float32)
    transitions = np.zeros((total_days, 4), dtype=np.float32)
    daily_cases = np.zeros(total_days, dtype=np.float32)
    daily_deaths = np.zeros(total_days, dtype=np.float32)

    for day in range(total_days):
        s_t, e_t, i_t = s, e, i
        n = max(1, s + e + i + r + dead)
        u1 = _poisson_transition(rng, beta * s_t * i_t / n, s_t)
        u2 = _poisson_transition(rng, e_t / z_period, e_t)
        u3 = _poisson_transition(rng, (1.0 - delta) * i_t / d_period, i_t)
        u4 = _poisson_transition(rng, delta * i_t / d_period, i_t - u3)

        s -= u1
        e += u1 - u2
        i += u2 - u3 - u4
        r += u3
        dead += u4

        daily_cases[day] = u2
        daily_deaths[day] = u4
        transitions[day] = (u1, u2, u3, u4)
        states[day] = (s, e, i, r, dead)

    cumulative_cases = np.cumsum(daily_cases)
    cumulative_deaths = np.cumsum(daily_deaths)
    if observation == "cumulative_cases":
        data = cumulative_cases[:, None]
    elif observation == "daily_cases":
        data = daily_cases[:, None]
    elif observation == "cases_deaths":
        data = np.column_stack([cumulative_cases, cumulative_deaths]).astype(np.float32)
    else:
        raise ValueError("observation must be 'cumulative_cases', 'daily_cases', or 'cases_deaths'.")

    result = {"data": data.astype(np.float32)}
    if return_states:
        result["states"] = states
    if return_transitions:
        result["transitions"] = transitions
    return result
