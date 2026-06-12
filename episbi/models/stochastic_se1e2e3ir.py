from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Optional

import numpy as np


Observation = Literal["daily_cases", "cumulative_cases", "weekly_incidence", "final_size"]


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

def weekly_incidence(daily_cases: Iterable[float], start_day: int = 4, end_day: int = 8*7+4) -> np.ndarray:
    daily = np.asarray(daily_cases, dtype=np.float32).reshape(-1)
    padded = np.pad(daily, (0, max(0, end_day - daily.size)))
    return np.array(
        [padded[start : start + 7].sum() for start in range(start_day, end_day, 7)],
        dtype=np.float32,
    )

def stochastic_se1e2e3ir(
    theta,
    total_days: int = 45,
    population: int = 149,
    dt: float = 0.01,
    incubation_mean: float = 18.0,
    infectious_mean: float = 8.0,
    intervention_day: Optional[float] = 35.0,
    post_intervention_r: float = 0.96,
    initial_e1: int = 0,
    initial_e2: int = 0,
    initial_e3: int = 0,
    initial_infectious: int = 0,
    initial_recovered: int = 0,
    seed: Optional[int] = None,
    observation: Observation = "daily_cases",
    return_states: bool = False,
    return_transitions: bool = False,
):

    if total_days <= 0:
        raise ValueError("total_days must be positive.")
    if population <= 0:
        raise ValueError("population must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if incubation_mean <= 0 or infectious_mean <= 0:
        raise ValueError("incubation_mean and infectious_mean must be positive.")

    r0 = _theta_value(theta, "R0", 0)
    r1 = _theta_value(theta, "R1", 1, post_intervention_r)
    gamma = 1.0 / infectious_mean
    sigma = 3.0 / incubation_mean
    
    e1 = max(0, int(round(initial_e1)))
    e2 = max(0, int(round(initial_e2)))
    e3 = max(0, int(round(initial_e3)))
    i = max(0, int(round(initial_infectious)))
    r = max(0, int(round(initial_recovered)))
    initial_infected = e1 + e2 + e3 + i
    s = int(population) - e1 - e2 - e3 - i - r
    if s < 0:
        raise ValueError("Initial compartments exceed population.")

    steps = int(np.ceil(total_days / dt))
    rng = np.random.default_rng(seed)
    event_trace = np.zeros((steps, 5), dtype=np.float32)
    daily_states = np.zeros((total_days, 6), dtype=np.float32)

    for step in range(steps):
        t = step * dt
        beta = (r1 if intervention_day is not None and t >= intervention_day else r0) * gamma
        n = max(1, s + e1 + e2 + e3 + i + r)

        u1 = _poisson_event(rng, beta * s * i / n * dt, s)
        u2 = _poisson_event(rng, sigma * e1 * dt, e1)
        u3 = _poisson_event(rng, sigma * e2 * dt, e2)
        u4 = _poisson_event(rng, sigma * e3 * dt, e3)
        u5 = _poisson_event(rng, gamma * i * dt, i)

        s -= u1
        e1 += u1 - u2
        e2 += u2 - u3
        e3 += u3 - u4
        i += u4 - u5
        r += u5
        event_trace[step] = (u1, u2, u3, u4, u5)

        day = int(np.floor(t))
        if day < total_days:
            daily_states[day] = (s, e1, e2, e3, i, r)

    daily_cases = _daily_bins(event_trace[:, 3], dt=dt, total_days=total_days)
    daily_infections = _daily_bins(event_trace[:, 0], dt=dt, total_days=total_days)
    cumulative_cases = np.cumsum(daily_cases)
    final_size = float(initial_infected + daily_infections.sum())

    if observation == "daily_cases":
        data = daily_cases[:, None]
    elif observation == "cumulative_cases":
        data = cumulative_cases[:, None]
    elif observation == "weekly_incidence":
        data = weekly_incidence(daily_cases)[:, None]
    elif observation == "final_size":
        data = np.array([[final_size]], dtype=np.float32)
    else:
        raise ValueError("observation must be daily_cases, cumulative_cases, weekly_incidence, or final_size.")

    result = {"data": data.astype(np.float32)}
    if return_states:
        result["states"] = daily_states
    if return_transitions:
        result["transitions"] = event_trace
        result["daily_cases"] = daily_cases
        result["daily_infections"] = daily_infections
        result["final_size"] = final_size
    return result
