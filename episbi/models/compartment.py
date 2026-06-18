from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.integrate import odeint


Rate = Union[str, Callable[[Mapping[str, float], Mapping[str, float], float], float]]
ModelType = Literal["deterministic", "stochastic"]
ObservationError = Optional[Literal["poisson"]]


@dataclass(frozen=True)
class Transition:
    source: str
    target: str
    rate: Rate
    name: Optional[str] = None


def _theta_to_mapping(theta, param_names: Sequence[str]) -> dict[str, float]:
    if isinstance(theta, Mapping):
        return {name: float(theta[name]) for name in param_names}
    values = theta.detach().cpu().numpy() if hasattr(theta, "detach") else theta
    values = np.asarray(values, dtype=float)
    if values.size < len(param_names):
        raise ValueError(f"theta has {values.size} values, but {len(param_names)} parameters are required.")
    return {name: float(values[i]) for i, name in enumerate(param_names)}


def _as_transition(transition) -> Transition:
    if isinstance(transition, Transition):
        return transition
    if isinstance(transition, Mapping):
        return Transition(
            source=str(transition["source"]),
            target=str(transition["target"]),
            rate=transition["rate"],
            name=transition.get("name"),
        )
    if isinstance(transition, Sequence) and not isinstance(transition, (str, bytes)):
        if len(transition) == 3:
            source, target, rate = transition
            return Transition(str(source), str(target), rate)
        if len(transition) == 4:
            source, target, rate, name = transition
            return Transition(str(source), str(target), rate, str(name))
    raise TypeError("Each transition must be a Transition, dict, or tuple of (source, target, rate[, name]).")


class DeterministicModel:
    model_type = "deterministic"

    def __init__(
        self,
        compartments: Sequence[str],
        transitions: Iterable[Transition],
        param_names: Sequence[str],
        *,
        initial_conditions: Optional[Mapping[str, float]] = None,
        observed: Optional[Mapping[str, Sequence[str]]] = None,
    ):
        if len(compartments) == 0:
            raise ValueError("At least one compartment is required.")
        if len(set(compartments)) != len(compartments):
            raise ValueError("Compartment names must be unique.")
        if len(param_names) == 0:
            raise ValueError("At least one parameter name is required.")

        self.compartments = tuple(str(name) for name in compartments)
        self.param_names = tuple(str(name) for name in param_names)
        self.transitions = tuple(_as_transition(transition) for transition in transitions)
        self.initial_conditions = dict(initial_conditions or {})
        self._index = {name: idx for idx, name in enumerate(self.compartments)}
        self._compiled_rates = tuple(self._compile_rate(transition.rate) for transition in self.transitions)
        self.transition_names = tuple(
            transition.name if transition.name is not None else f"{transition.source}_to_{transition.target}"
            for transition in self.transitions
        )

        for transition in self.transitions:
            if transition.source not in self._index:
                raise ValueError(f"Unknown source compartment {transition.source!r}.")
            if transition.target not in self._index:
                raise ValueError(f"Unknown target compartment {transition.target!r}.")
        self.observed = self._normalize_observed(observed)
        self.observed_names = self.observed["compartments"] + self.observed["transitions"]
        if len(set(self.observed_names)) != len(self.observed_names):
            raise ValueError("Observed names must be unique across compartments and transitions.")

    def _normalize_observed(self, observed: Optional[Mapping[str, Sequence[str]]]) -> dict[str, tuple[str, ...]]:
        if observed is None:
            observed = {"transitions": self.transition_names}
        compartments = tuple(str(name) for name in observed.get("compartments", ()))
        transitions = tuple(str(name) for name in observed.get("transitions", ()))
        unknown_compartments = sorted(set(compartments).difference(self.compartments))
        unknown_transitions = sorted(set(transitions).difference(self.transition_names))
        if unknown_compartments:
            raise ValueError(f"Unknown observed compartments: {unknown_compartments}")
        if unknown_transitions:
            raise ValueError(f"Unknown observed transitions: {unknown_transitions}")
        if not compartments and not transitions:
            raise ValueError("At least one observed compartment or transition is required.")
        return {"compartments": compartments, "transitions": transitions}

    def _compile_rate(self, rate: Rate):
        if callable(rate):
            return rate
        if not isinstance(rate, str):
            raise TypeError("Transition rate must be a string expression or callable.")
        code = compile(rate, "<compartment-rate>", "eval")
        allowed = set(self.compartments) | set(self.param_names) | {"N", "t", "np"}
        unknown = set(code.co_names) - allowed
        if unknown:
            raise ValueError(f"Unknown names in rate expression {rate!r}: {sorted(unknown)}")

        def evaluate(state: Mapping[str, float], params: Mapping[str, float], t: float) -> float:
            namespace = {**state, **params, "N": sum(state.values()), "t": t, "np": np}
            return float(eval(code, {"__builtins__": {}}, namespace))

        return evaluate

    def _initial_vector(
        self,
        population: Optional[float],
        initial_conditions: Optional[Mapping[str, float]],
    ) -> np.ndarray:
        values = dict(self.initial_conditions)
        if initial_conditions is not None:
            values.update(initial_conditions)

        missing = [name for name in self.compartments if name not in values]
        if len(missing) == 1 and population is not None:
            known_total = sum(float(values.get(name, 0.0)) for name in self.compartments if name in values)
            values[missing[0]] = float(population) - known_total
        elif missing:
            raise ValueError(f"Missing initial conditions for compartments: {missing}")

        y0 = np.asarray([float(values[name]) for name in self.compartments], dtype=float)
        if np.any(y0 < 0):
            raise ValueError("Initial compartments must be non-negative.")
        if population is not None and not np.isclose(y0.sum(), float(population)):
            raise ValueError("Initial compartments must sum to population.")
        if y0.sum() <= 0:
            raise ValueError("Initial compartments must sum to a positive population.")
        return y0

    def _rates(self, y: np.ndarray, params: Mapping[str, float], t: float) -> np.ndarray:
        state = {name: float(y[idx]) for name, idx in self._index.items()}
        rates = np.asarray([rate(state, params, t) for rate in self._compiled_rates], dtype=float)
        return np.clip(rates, a_min=0.0, a_max=None)

    def _rhs(self, y: np.ndarray, t: float, params: Mapping[str, float]) -> np.ndarray:
        dydt = np.zeros(len(self.compartments), dtype=float)
        for transition, rate in zip(self.transitions, self._rates(y, params, t)):
            dydt[self._index[transition.source]] -= rate
            dydt[self._index[transition.target]] += rate
        return dydt

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "compartments": list(self.compartments),
            "parameters": list(self.param_names),
            "initial_conditions": dict(self.initial_conditions),
            "observed": {key: list(value) for key, value in self.observed.items()},
            "transitions": [
                {
                    "name": name,
                    "source": transition.source,
                    "target": transition.target,
                    "rate": transition.rate if isinstance(transition.rate, str) else repr(transition.rate),
                }
                for name, transition in zip(self.transition_names, self.transitions)
            ],
        }

    def show(self) -> str:
        lines = [
            f"Model type: {self.model_type}",
            f"Compartments: {', '.join(self.compartments)}",
            f"Parameters: {', '.join(self.param_names)}",
            "Transitions:",
        ]
        for name, transition in zip(self.transition_names, self.transitions):
            lines.append(f"  {name}: {transition.source} -> {transition.target}, rate={transition.rate}")
        lines.append("Observed:")
        for name in self.observed_names:
            lines.append(f"  {name}")
        return "\n".join(lines)

    def _transition_flows(self, trajectory: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
        flows = np.zeros((trajectory.shape[0], len(self.transitions)), dtype=np.float32)
        for day, y in enumerate(trajectory, start=1):
            flows[day - 1] = self._rates(y, params, float(day))
        return flows

    def _observed_values(self, compartments: np.ndarray, transitions: np.ndarray) -> np.ndarray:
        values = []
        for name in self.observed["compartments"]:
            values.append(compartments[:, self.compartments.index(name)])
        for name in self.observed["transitions"]:
            values.append(transitions[:, self.transition_names.index(name)])
        return np.stack(values, axis=1).astype(np.float32)

    def _observed_result(self, observed: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: observed[:, idx].astype(np.float32)
            for idx, name in enumerate(self.observed_names)
        }

    def _apply_observation_error(
        self,
        observed: np.ndarray,
        observation_error: ObservationError = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if observation_error is None:
            return observed.astype(np.float32)
        if observation_error == "poisson":
            rng = np.random.default_rng(seed)
            return rng.poisson(np.clip(observed, a_min=0, a_max=None)).astype(np.float32)
        raise ValueError("observation_error must be None or 'poisson'.")

    def simulate(
        self,
        theta,
        *,
        total_days: int = 100,
        population: Optional[float] = None,
        initial_conditions: Optional[Mapping[str, float]] = None,
        observation_error: ObservationError = None,
        observation_noise: ObservationError = None,
        seed: Optional[int] = None,
        return_compartments: bool = False,
    ):
        if total_days <= 0:
            raise ValueError("total_days must be positive.")

        params = _theta_to_mapping(theta, self.param_names)
        y0 = self._initial_vector(population, initial_conditions)
        t = np.arange(total_days + 1)
        trajectory = odeint(self._rhs, y0, t, args=(params,))[1:].astype(np.float32)
        transitions = self._transition_flows(trajectory, params)
        observed = self._observed_values(trajectory, transitions)
        if observation_noise is not None and observation_error is None:
            observation_error = observation_noise
        observed = self._apply_observation_error(observed, observation_error, seed)

        return self._observed_result(observed)

    def _incidence(self, trajectory: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
        transitions = self._transition_flows(trajectory, params)
        return self._observed_values(trajectory, transitions)

    def __call__(self, theta, **kwargs):
        return self.simulate(theta, **kwargs)

class StochasticModel(DeterministicModel):
    model_type = "stochastic"

    def simulate(
        self,
        theta,
        *,
        total_days: int = 100,
        population: Optional[float] = None,
        initial_conditions: Optional[Mapping[str, float]] = None,
        dt: float = 0.01,
        observation_error: ObservationError = None,
        observation_noise: ObservationError = None,
        seed: Optional[int] = None,
        return_compartments: bool = False,
        return_transitions: bool = False,
    ):
        if total_days <= 0:
            raise ValueError("total_days must be positive.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        params = _theta_to_mapping(theta, self.param_names)
        state = np.rint(self._initial_vector(population, initial_conditions)).astype(int)
        if np.any(state < 0):
            raise ValueError("Initial compartments must be non-negative.")

        rng = np.random.default_rng(seed)
        steps = int(np.ceil(total_days / dt))
        step_events = np.zeros((steps, len(self.transitions)), dtype=np.float32)
        daily_events = np.zeros((total_days, len(self.transitions)), dtype=np.float32)
        daily_compartments = np.zeros((total_days, len(self.compartments)), dtype=np.float32)

        for step in range(steps):
            t = step * dt
            rates = self._rates(state.astype(float), params, t)
            events = rng.poisson(np.clip(rates * dt, a_min=0.0, a_max=None)).astype(int)
            events = self._cap_outgoing_events(events, state, rng)

            for idx, transition in enumerate(self.transitions):
                event_count = int(events[idx])
                if event_count == 0:
                    continue
                state[self._index[transition.source]] -= event_count
                state[self._index[transition.target]] += event_count

            day = min(total_days - 1, int(np.floor(t)))
            step_events[step] = events
            daily_events[day] += events
            daily_compartments[day] = state

        daily_compartments = self._fill_daily_compartments(daily_compartments)
        observed = self._observed_values(daily_compartments, daily_events)
        if observation_noise is not None and observation_error is None:
            observation_error = observation_noise
        observed = self._apply_observation_error(observed, observation_error, seed)

        return self._observed_result(observed)

    def _cap_outgoing_events(
        self,
        events: np.ndarray,
        state: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        capped = events.copy()
        for compartment, source_idx in self._index.items():
            outgoing = [idx for idx, transition in enumerate(self.transitions) if transition.source == compartment]
            if not outgoing:
                continue
            total = int(capped[outgoing].sum())
            available = int(state[source_idx])
            if total <= available:
                continue
            if available <= 0:
                capped[outgoing] = 0
                continue
            weights = capped[outgoing].astype(float)
            probabilities = weights / weights.sum()
            capped[outgoing] = rng.multinomial(available, probabilities)
        return capped

    def _fill_daily_compartments(self, daily_compartments: np.ndarray) -> np.ndarray:
        last = daily_compartments[0].copy()
        for day in range(daily_compartments.shape[0]):
            if np.any(daily_compartments[day]):
                last = daily_compartments[day].copy()
            else:
                daily_compartments[day] = last
        return daily_compartments


def compartment_model(
    compartments: Sequence[str],
    transitions: Iterable[Transition],
    param_names: Sequence[str],
    *,
    model_type: ModelType = "deterministic",
    **kwargs,
) -> Union[DeterministicModel, StochasticModel]:
    if model_type == "deterministic":
        return DeterministicModel(compartments, transitions, param_names, **kwargs)
    if model_type == "stochastic":
        return StochasticModel(compartments, transitions, param_names, **kwargs)
    raise ValueError("model_type must be 'deterministic' or 'stochastic'.")
