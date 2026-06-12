from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch
import pyabc
from sbi.utils import BoxUniform

@dataclass
class UniformPrior:
    bounds: Dict[str, Tuple[float, float]]
    device: str = "cpu"

    @property
    def names(self):
        return list(self.bounds.keys())

    @property
    def low(self):
        return np.array([self.bounds[name][0] for name in self.names], dtype=np.float32)

    @property
    def high(self):
        return np.array([self.bounds[name][1] for name in self.names], dtype=np.float32)

    @property
    def pyabc(self):
        return pyabc.Distribution(**{name: pyabc.RV("uniform", low, high - low) for name, (low, high) in self.bounds.items()})

    @property
    def sbi(self):
        return BoxUniform(
            low=torch.tensor(self.low, dtype=torch.float32, device=self.device),
            high=torch.tensor(self.high, dtype=torch.float32, device=self.device),
        )

    def to_dict(self, theta):
        if isinstance(theta, dict):
            return theta
        if hasattr(theta, "to_dict"):
            return theta.to_dict()
        if hasattr(theta, "detach"):
            theta = theta.detach().cpu().numpy()
        theta = np.asarray(theta, dtype=float)
        return {name: float(theta[i]) for i, name in enumerate(self.names)}

    def to_array(self, theta):
        if isinstance(theta, dict):
            return np.array([theta[name] for name in self.names], dtype=np.float32)
        if hasattr(theta, "to_dict"):
            d = theta.to_dict()
            return np.array([d[name] for name in self.names], dtype=np.float32)
        if hasattr(theta, "detach"):
            theta = theta.detach().cpu().numpy()
        return np.asarray(theta, dtype=np.float32)

@dataclass
class MixedPrior:
    continuous_bounds: Dict[str, Tuple[float, float]]
    discrete_bounds: Dict[str, Tuple[int, int]]

    @property
    def names(self):
        return list(self.continuous_bounds.keys()) + list(self.discrete_bounds.keys())

    @property
    def discrete_names(self):
        return list(self.discrete_bounds.keys())

    @property
    def pyabc(self):
        distributions = {
            name: pyabc.RV("uniform", low, high - low)
            for name, (low, high) in self.continuous_bounds.items()
        }
        distributions.update(
            {
                name: pyabc.RV("randint", low, high)
                for name, (low, high) in self.discrete_bounds.items()
            }
        )
        return pyabc.Distribution(**distributions)

    def to_dict(self, theta):
        if isinstance(theta, dict):
            return {
                name: int(theta[name]) if name in self.discrete_names else float(theta[name])
                for name in self.names
            }
        if hasattr(theta, "to_dict"):
            theta = theta.to_dict()
            return self.to_dict(theta)
        theta = np.asarray(theta, dtype=float)
        values = {}
        for i, name in enumerate(self.names):
            values[name] = int(round(theta[i])) if name in self.discrete_names else float(theta[i])
        return values