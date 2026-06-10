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