from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Tuple
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyabc
from sbi.inference import NPE, NRE
from sbi.neural_nets import posterior_nn
from .embedding import LSTMembedding

class SBIEngine:
    def __init__(self, density_estimator: str = "maf", device: str = "cpu", batch_size: int = 256):
        self.de_type = density_estimator
        self.device = device
        self.batch_size = batch_size

    def _param_names(self, prior):
        return list(prior.names) if hasattr(prior, "names") else None

    def _discrete_param_names(self, prior):
        return list(prior.discrete_names) if hasattr(prior, "discrete_names") else []

    def _reorder_parameter_frame(self, values, param_names: Optional[Sequence[str]] = None):
        if param_names is None or not isinstance(values, pd.DataFrame):
            return values
        ordered = [name for name in param_names if name in values.columns]
        remaining = [name for name in values.columns if name not in ordered]
        return values[ordered + remaining]

    def _resample_parameter_particles(self, values, weights, num_samples: int, random_state: int = 0):
        if not isinstance(values, pd.DataFrame):
            return values
        weights = np.asarray(weights, dtype=float)
        if weights.sum() <= 0:
            weights = None
        else:
            weights = weights / weights.sum()
        return values.sample(n=num_samples, replace=True, weights=weights, random_state=random_state).reset_index(drop=True)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def _obs_to_tensor(self, obs_data):
        if isinstance(obs_data, dict):
            key = list(obs_data.keys())[0]
            return self._to_tensor(obs_data[key])
        return self._to_tensor(obs_data)

    def _obs_to_pyabc_dict(self, obs_data):
        if isinstance(obs_data, dict):
            return {key: np.asarray(value, dtype=np.float32) for key, value in obs_data.items()}
        return self._array_to_pyabc_dict(obs_data)

    def _sim_to_array(self, sim_output):
        if isinstance(sim_output, dict):
            key = list(sim_output.keys())[0]
            sim_output = sim_output[key]
        return np.asarray(sim_output, dtype=np.float32)

    def _get_neural_net(self, use_lstm: bool = False, input_dim: int = 1, embedding_net: Optional[nn.Module] = None):
        if embedding_net is None:
            embedding_net = LSTMembedding(input_dim=input_dim).to(self.device) if use_lstm else nn.Identity()
        return posterior_nn(model=self.de_type, embedding_net=embedding_net)


    def _array_to_pyabc_dict(self, values, prefix: str = "data"):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2 and values.shape[1] > 1:
            return {f"{prefix}{idx + 1}": values[:, idx] for idx in range(values.shape[1])}
        if values.ndim == 3 and values.shape[-1] > 1:
            values = values.reshape(values.shape[0], -1)
            return {f"{prefix}{idx + 1}": values[:, idx] for idx in range(values.shape[1])}
        return {prefix: values}


    def _make_pyabc_simulator(self, simulator_func: Callable, simulator_kwargs: Optional[Dict] = None):
        simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs

        def wrapped_simulator(theta):
            sim_output = simulator_func(theta, **simulator_kwargs)
            return self._obs_to_pyabc_dict(sim_output)

        return wrapped_simulator

    def run_abc(self, obs_data, prior, simulator_func: Callable,
        num_simulations: int = 1000, population_size: int = 100, num_samples: int = 10000,
        eps_alpha: float = 0.2, transition_scaling: float = 0.5, simulator_kwargs: Optional[Dict] = None,
        resample_seed: int = 0):
        print("[*] Running SMC-ABC...")

        param_names = self._param_names(prior)
        discrete_param_names = self._discrete_param_names(prior)
        prior = prior.pyabc if hasattr(prior, "pyabc") else prior
        distance = pyabc.AdaptivePNormDistance(p=2, scale_function=pyabc.distance.std, all_particles_for_scale=True)
        obs_dict = self._obs_to_pyabc_dict(obs_data)
        # print(obs_dict)
        abc_simulator = self._make_pyabc_simulator(simulator_func, simulator_kwargs)
        # print(abc_simulator)
        
        eps = pyabc.QuantileEpsilon(initial_epsilon="from_sample", alpha=eps_alpha)
        transition = pyabc.MultivariateNormalTransition(scaling=transition_scaling)
        
        abc = pyabc.ABCSMC(abc_simulator, prior, distance, eps=eps, transitions=transition, population_size=population_size)
        db_path = "sqlite:///" + tempfile.mkstemp(suffix=".db")[1]
        abc.new(db_path, obs_dict)
        
        history = abc.run(max_total_nr_simulations=num_simulations)
        
        df, weights = history.get_distribution()
        df = self._reorder_parameter_frame(df, param_names)
        if discrete_param_names:
            samples = self._resample_parameter_particles(df, weights, num_samples, random_state=resample_seed)
            for name in discrete_param_names:
                if name in samples:
                    samples[name] = samples[name].round().astype(int)
        else:
            kde = pyabc.transition.MultivariateNormalTransition()
            kde.fit(df, weights)
            samples = kde.rvs(num_samples)
            samples = self._reorder_parameter_frame(samples, param_names)
        
        return history, samples

    def run_npe(self, obs_data, prior=None, thetas=None, xs=None,
        num_simulations: Optional[int] = None, use_lstm: bool = False, input_dim: int = 1,
        learning_rate: float = 1e-3, num_samples: int = 10000, batch_size: Optional[int] = None,
        embedding_net: Optional[nn.Module] = None, simulator_kwargs: Optional[Dict] = None,
        show_train_summary: bool = True):
        batch_size = self.batch_size if batch_size is None else batch_size
        prior = prior.sbi if hasattr(prior, "sbi") else prior
        print(f"[*] Running NPE (use_lstm={use_lstm}) with batch size {batch_size}...")

        thetas = self._to_tensor(thetas)
        xs = self._to_tensor(xs)
        x_obs = self._obs_to_tensor(obs_data)
        x_transform = None

        neural_net = self._get_neural_net(use_lstm=use_lstm, input_dim=input_dim, embedding_net=embedding_net)
        inference = NPE(prior=prior, density_estimator=neural_net, device=self.device)
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=batch_size, learning_rate=learning_rate, show_train_summary=show_train_summary
        )
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((num_samples,), x=x_obs)
        return posterior, samples.detach().cpu().numpy()

    def run_npe_lstm(self, obs_data, prior=None, thetas=None, xs=None, input_dim: int = 1, learning_rate: float = 1e-3,
        num_samples: int = 10000, batch_size: Optional[int] = None, normalize_x: bool = False, show_train_summary: bool = True,):
        
        return self.run_npe(
            obs_data=obs_data, prior=prior, thetas=thetas, xs=xs, use_lstm=True, input_dim=input_dim, learning_rate=learning_rate,
            num_samples=num_samples, batch_size=batch_size, show_train_summary=show_train_summary)

    def run_nre(self, obs_data, prior=None, thetas=None, xs=None,
        learning_rate: float = 1e-3, num_samples: Optional[int] = None, batch_size: Optional[int] = None,
        classifier=None, show_train_summary: bool = True):
        batch_size = self.batch_size if batch_size is None else batch_size
        prior = prior.sbi if hasattr(prior, "sbi") else prior
        print(f"[*] Running NRE with batch size {batch_size}...")

        thetas = self._to_tensor(thetas)
        xs = self._to_tensor(xs)
        x_obs = self._obs_to_tensor(obs_data)

        inference_kwargs = {"prior": prior, "device": self.device}
        if classifier is not None:
            inference_kwargs["classifier"] = classifier
        inference = NRE(**inference_kwargs)
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=batch_size, learning_rate=learning_rate, show_train_summary=show_train_summary
        )
        posterior = inference.build_posterior(density_estimator)
        result = {"posterior": posterior, "density_estimator": density_estimator, "x_obs": x_obs}
        if num_samples is not None and num_samples > 0:
            result["samples"] = posterior.sample((num_samples,), x=x_obs)
        return result

    
    def run_pnpe(self, obs_data, prior, simulator_func: Callable,
        num_simulations: int = 10000, num_samples: int = 10000, batch_size: Optional[int] = None,
        population_size: int = 100, abc_fraction: float = 0.5, use_lstm: bool = False, input_dim: int = 1,
        learning_rate: float = 1e-3, normalize_x: bool = True, show_train_summary: bool = True, simulator_kwargs: Optional[Dict] = None):

        
        batch_size = self.batch_size if batch_size is None else batch_size
        abc_sims = int(num_simulations * abc_fraction)
        
        print("[*] PNPE Stage 1: ABC preconditioning...")

        _, refined_thetas = self.run_abc(
            obs_data=obs_data, prior=prior, simulator_func=simulator_func,
            num_simulations=abc_sims, population_size=population_size, num_samples=num_simulations - abc_sims, simulator_kwargs=simulator_kwargs
        )
        
        print("[*] PNPE Stage 2: Training NPE with preconditioned samples...")

        
        theta_values = refined_thetas.values if isinstance(refined_thetas, pd.DataFrame) else np.asarray(refined_thetas)
        thetas = torch.as_tensor(theta_values, dtype=torch.float32, device=self.device)
        xs = []

        for _, theta in refined_thetas.iterrows() if isinstance(refined_thetas, pd.DataFrame) else enumerate(theta_values):
            x = simulator_func(theta, **simulator_kwargs)
            x = self._sim_to_array(x)
            xs.append(torch.as_tensor(x, dtype=torch.float32))

        xs = torch.stack(xs).to(self.device)
        print(xs.shape)
        print(obs_data.shape)
        result = self.run_npe(
            obs_data=obs_data, prior=prior, thetas=thetas, xs=xs.squeeze(-1), use_lstm=use_lstm, input_dim=input_dim,
            learning_rate=learning_rate, num_samples=num_samples, batch_size=batch_size, show_train_summary=show_train_summary,
        )

        return result

    def sample_posterior(self, posterior, obs_data, num_samples: int = 10000, x_transform: Optional[Dict] = None):
        x_obs = self._obs_to_tensor(obs_data)
        if x_transform is not None:
            mean_xs = x_transform["mean"].to(self.device)
            std_xs = x_transform["std"].to(self.device)
            x_obs = (x_obs - mean_xs.squeeze(0)) / std_xs.squeeze(0)
        return posterior.sample((num_samples,), x=x_obs)