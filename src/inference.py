import torch
import pyabc
import tempfile
import numpy as np
import pandas as pd
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from .embedding import LSTMembedding

class SBIEngine:
    """
    Unified Inference Engine for Epidemic Models.
    Provides three main methods: run_abc, run_npe, and run_pnpe.
    """
    def __init__(self, density_estimator='maf', device='cpu', batch_size=256):
        """
        Initialize the inference engine.
        
        Args:
            density_estimator (str): Type of flow-based model ('maf' or 'nsf'). Default is 'maf'.
            device (str): Device for neural network training ('cpu' or 'cuda').
            batch_size (int): Batch size for NPE training.
        """
        self.de_type = density_estimator
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.posterior = None

    def _get_neural_net(self, use_embedding=False):
        """Builds the neural posterior architecture (MAF/NSF) with optional LSTM embedding."""
        embedding_net = LSTMembedding().to(self.device) if use_embedding else None
        return posterior_nn(
            model=self.de_type, 
            embedding_net=embedding_net,
            device=self.device.type
        )

    def run_abc(self, obs_data, prior, simulator_func, num_simulations=10000, population_size=100):
        """
        Runs Approximate Bayesian Computation (ABC) with SMC.
        
        Args:
            obs_data (dict): Observed data in dictionary format (e.g., {"data": array}).
            prior: pyabc.Distribution object.
            simulator_func: Simulator function returning a dictionary.
            num_simulations (int): Total simulation budget.
            population_size (int): Size of the ABC population.
        """
        print(f"[*] Running ABC-SMC...")
        
        # Configure Epsilon and Transition as requested
        eps = pyabc.QuantileEpsilon(initial_epsilon='from_sample', alpha=0.2)
        transition = pyabc.MultivariateNormalTransition(scaling=0.5)
        
        abc = pyabc.ABCSMC(
            simulator_func, 
            prior, 
            eps=eps, 
            transitions=transition,
            population_size=population_size
        )
        
        db_path = "sqlite:///" + tempfile.mkstemp(suffix=".db")[1]
        abc.new(db_path, obs_data)
        
        history = abc.run(max_total_nr_simulations=num_simulations)
        return history

    def run_npe(self, prior, thetas, xs, use_lstm=True, learning_rate=0.0005):
        """
        Runs Neural Posterior Estimation (NPE).
        
        Args:
            prior: sbi prior object.
            thetas (Tensor): Simulated parameters.
            xs (Tensor): Simulated trajectories.
            use_lstm (bool): Whether to use LSTM embedding (NPE-LSTM).
            learning_rate (float): Optimizer learning rate.
        """
        print(f"[*] Running NPE (use_lstm={use_lstm}) with batch size {self.batch_size}...")
        
        neural_net = self._get_neural_net(use_embedding=use_lstm)
        inference = NPE(prior=prior, density_estimator=neural_net, device=self.device.type)
        
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=self.batch_size,
            learning_rate=learning_rate,
            show_train_summary=True
        )
        
        self.posterior = inference.build_posterior(density_estimator)
        return self.posterior

    def run_pnpe(self, obs_data, sbi_prior, pyabc_prior, simulator_func, abc_sims=5000):
        """
        Runs Preconditioned Neural Posterior Estimation (PNPE).
        Stage 1: ABC-SMC preconditioning to narrow parameter space.
        Stage 2: Training NPE on the refined region.
        """
        print(f"[*] PNPE Stage 1: ABC Preconditioning...")
        
        # Step 1: Rapid ABC-SMC to find the high-probability region
        abc_history = self.run_abc(
            {"data": obs_data}, 
            pyabc_prior, 
            simulator_func, 
            num_simulations=abc_sims
        )
        
        df, weights = abc_history.get_distribution()
        kde = pyabc.transition.MultivariateNormalTransition()
        kde.fit(df, weights)
        refined_thetas = kde.rvs(abc_sims)
        
        print(f"[*] PNPE Stage 2: Training NPE with preconditioned samples...")
        thetas_t = torch.tensor(refined_thetas.values, dtype=torch.float32)
        
        # Generate simulations for the refined parameters
        xs_list = []
        for p in refined_thetas.values:
            sim_res = simulator_func(p)
            xs_list.append(torch.tensor(sim_res["data"], dtype=torch.float32))
        xs_t = torch.stack(xs_list)

        # Step 2: Train NPE (NPE-LSTM is standard for PNPE)
        return self.run_npe(sbi_prior, thetas_t, xs_t, use_lstm=True)