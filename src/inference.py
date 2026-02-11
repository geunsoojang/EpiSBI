import torch
import torch.nn as nn
import pyabc
import tempfile
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from .embedding import LSTMembedding
from .distance import euclidean_distance


class SBIEngine:
    """
    Unified Inference Engine for Epidemic Models.
    Provides three main methods: run_abc, run_npe, and run_pnpe.
    """
    def __init__(self, density_estimator='maf', device='cpu', batch_size=256):
        """
        Initialize the inference engine.
        Args:
            density_estimator (str): Type of flow-based model ('maf' or 'nsf').
            device (str): Device for neural network training ('cpu' or 'cuda').
            batch_size (int): Batch size for NPE training.
            low (list): Lower bounds for uniform prior.
            high (list): Upper bounds for uniform prior.
        """
        self.de_type = density_estimator
        self.device = device
        self.batch_size = batch_size
        
    def _get_neural_net(self, use_embedding=False, input_dim=1):
        """Builds the neural posterior architecture (MAF/NSF) with optional
        LSTM embedding."""
        embedding_net = (LSTMembedding(input_dim=input_dim).to(self.device)
                         if use_embedding else nn.Identity())
        return posterior_nn(
            model=self.de_type,
            embedding_net=embedding_net
        )

    def run_abc(self, obs_data, prior, simulator_func, distance=None,
                num_simulations=10000, population_size=1000,
                num_samples=10000):
        """
        Runs Approximate Bayesian Computation (ABC) with SMC.
        Args:
            obs_data (dict): Observed data in dictionary format
            (e.g.,{"data":array}).
            prior: pyabc.Distribution object.
            simulator_func: Simulator function returning a dictionary.
            num_simulations (int): Total simulation budget.
            population_size (int): Size of the ABC population.
            num_samples (int): Number of samples to draw from the posterior.
        """
        print("[*] Running SMC-ABC...")
        if distance is None:
            distance = euclidean_distance

        if not isinstance(obs_data, dict):
            obs_data = {"data": obs_data}
        
        def simulator_pyabc(x):
            return {"data": simulator_func(x)}
        
        # Configure Epsilon and Transition as requested
        eps = pyabc.QuantileEpsilon(initial_epsilon='from_sample', alpha=0.2)
        transition = pyabc.MultivariateNormalTransition(scaling=0.5)
        
        abc = pyabc.ABCSMC(
            simulator_pyabc,
            prior,
            distance,
            eps=eps,
            transitions=transition,
            population_size=population_size
        )
        
        db_path = "sqlite:///" + tempfile.mkstemp(suffix=".db")[1]
        abc.new(db_path, obs_data)
        
        history = abc.run(max_total_nr_simulations=num_simulations)
        
        # Draw samples from the posterior distribution
        df, weights = history.get_distribution()
        kde = pyabc.transition.MultivariateNormalTransition()
        kde.fit(df, weights)
        
        return kde.rvs(num_samples)

    def run_npe(self, obs_data, prior=None, thetas=None, xs=None,
                use_lstm=False, input_dim=1,
                learning_rate=0.001, num_samples=10000, batch_size=256):
        """
        Runs Neural Posterior Estimation (NPE).
        Args:
            prior: sbi prior object.
            thetas (Tensor): Simulated parameters.
            xs (Tensor): Simulated trajectories.
            use_lstm (bool): Whether to use LSTM embedding (NPE-LSTM).
            learning_rate (float): Optimizer learning rate.
            num_samples (int): Number of samples to draw from the posterior.
            batch_size (int or None): Training batch size.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        print(f"[*] Running NPE (use_lstm={use_lstm}) with batch size "
              f"{batch_size}...")
        
        # 1. Handle Observation Data (Convert to Tensor)
        if isinstance(obs_data, dict):
            x_obs = torch.tensor(obs_data["data"], dtype=torch.float32).to(self.device)
        else:
            x_obs = torch.tensor(obs_data, dtype=torch.float32).to(self.device)
            
        # 2. Normalization Logic (Z-score)
        if xs.dim() >= 2:
            # Calculate stats along the batch and sequence dimensions
            mean_xs = xs.mean(dim=0, keepdim=True)
            std_xs = xs.std(dim=0, keepdim=True) + 1e-6
            xs = (xs - mean_xs) / std_xs
            x_obs = (x_obs - mean_xs.squeeze(0)) / std_xs.squeeze(0)
                      
        # 3. Setup and Train
        neural_net = self._get_neural_net(use_embedding=use_lstm,
                                          input_dim=input_dim)
        inference = NPE(prior=prior, density_estimator=neural_net,
                        device=self.device)
        
        density_estimator = inference.append_simulations(thetas, xs).train(
            training_batch_size=batch_size,
            learning_rate=learning_rate,
            show_train_summary=True
        )
        
        posterior = inference.build_posterior(density_estimator)
        samples = posterior.sample((num_samples,), x=x_obs)
        
        return posterior, samples

    def run_pnpe(self, obs_data, pyabc_prior, sbi_prior, simulator_func,
                 num_simulations=10000, num_samples=10000, batch_size=256):
        """
        Runs Preconditioned Neural Posterior Estimation (PNPE).
        Stage 1: ABC-SMC preconditioning to narrow parameter space.
        Stage 2: Training NPE on the refined region.
        """
        print("[*] PNPE Stage 1: ABC Preconditioning...")
        
        # Step 1: Rapid ABC-SMC to find the high-probability region
        refined_thetas = self.run_abc(
            obs_data,
            pyabc_prior,
            simulator_func,
            num_simulations=num_simulations//2,
            population_size=100,
            num_samples=num_simulations//2
        )        
    
        print("[*] PNPE Stage 2: Training NPE with preconditioned samples...")

        thetas_t = torch.tensor(refined_thetas.values, dtype=torch.float32)
        
        # Generate simulations for the refined parameters
        xs_list = []
        for i, p in refined_thetas.iterrows():
            sim_res = simulator_func(p)
            xs_list.append(torch.tensor(sim_res, dtype=torch.float32))
        xs_t = torch.stack(xs_list)

        # Step 2: Train NPE (NPE-LSTM is standard for PNPE)
        _, samples = self.run_npe(obs_data, sbi_prior, thetas_t, xs_t,
                                  use_lstm=True, num_samples=num_samples,
                                  batch_size=batch_size)
        
        return samples
