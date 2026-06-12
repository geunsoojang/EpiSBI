# EpiSBI

Simulation-based inference tools and tutorials for epidemic models.

This repository accompanies the paper "*A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*" and also provides a reusable Python package, `episbi`, for running and evaluating simulation-based inference workflows.

## What Is Included

- `episbi/`: a lightweight toolbox for epidemic-model SBI workflows.
- `tutorials/`: Jupyter notebooks showing deterministic SEIR simulation and inference workflows.
- `data/`, `notebooks/`, `temp_backup/`: supporting materials from the original manuscript repository.

## Inference Methods

The package and tutorials focus on four inference approaches:

| Method | Type | Toolkit | Description |
| --- | --- | --- | --- |
| ABC | Approximate Bayesian computation | `pyabc` | Sequential Monte Carlo ABC for likelihood-free inference. |
| NPE | Neural posterior estimation | `sbi` | Learns the posterior distribution `p(theta | x)` from simulations. |
| NPE-LSTM | Neural posterior estimation | `sbi` + PyTorch | Uses an LSTM embedding network for time-series observations. |
| PNPE | Preconditioned NPE | `pyabc` + `sbi` | Uses ABC samples to precondition NPE training. |

## Package Features

`episbi` currently provides:

- `SBIEngine` for running ABC, NPE, NPE-LSTM, and PNPE workflows.
- `LSTMembedding` for time-series embedding in neural posterior estimation.
- `simulate_for_sbi` and `sample_prior` helpers for generating simulation datasets.
- Posterior predictive evaluation utilities, including MAE, RMSE, 95% interval coverage, interval score, and weighted interval score.
- `plot_prediction_windows` for visualizing inference and forecast windows.

## Installation

Clone the repository:

```bash
git clone https://github.com/geunsoojang/EpiSBI.git
cd EpiSBI
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Because this repository is organized as source code rather than a packaged PyPI release, run notebooks and scripts from the repository root, or add the repository root to your Python path.

## Quick Start

```python
import numpy as np

from episbi import SBIEngine, simulate_for_sbi
from episbi.metric import evaluate_prediction_windows
from episbi.utils import plot_prediction_windows

engine = SBIEngine(device="cpu", batch_size=256)

# Generate simulations from a user-defined prior and simulator.
thetas, xs = simulate_for_sbi(
     prior=prior,
     simulator=simulator,
     num_simulations=1000,
     total_days=100,
)

# Fit NPE when simulated parameters and trajectories are available.
posterior, samples = engine.run_npe(
     obs_data=x_obs,
     prior=prior,
     thetas=thetas,
     xs=xs,
     num_samples=10000,
)

# Evaluate externally generated posterior predictive trajectories.
# prediction shape: (n_samples, time, n_outputs)
metrics = evaluate_prediction_windows(
    y_obs=np.zeros((100, 1)),
    prediction=np.zeros((200, 100, 1)),
    inference_days=90,
    forecast_days=10,
    output_names=["incidence"],
)

print(metrics)
```

## Tutorials

The `tutorials/` directory contains example notebooks:

| Notebook | Purpose |
| --- | --- |
| `01_deterministic_seir.ipynb` | Deterministic SEIR simulation workflow. |
| `02_ABC_seir.ipynb` | ABC inference for the SEIR model. |
| `02_npe_seir.ipynb` | Neural posterior estimation for SEIR. |
| `02_npe_LSTM_seir.ipynb` | NPE with an LSTM embedding network. |
| `02_pnpe_seir.ipynb` | Preconditioned NPE workflow. |
| `02_sbi_seir.ipynb` | Combined SBI example workflow. |

Start Jupyter from the repository root so notebook imports can find `episbi`:

```bash
jupyter notebook
```

## Repository Structure

```text
EpiSBI/
|-- episbi/          # reusable SBI toolbox code
|-- tutorials/       # tutorial notebooks
|-- notebooks/       # manuscript and exploratory notebooks
|-- data/            # example data
|-- temp_backup/     # original supporting source files
|-- README.md
`-- requirements.txt
```

## Citation

If you use this repository, please cite:

Jang G, Candan KS, Chowell G. A comparative study of simulation-based inference methods for epidemic models with identifiability considerations. *PLOS Computational Biology*. 2026;22(6):e1014364.
