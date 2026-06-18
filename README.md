# EpiSBI

Simulation-based inference resources for epidemic models.

This repository has two related purposes:

1. It provides code and data associated with the paper *A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*.
2. It provides `episbi`, a reusable Python toolbox for simulation-based inference workflows with epidemic and compartmental models.

## Paper Repository

### Manuscript

This repository accompanies:

Jang G, Candan KS, Chowell G. A comparative study of simulation-based inference methods for epidemic models with identifiability considerations. *PLOS Computational Biology*. 2026;22(6):e1014364.

### Paper Goal

The paper compares likelihood-free and neural simulation-based inference approaches for epidemic models, with emphasis on how model structure and parameter identifiability affect posterior inference.

### Inference Methods Compared

| Method | Class | Toolkit | Description |
| --- | --- | --- | --- |
| ABC | Approximate Bayesian computation | `pyabc` | Sequential Monte Carlo ABC for likelihood-free posterior inference. |
| NPE | Neural posterior estimation | `sbi` | Learns the posterior distribution `p(theta | x)` from simulated parameter-observation pairs. |
| NPE-LSTM | Neural posterior estimation | `sbi` + PyTorch | Adds an LSTM embedding network for time-series epidemic observations. |
| PNPE | Preconditioned neural posterior estimation | `pyabc` + `sbi` | Uses ABC-based preconditioning before neural posterior estimation. |

### Epidemic Models

The manuscript materials include compartmental epidemic-model examples such as:

- SEIR model for standard susceptible-exposed-infectious-recovered dynamics.
- Stochastic SEIR model for event-based epidemic simulation.
- Extended stochastic SE1E2E3IR and SEIRD-style models for richer latent-stage or outcome structure.

### Paper Materials

| Path | Contents |
| --- | --- |
| `data/` | Example data used by the original manuscript workflows. |
| `notebooks/` | Manuscript-oriented notebooks and exploratory analyses. |
| `temp_backup/` | Supporting source files from the earlier paper-code organization. |

## EpiSBI Toolbox

### Overview

`episbi` is a lightweight toolbox for building simulation-based inference workflows around user-defined epidemic simulators. It is designed to keep model simulation, prior definition, inference, posterior predictive evaluation, and tutorial examples in separate, reusable pieces.

### Main Features

- `SBIEngine` for ABC, NPE, NPE-LSTM, NRE, and PNPE workflows.
- `LSTMembedding` for neural inference with time-series observations.
- Prior helpers for `pyabc` and `sbi` workflows.
- `simulate_for_sbi` and `sample_prior` utilities for generating simulation datasets.
- Compartment-model builders for deterministic and stochastic models.
- Built-in model examples including deterministic SEIR, stochastic SEIR, stochastic SE1E2E3IR, and stochastic SEIRD.
- Posterior predictive evaluation metrics including MAE, RMSE, 95% interval coverage, interval score, and weighted interval score.
- Plotting utilities for inference and forecast windows.

### Installation

Clone the repository:

```bash
git clone https://github.com/geunsoojang/EpiSBI.git
cd EpiSBI
```

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks and scripts from the repository root so Python can find the local `episbi` package.

### Quick Start

```python
import numpy as np

from episbi import SBIEngine, simulate_for_sbi
from episbi.metric import evaluate_prediction_windows
from episbi.models import Transition, compartment_model

engine = SBIEngine(device="cpu", batch_size=256)

model = compartment_model(
    compartments=["S", "E", "I", "R"],
    transitions=[
        Transition("S", "E", "beta * S * I / N", name="S_to_E"),
        Transition("E", "I", "sigma * E", name="E_to_I"),
        Transition("I", "R", "gamma * I", name="I_to_R"),
    ],
    param_names=["beta", "sigma", "gamma"],
    population=1000,
    initial_conditions={"S": 990, "E": 5, "I": 5, "R": 0},
    observed={"transitions": ["S_to_E"]},
    model_type="deterministic",
)

# Example simulator call.
sim = model.simulate(
    {"beta": 0.35, "sigma": 0.2, "gamma": 0.1},
    total_days=100,
)

# Posterior predictive trajectories can be evaluated in split windows.
metrics = evaluate_prediction_windows(
    y_obs=np.zeros((100, 1)),
    prediction=np.zeros((200, 100, 1)),
    inference_days=90,
    forecast_days=10,
    output_names=["incidence"],
)

print(sim["observed"].shape)
print(metrics)
```

### Tutorials

The `tutorials/` directory contains notebook examples organized by model type and inference method.

| Notebook | Purpose |
| --- | --- |
| `01-1_Deterministic_seir_model.ipynb` | Build and simulate a deterministic SEIR model. |
| `01-2_Stochastic_seir_model.ipynb` | Build and simulate a stochastic SEIR model. |
| `02-1_ABC_deterministic_seir.ipynb` | ABC inference for deterministic SEIR. |
| `02-2_NPE_deterministic_seir.ipynb` | NPE inference for deterministic SEIR. |
| `02-3_NPE_LSTM_deterministic_seir.ipynb` | NPE-LSTM inference for deterministic SEIR. |
| `02-4_PNPE_deterministic_seir.ipynb` | PNPE inference for deterministic SEIR. |
| `03-1_ABC_stochastic_seir.ipynb` | ABC inference for stochastic SEIR. |
| `03-2_NPE_stochastic_seir.ipynb` | NPE inference for stochastic SEIR. |
| `03-3_NPE_LSTM_stochastic_seir.ipynb` | NPE-LSTM inference for stochastic SEIR. |
| `03-4_PNPE_stochastic_seir.ipynb` | PNPE inference for stochastic SEIR. |

Start Jupyter from the repository root:

```bash
jupyter notebook
```

### Repository Structure

```text
EpiSBI/
|-- data/            # paper data
|-- notebooks/       # paper and exploratory notebooks
|-- temp_backup/     # earlier manuscript-code organization
|-- episbi/          # reusable toolbox package
|-- tutorials/       # toolbox tutorials
|-- README.md
`-- requirements.txt
```

## Citation

If you use the paper materials or the toolbox, please cite:

Jang G, Candan KS, Chowell G. A comparative study of simulation-based inference methods for epidemic models with identifiability considerations. *PLOS Computational Biology*. 2026;22(6):e1014364.
