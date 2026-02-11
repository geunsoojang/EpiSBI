# Simulation-Based Inference for Epidemic Models
**Data and Code Repository for the Manuscript:**  
*A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*  


## Inference Algorithms & Toolkits

We implement and compare four distinct inference methods. Each is integrated with standard libraries to ensure robustness and reproducibility.
| Algorithm | Method Class | Core Toolkit | Description |
| :--- | :--- | :--- | :--- |
| **ABC** | Rejection-based | [**pyABC**](https://github.com/icb-dcm/pyabc) | Sequential Monte Carlo (SMC-ABC) for high-performance approximate Bayesian inference. |
| **NPE** | Neural Inference | [**sbi**](https://github.com/sbi-dev/sbi) | Neural Posterior Estimation using Normalizing Flows to learn the posterior $p(\theta|x)$ directly. |
| **NPE-LSTM** | Neural Inference | [**sbi**](https://github.com/sbi-dev/sbi) | NPE with an **LSTM embedding network** to automatically extract features from time-series trajectories. |
| **PNPE** | Neural Inference | Custom / **sbi** | **Preconditioned** NPE designed to improve convergence in models with complex identifiability constraints. |

## Epidemic Models
The repository includes four compartmental models of varying complexity to benchmark the inference algorithms:
1. **Model 1: SEIR Model** – Standard Susceptible-Exposed-Infectious-Recovered dynamics.
2. **Model 2: SEIAR Model** – Includes an **Asymptomatic** infectious class.
3. **Model 3: Ebola Model** – A specialized model tailored to the transmission dynamics of the Ebola virus.
4. **Model 4: SIRTEM Model** – A complex model considering multiple transmission routes and immunity states.


