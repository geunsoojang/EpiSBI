# Simulation-Based Inference for Epidemic Models
**Data and Code Repository for the Manuscript:**  
*A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*  


## Inference Algorithms & Toolkits

We implement and compare four distinct inference methods. Each is integrated with standard libraries to ensure robustness and reproducibility.
| Algorithm | Method Class | Core Toolkit | Description |
| :--- | :--- | :--- | :--- |
| **ABC** | Rejection-based | [**pyABC**](https://github.com/icb-dcm/pyabc) | Sequential Monte Carlo (SMC-ABC) for high-performance approximate Bayesian inference. |
| **NPE** | Neural Inference | [**sbi**](https://github.com/sbi-dev/sbi) | Neural Posterior Estimation using Normalizing Flows to learn the posterior $p(\theta|x)$ directly. |
| **NPE-LSTM** | Neural Inference | [**sbi**](https://github.com/sbi-dev/sbi) | NPE with an LSTM embedding network to automatically extract features from time-series trajectories. |
| **PNPE** | Neural Inference | Custom / **sbi** | Preconditioned NPE designed to improve convergence in models with complex identifiability constraints. |

## Epidemic Models
The repository includes four compartmental models of varying complexity to benchmark the inference algorithms: 
1. **Model 1: SEIR Model** – Standard Susceptible-Exposed-Infectious-Recovered dynamics. (Brauer et al. [Mathematical models in epidemiology](https://link.springer.com/book/10.1007/978-1-4939-9828-9))
2. **Model 2: SEIAR Model** – Includes an Asymptomatic infectious class. (Chowell et al. [Comparative estimation of the reproduction number for pandemic influenza from daily case notification data](https://royalsocietypublishing.org/rsif/article/4/12/155/65217/Comparative-estimation-of-the-reproduction-number))
3. **Model 3: Ebola Model** – A specialized model tailored to the transmission dynamics of the Ebola virus. (Legrand et al. [Understanding the dynamics of Ebola epidemics](https://pmc.ncbi.nlm.nih.gov/articles/PMC2870608/))
4. **Model 4: SIRTEM Model** – A complex model considering multiple transmission routes and immunity states. (Azad et al. [SIRTEM: Spatially Informed Rapid Testing for Epidemic Modeling and Response to COVID-19](https://dl.acm.org/doi/10.1145/3555310))

## Citation

