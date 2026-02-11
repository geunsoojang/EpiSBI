# Simulation-Based Inference for Epidemic Models
**Data and Code Repository for the Manuscript:**  
*A comparative study of simulation-based inference methods for epidemic models with identifiability considerations*  


This project implements and compares several inference frameworks to estimate parameters for epidemiological models where likelihoods are computationally intractable.🛠 Algorithm OverviewAlgorithmMethod ClassCore ToolkitDescriptionABCRejection-basedpyABCApproximate Bayesian Computation using Sequential Monte Carlo (SMC) for population-based sampling.NPENeural InferencesbiNeural Posterior Estimation using Normalizing Flows (e.g., NSF, MAF) to directly learn the posterior $p(\thetaNPE-LSTMNeural InferencesbiNPE enhanced with an LSTM-based embedding net to automatically extract summary statistics from time-series data.PNPENeural InferenceCustom / sbiPreconditioned Neural Posterior Estimation designed to improve convergence and accuracy under identifiability constraints.

## Epidemic Models

- Model1: SEIR (Susceptible-Exposed-Infectious-Recovered) model
- Model2: SEIAR model
- Model3: Ebola model
- Model4: SIRTEM model


