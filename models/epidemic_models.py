import numpy as np
from scipy.integrate import odeint

def seir_ode(y, t, beta, kappa, gamma, N):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - kappa * E
    dIdt = kappa * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def simulate_seir(params, ic, duration, noise=True):
    beta, kappa, gamma = params
    N = sum(ic)
    t = np.linspace(0, duration, duration + 1)
    sol = odeint(seir_ode, ic, t, args=(beta, kappa, gamma, N))
    
    # New infections (incidence)
    incidence = kappa * sol[:-1, 1] 
    if noise:
        incidence = np.random.poisson(np.maximum(incidence, 1e-6))
    return incidence