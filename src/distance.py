import numpy as np


def euclidean_distance(simulation, observation):
    """
    Standard Euclidean distance (L2 norm) between two trajectories.
    Suitable for most epidemic time-series data.
    Args:
        simulation (dict): Simulated data from the model
            (e.g., {"data": array}).
        observation (dict): Observed data (e.g., {"data": array}).
    """
    # Extract arrays from dictionaries (pyabc requirement)
    sim_data = simulation["data"]
    obs_data = observation["data"]
    # Calculate L2 norm
    return np.linalg.norm(sim_data - obs_data)


def manhattan_distance(simulation, observation):
    """
    Manhattan distance (L1 norm) between two trajectories.
    Can be useful when outliers are present in epidemic data.
    Args:
        simulation (dict): Simulated data from the model
            (e.g., {"data": array}).
        observation (dict): Observed data (e.g., {"data": array}).
    """
    # Extract arrays from dictionaries (pyabc requirement)
    sim_data = simulation["data"]
    obs_data = observation["data"]
    # Calculate L1 norm
    return np.sum(np.abs(sim_data - obs_data))