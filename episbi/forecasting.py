from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_numpy(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected shape (time,) or (time, n_outputs), got {x.shape}.")


def _sim_to_array(sim_output):
    if isinstance(sim_output, dict):
        key = list(sim_output.keys())[0]
        sim_output = sim_output[key]
    return _to_numpy(sim_output)


def _unwrap_samples_container(posterior_samples):
    if isinstance(posterior_samples, dict) and "samples" in posterior_samples:
        return posterior_samples["samples"]
    if isinstance(posterior_samples, tuple) and len(posterior_samples) > 0:
        return posterior_samples[0]
    return posterior_samples


def _prepare_samples(posterior_samples, param_names: Optional[Sequence[str]] = None):
    posterior_samples = _unwrap_samples_container(posterior_samples)
    if isinstance(posterior_samples, pd.DataFrame):
        if param_names is not None:
            missing = [p for p in param_names if p not in posterior_samples.columns]
            if missing:
                raise ValueError(f"Missing parameters: {missing}")
            samples = posterior_samples[list(param_names)].to_numpy()
        else:
            samples = posterior_samples.to_numpy()
    else:
        samples = _to_numpy(posterior_samples)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"posterior_samples must have shape (n_samples, n_params), got {samples.shape}.")
    return samples


def _prepare_simulator_inputs(
    posterior_samples,
    samples: np.ndarray,
    param_names: Optional[Sequence[str]] = None,
    theta_transform: Optional[Callable] = None,
):
    if theta_transform is not None:
        return [theta_transform(theta) for theta in samples]

    posterior_samples = _unwrap_samples_container(posterior_samples)
    if isinstance(posterior_samples, pd.DataFrame):
        cols = list(param_names) if param_names is not None else list(posterior_samples.columns)
        missing = [p for p in cols if p not in posterior_samples.columns]
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        return [row.to_dict() for _, row in posterior_samples[cols].iterrows()]

    if param_names is not None:
        names = list(param_names)
        if len(names) != samples.shape[1]:
            raise ValueError(f"param_names must have length {samples.shape[1]}, got {len(names)}.")
        return [dict(zip(names, theta)) for theta in samples]

    return samples


def _call_simulator(simulator: Callable, theta, total_days: int, simulator_kwargs: Optional[Dict] = None):
    simulator_kwargs = {} if simulator_kwargs is None else simulator_kwargs
    try:
        return simulator(theta, total_days=total_days, **simulator_kwargs)
    except TypeError as exc:
        try:
            return simulator(theta, T=total_days, **simulator_kwargs)
        except TypeError:
            try:
                return simulator(theta, total_days, **simulator_kwargs)
            except TypeError:
                raise exc


def _simulate_trajectories(
    theta_inputs,
    simulator: Callable,
    total_days: int,
    simulator_kwargs: Optional[Dict] = None,
    observation_noise: Optional[Callable] = None,
):
    sims = []
    for theta in theta_inputs:
        sim = _call_simulator(simulator, theta, total_days=total_days, simulator_kwargs=simulator_kwargs)
        if observation_noise is not None:
            sim = observation_noise(sim)
        sim = _ensure_2d(_sim_to_array(sim))
        if sim.shape[0] < total_days:
            raise ValueError(f"Simulator output must have at least {total_days} time points, got {sim.shape[0]}.")
        sims.append(sim[:total_days])
    return np.stack(sims, axis=0)


def split_train_forecast(y_obs, inference_days: int = 90, forecast_days: int = 10):
    y_obs = _ensure_2d(_to_numpy(y_obs))
    total_days = inference_days + forecast_days
    if y_obs.shape[0] < total_days:
        raise ValueError(f"y_obs must contain at least {total_days} time points, got {y_obs.shape[0]}.")
    return y_obs[:inference_days], y_obs[inference_days:total_days]


def posterior_predictive_forecast(
    posterior_samples,
    simulator: Callable,
    inference_days: int = 90,
    forecast_days: int = 10,
    param_names: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = None,
    seed: int = 0,
    simulator_kwargs: Optional[Dict] = None,
    observation_noise: Optional[Callable] = None,
    theta_transform: Optional[Callable] = None,
):
    rng = np.random.default_rng(seed)
    samples = _prepare_samples(posterior_samples, param_names)

    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if n_samples < samples.shape[0]:
            sample_indices = rng.choice(samples.shape[0], size=n_samples, replace=False)
            posterior_samples = _unwrap_samples_container(posterior_samples)
            if isinstance(posterior_samples, pd.DataFrame):
                posterior_samples = posterior_samples.iloc[sample_indices].reset_index(drop=True)
            samples = samples[sample_indices]

    total_days = inference_days + forecast_days
    theta_inputs = _prepare_simulator_inputs(posterior_samples, samples, param_names, theta_transform)
    sims_full = _simulate_trajectories(theta_inputs, simulator, total_days, simulator_kwargs, observation_noise)
    return {
        "sims_full": sims_full,
        "sims_inference": sims_full[:, :inference_days, :],
        "sims_forecast": sims_full[:, inference_days:total_days, :],
    }


def summarize_trajectories(sims: np.ndarray, probs: Tuple[float, float, float] = (0.025, 0.5, 0.975)):
    sims = np.asarray(sims)
    if sims.ndim != 3:
        raise ValueError(f"sims must have shape (n_samples, time, n_outputs), got {sims.shape}.")
    return {
        "lower": np.quantile(sims, probs[0], axis=0),
        "median": np.quantile(sims, probs[1], axis=0),
        "upper": np.quantile(sims, probs[2], axis=0),
        "mean": np.mean(sims, axis=0),
    }


summarize_forecast = summarize_trajectories


def interval_score(y_true, lower, upper, alpha: float = 0.05):
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    score = upper - lower
    score += (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    score += (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return score


def evaluate_trajectory(y_true, summary, output_names: Optional[Sequence[str]] = None, alpha: float = 0.05):
    y_true = _ensure_2d(_to_numpy(y_true))
    median = _ensure_2d(summary["median"])
    lower = _ensure_2d(summary["lower"])
    upper = _ensure_2d(summary["upper"])
    if y_true.shape != median.shape:
        raise ValueError(f"y_true and median must have the same shape, got {y_true.shape} and {median.shape}.")

    n_outputs = y_true.shape[1]
    output_names = [f"output_{i}" for i in range(n_outputs)] if output_names is None else list(output_names)
    if len(output_names) != n_outputs:
        raise ValueError(f"output_names must have length {n_outputs}, got {len(output_names)}.")

    rows = []
    for j in range(n_outputs):
        error = median[:, j] - y_true[:, j]
        coverage = np.mean((y_true[:, j] >= lower[:, j]) & (y_true[:, j] <= upper[:, j]))
        score = interval_score(y_true[:, j], lower[:, j], upper[:, j], alpha)
        rows.append(
            {
                "output": output_names[j],
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "PI95_coverage": float(coverage),
                "interval_score": float(np.mean(score)),
            }
        )
    return pd.DataFrame(rows)


evaluate_forecast = evaluate_trajectory


def run_posterior_forecast_evaluation(
    posterior_samples,
    simulator: Callable,
    y_obs,
    inference_days: int = 90,
    forecast_days: int = 10,
    param_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = 1000,
    seed: int = 0,
    simulator_kwargs: Optional[Dict] = None,
    observation_noise: Optional[Callable] = None,
    theta_transform: Optional[Callable] = None,
    alpha: float = 0.05,
):
    y_inference, y_forecast = split_train_forecast(y_obs, inference_days, forecast_days)
    sims = posterior_predictive_forecast(
        posterior_samples=posterior_samples,
        simulator=simulator,
        inference_days=inference_days,
        forecast_days=forecast_days,
        param_names=param_names,
        n_samples=n_samples,
        seed=seed,
        simulator_kwargs=simulator_kwargs,
        observation_noise=observation_noise,
        theta_transform=theta_transform,
    )

    probs = (alpha / 2, 0.5, 1 - alpha / 2)
    inference_summary = summarize_trajectories(sims["sims_inference"], probs=probs)
    forecast_summary = summarize_trajectories(sims["sims_forecast"], probs=probs)
    inference_metrics = evaluate_trajectory(y_inference, inference_summary, output_names, alpha)
    forecast_metrics = evaluate_trajectory(y_forecast, forecast_summary, output_names, alpha)
    inference_metrics.insert(0, "window", "inference")
    forecast_metrics.insert(0, "window", "forecast")
    metrics = pd.concat([inference_metrics, forecast_metrics], ignore_index=True)

    return {
        "y_inference": y_inference,
        "y_forecast": y_forecast,
        "sims_full": sims["sims_full"],
        "sims_inference": sims["sims_inference"],
        "sims_forecast": sims["sims_forecast"],
        "inference_summary": inference_summary,
        "forecast_summary": forecast_summary,
        "inference_metrics": inference_metrics,
        "forecast_metrics": forecast_metrics,
        "metrics": metrics,
    }


def run_forecast_evaluation(*args, calibration_days: Optional[int] = None, inference_days: int = 90, **kwargs):
    if calibration_days is not None:
        inference_days = calibration_days
    return run_posterior_forecast_evaluation(*args, inference_days=inference_days, **kwargs)


def compare_forecast_windows(
    posterior_samples,
    simulator: Callable,
    y_obs,
    settings: Sequence[Tuple[int, int]],
    param_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = 1000,
    seed: int = 0,
    simulator_kwargs: Optional[Dict] = None,
    observation_noise: Optional[Callable] = None,
    theta_transform: Optional[Callable] = None,
    alpha: float = 0.05,
):
    rows = []
    for i, (inference_days, forecast_days) in enumerate(settings):
        result = run_posterior_forecast_evaluation(
            posterior_samples=posterior_samples,
            simulator=simulator,
            y_obs=y_obs,
            inference_days=inference_days,
            forecast_days=forecast_days,
            param_names=param_names,
            output_names=output_names,
            n_samples=n_samples,
            seed=seed + i,
            simulator_kwargs=simulator_kwargs,
            observation_noise=observation_noise,
            theta_transform=theta_transform,
            alpha=alpha,
        )
        metrics = result["metrics"].copy()
        metrics["inference_days"] = inference_days
        metrics["forecast_days"] = forecast_days
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def evaluate_multiple_cases(
    posterior_samples_list: Sequence,
    simulator: Callable,
    y_obs_list: Sequence,
    inference_days: int = 90,
    forecast_days: int = 10,
    param_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = 1000,
    seed: int = 0,
    simulator_kwargs: Optional[Dict] = None,
    observation_noise: Optional[Callable] = None,
    theta_transform: Optional[Callable] = None,
    alpha: float = 0.05,
):
    if len(posterior_samples_list) != len(y_obs_list):
        raise ValueError("posterior_samples_list and y_obs_list must have the same length.")

    all_metrics, all_results = [], []
    for case_id, posterior_samples in enumerate(posterior_samples_list):
        result = run_posterior_forecast_evaluation(
            posterior_samples=posterior_samples,
            simulator=simulator,
            y_obs=y_obs_list[case_id],
            inference_days=inference_days,
            forecast_days=forecast_days,
            param_names=param_names,
            output_names=output_names,
            n_samples=n_samples,
            seed=seed + case_id,
            simulator_kwargs=simulator_kwargs,
            observation_noise=observation_noise,
            theta_transform=theta_transform,
            alpha=alpha,
        )
        metrics = result["metrics"].copy()
        metrics["case_id"] = case_id
        metrics["inference_days"] = inference_days
        metrics["forecast_days"] = forecast_days
        all_metrics.append(metrics)
        all_results.append(result)
    return pd.concat(all_metrics, ignore_index=True), all_results


def plot_forecast_result(
    result,
    y_obs,
    inference_days: int = 90,
    forecast_days: int = 10,
    output_index: int = 0,
    title: Optional[str] = None,
    ylabel: str = "Incidence",
    show_mean: bool = False,
):
    y_obs = _ensure_2d(_to_numpy(y_obs))
    total_days = inference_days + forecast_days
    if y_obs.shape[0] < total_days:
        raise ValueError(f"y_obs must contain at least {total_days} time points, got {y_obs.shape[0]}.")

    inference_summary = result["inference_summary"]
    forecast_summary = result["forecast_summary"]
    days = np.arange(1, total_days + 1)
    inference_axis = days[:inference_days]
    forecast_axis = days[inference_days:total_days]

    plt.figure(figsize=(8, 4))
    plt.plot(inference_axis, y_obs[:inference_days, output_index], label="Inference data")
    plt.plot(forecast_axis, y_obs[inference_days:total_days, output_index], marker="o", label="Observed forecast")
    plt.plot(inference_axis, inference_summary["median"][:, output_index], label="Posterior median fit")
    plt.plot(forecast_axis, forecast_summary["median"][:, output_index], label="Posterior median forecast")
    if show_mean:
        plt.plot(inference_axis, inference_summary["mean"][:, output_index], linestyle=":", label="Posterior mean fit")
        plt.plot(forecast_axis, forecast_summary["mean"][:, output_index], linestyle=":", label="Posterior mean forecast")
    plt.fill_between(
        inference_axis,
        inference_summary["lower"][:, output_index],
        inference_summary["upper"][:, output_index],
        alpha=0.2,
        label="95% interval fit",
    )
    plt.fill_between(
        forecast_axis,
        forecast_summary["lower"][:, output_index],
        forecast_summary["upper"][:, output_index],
        alpha=0.25,
        label="95% interval forecast",
    )
    plt.axvline(inference_days, linestyle="--", linewidth=1)
    plt.xlabel("Day")
    plt.ylabel(ylabel)
    plt.title(title or f"{inference_days}-day inference and {forecast_days}-day forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_forecast_metrics(metrics: pd.DataFrame, filename: str):
    metrics.to_csv(filename, index=False)


def summarize_metrics_by_output(metrics: pd.DataFrame):
    metric_cols = ["MAE", "RMSE", "PI95_coverage", "interval_score"]
    group_cols = ["window", "output"] if "window" in metrics.columns else ["output"]
    if "inference_days" in metrics.columns:
        group_cols.append("inference_days")
    if "forecast_days" in metrics.columns:
        group_cols.append("forecast_days")
    return metrics.groupby(group_cols)[metric_cols].agg(["mean", "std", "median"]).reset_index()
