from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def to_numpy(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected shape (time,) or (time, n_outputs), got {x.shape}.")


def simulation_to_array(sim_output):
    if isinstance(sim_output, dict):
        key = list(sim_output.keys())[0]
        sim_output = sim_output[key]
    return to_numpy(sim_output)


def prediction_samples_to_array(prediction):
    if isinstance(prediction, (list, tuple)):
        prediction = [ensure_2d(simulation_to_array(sim)) for sim in prediction]
        return np.stack(prediction, axis=0)

    prediction = np.asarray(to_numpy(prediction), dtype=float)
    if prediction.ndim == 1:
        return prediction[None, :, None]
    if prediction.ndim == 2:
        return prediction[:, :, None]
    if prediction.ndim == 3:
        return prediction
    raise ValueError(
        "prediction must be a summary dict, a list of simulations, or an array with shape "
        "(n_samples, time, n_outputs), (n_samples, time), or (time,)."
    )


def split_train_forecast(y_obs, inference_days: int = 90, forecast_days: int = 10):
    y_obs = ensure_2d(to_numpy(y_obs))
    total_days = inference_days + forecast_days
    if y_obs.shape[0] < total_days:
        raise ValueError(f"y_obs must contain at least {total_days} time points, got {y_obs.shape[0]}.")
    return y_obs[:inference_days], y_obs[inference_days:total_days]


def summarize_trajectories(sims: np.ndarray, probs: Tuple[float, float, float] = (0.025, 0.5, 0.975)):
    sims = prediction_samples_to_array(sims)

    return {
        "lower": np.quantile(sims, probs[0], axis=0),
        "median": np.quantile(sims, probs[1], axis=0),
        "upper": np.quantile(sims, probs[2], axis=0),
        "mean": np.mean(sims, axis=0),
    }


summarize_forecast = summarize_trajectories


def summary_from_prediction(prediction, alpha: float = 0.05):
    if isinstance(prediction, dict):
        required = {"lower", "median", "upper"}
        missing = required.difference(prediction)
        if missing:
            raise ValueError(f"Prediction summary is missing keys: {sorted(missing)}")
        summary = {key: ensure_2d(prediction[key]) for key in required}
        summary["mean"] = ensure_2d(prediction.get("mean", prediction["median"]))
        return summary

    return summarize_trajectories(prediction, probs=(alpha / 2, 0.5, 1 - alpha / 2))


def split_prediction_windows(prediction, inference_days: int = 90, forecast_days: int = 10):
    total_days = inference_days + forecast_days
    if isinstance(prediction, dict):
        split = {}
        for key, value in prediction.items():
            arr = ensure_2d(to_numpy(value))
            if arr.shape[0] < total_days:
                raise ValueError(f"prediction[{key!r}] must contain at least {total_days} time points.")
            split[key] = {
                "inference": arr[:inference_days],
                "forecast": arr[inference_days:total_days],
            }
        return {window: {key: value[window] for key, value in split.items()} for window in ("inference", "forecast")}

    prediction = prediction_samples_to_array(prediction)
    if prediction.shape[1] < total_days:
        raise ValueError(f"prediction must contain at least {total_days} time points.")
    return {
        "inference": prediction[:, :inference_days, :],
        "forecast": prediction[:, inference_days:total_days, :],
    }


def interval_score(y_true, lower, upper, alpha: float = 0.05):
    y_true, lower, upper = np.asarray(y_true), np.asarray(lower), np.asarray(upper)
    score = upper - lower
    score += (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    score += (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return score


def weighted_interval_score(y_true, y_samples, alphas: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5)):
    y_true = ensure_2d(to_numpy(y_true))
    y_samples = prediction_samples_to_array(y_samples)
    if y_samples.shape[1:] != y_true.shape:
        raise ValueError(f"y_samples shape {y_samples.shape[1:]} does not match y_true shape {y_true.shape}.")

    y_median = np.median(y_samples, axis=0)
    wis_total = 0.5 * np.abs(y_true - y_median)
    weight_sum = 0.5

    for alpha in alphas:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("All alpha values must be between 0 and 1.")
        lower = np.percentile(y_samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(y_samples, 100 * (1 - alpha / 2), axis=0)
        is_alpha = interval_score(y_true, lower, upper, alpha)
        wis_total += (alpha / 2) * is_alpha
        weight_sum += alpha / 2

    return wis_total / weight_sum


def evaluate_trajectory(
    y_true,
    summary,
    output_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    y_samples=None,
    wis_alphas: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5),
):
    y_true = ensure_2d(to_numpy(y_true))
    median = ensure_2d(summary["median"])
    lower = ensure_2d(summary["lower"])
    upper = ensure_2d(summary["upper"])
    if y_true.shape != median.shape:
        raise ValueError(f"y_true and median must have the same shape, got {y_true.shape} and {median.shape}.")

    n_outputs = y_true.shape[1]
    output_names = [f"output_{i}" for i in range(n_outputs)] if output_names is None else list(output_names)
    if len(output_names) != n_outputs:
        raise ValueError(f"output_names must have length {n_outputs}, got {len(output_names)}.")

    rows = []
    wis = weighted_interval_score(y_true, y_samples, wis_alphas) if y_samples is not None else None
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
                "WIS": float(np.mean(wis[:, j])) if wis is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


evaluate_forecast = evaluate_trajectory


def evaluate_prediction_windows_for_plot(
    y_obs, prediction, inference_days: int = 90, forecast_days: int = 10,
    output_names: Optional[Sequence[str]] = None, alpha: float = 0.05,
    wis_alphas: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5)):

    
    y_inference, y_forecast = split_train_forecast(y_obs, inference_days, forecast_days)
    pred = split_prediction_windows(prediction, inference_days, forecast_days)

    inference_summary = summary_from_prediction(pred["inference"], alpha)
    forecast_summary = summary_from_prediction(pred["forecast"], alpha)
    inference_samples = None if isinstance(pred["inference"], dict) else pred["inference"]
    forecast_samples = None if isinstance(pred["forecast"], dict) else pred["forecast"]
    inference_metrics = evaluate_trajectory(
        y_inference, inference_summary, output_names, alpha, inference_samples, wis_alphas
    )
    forecast_metrics = evaluate_trajectory(
        y_forecast, forecast_summary, output_names, alpha, forecast_samples, wis_alphas
    )
    inference_metrics.insert(0, "window", "inference")
    forecast_metrics.insert(0, "window", "forecast")

    return {"inference_summary": inference_summary, "forecast_summary": forecast_summary}


def evaluate_prediction_windows(
    y_obs,
    prediction,
    inference_days: int = 90,
    forecast_days: int = 10,
    output_names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    wis_alphas: Sequence[float] = (0.02, 0.05, 0.1, 0.2, 0.5),
):
    y_inference, y_forecast = split_train_forecast(y_obs, inference_days, forecast_days)
    pred = split_prediction_windows(prediction, inference_days, forecast_days)

    inference_summary = summary_from_prediction(pred["inference"], alpha)
    forecast_summary = summary_from_prediction(pred["forecast"], alpha)
    inference_samples = None if isinstance(pred["inference"], dict) else pred["inference"]
    forecast_samples = None if isinstance(pred["forecast"], dict) else pred["forecast"]
    inference_metrics = evaluate_trajectory(
        y_inference, inference_summary, output_names, alpha, inference_samples, wis_alphas
    )
    forecast_metrics = evaluate_trajectory(
        y_forecast, forecast_summary, output_names, alpha, forecast_samples, wis_alphas
    )
    inference_metrics.insert(0, "window", "inference")
    forecast_metrics.insert(0, "window", "forecast")

    return pd.concat([inference_metrics, forecast_metrics], ignore_index=True)


def save_metrics(metrics: pd.DataFrame, filename: str):
    metrics.to_csv(filename, index=False)


def summarize_metrics_by_output(metrics: pd.DataFrame):
    metric_cols = [col for col in ["MAE", "RMSE", "PI95_coverage", "interval_score", "WIS"] if col in metrics.columns]
    group_cols = ["window", "output"] if "window" in metrics.columns else ["output"]
    if "inference_days" in metrics.columns:
        group_cols.append("inference_days")
    if "forecast_days" in metrics.columns:
        group_cols.append("forecast_days")
    return metrics.groupby(group_cols)[metric_cols].agg(["mean", "std", "median"]).reset_index()
