from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .metric import ensure_2d, evaluate_prediction_windows_for_plot, to_numpy


def plot_prediction_windows(y_obs, prediction, inference_days: int = 100,forecast_days: int = 10,
    output_index: int = 0, title: Optional[str] = None, ylabel: str = "Daily incidence",
    alpha: float = 0.05, center: str = "mean", figsize=(6, 4), ax=None,):
    
    y_obs = ensure_2d(to_numpy(y_obs))
    total_days = inference_days + forecast_days
    
    if y_obs.shape[0] < total_days:
        raise ValueError(f"y_obs must contain at least {total_days} time points, got {y_obs.shape[0]}.")

    result = evaluate_prediction_windows_for_plot( y_obs=y_obs, prediction=prediction, inference_days=inference_days,
        forecast_days=forecast_days,alpha=alpha)
    
    inference_summary = result["inference_summary"]
    forecast_summary = result["forecast_summary"]

    if center not in {"mean", "median"}:
        raise ValueError("center must be either 'mean' or 'median'.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi =300)

    days = np.arange(1, total_days + 1)
    inference_axis = days[:inference_days]
    forecast_axis = days[inference_days:total_days]
    interval_label = f"{int((1 - alpha) * 100)}% interval"
    calibration_color = "0.72"
    prediction_color = "#2962FF"
    prediction_line = "#1f77ff"

    ax.fill_between(inference_axis, inference_summary["lower"][:, output_index], inference_summary["upper"][:, output_index],
        color=calibration_color, alpha=0.45, label=f"{interval_label}")
    ax.plot(inference_axis, inference_summary[center][:, output_index], color="0.45",linewidth=2, label=f"Pred {center}")

    
    ax.fill_between(forecast_axis, forecast_summary["lower"][:, output_index], forecast_summary["upper"][:, output_index],
        color=prediction_color, alpha=0.7, label=f"{interval_label} prediction")
    ax.plot(forecast_axis,forecast_summary[center][:, output_index],color=prediction_line,linewidth=2,label=f"Pred {center} prediction")

    ax.scatter(inference_axis,y_obs[:inference_days, output_index],color="black",
        s=12,label="Observed data")
    ax.scatter(forecast_axis, y_obs[inference_days:total_days, output_index],color="black",
        s=12)

    ax.axvline(inference_days + 0.5, color="0.4", linestyle=":", linewidth=1)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    return ax
