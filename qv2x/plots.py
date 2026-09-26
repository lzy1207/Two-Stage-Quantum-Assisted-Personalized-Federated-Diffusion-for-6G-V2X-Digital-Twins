"""Plots from recorded measurements; never insert manuscript result values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plot_federated_history(output_dir) -> list[Path]:
    """Plot history.json from train_federated and return generated PNG paths.

    The displayed validation RMSE comes from local held-out observations. It
    must not be labeled a dense confidence-fused coarse-REM RMSE. Uplink uses
    decimal MB (1 MB = 1,000,000 bytes). The paper payload includes the bitmap;
    serialized payload additionally includes packet headers and byte padding.
    Missing/empty history is a no-op so this can accompany Stage-II-only plots.
    """
    directory = Path(output_dir)
    source = directory / "history.json"
    if not source.exists():
        return []
    history = json.loads(source.read_text(encoding="utf-8"))
    if not history:
        return []
    required = ("round", "val_rmse_db", "cumulative_uplink_mb", "cumulative_wire_mb")
    missing = {key for row in history for key in required if key not in row}
    if missing:
        raise ValueError(f"Federated history lacks required columns: {sorted(missing)}")
    arrays = {key: np.asarray([row[key] for row in history], dtype=float) for key in required}
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Federated plot columns must contain finite recorded values")
    if (np.diff(arrays["round"]) <= 0).any():
        raise ValueError("Federated history rounds must be strictly increasing")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rounds, rmse = arrays["round"], arrays["val_rmse_db"]
    paper, wire = arrays["cumulative_uplink_mb"], arrays["cumulative_wire_mb"]
    paths = []

    def save(fig, name):
        path = directory / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(rounds, rmse, color="#166534", linewidth=1.8)
    ax.set(xlabel="Federated round", ylabel="Local validation RMSE (dB)",
           title="Stage-I validation convergence")
    ax.grid(alpha=.2)
    save(fig, "fl_rmse_vs_round.png")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(paper, rmse, color="#166534", linewidth=1.8, label="Paper payload")
    ax.plot(wire, rmse, color="#2563eb", linewidth=1.5, linestyle="--", label="Serialized payload")
    ax.set(xlabel="Cumulative uplink (MB)", ylabel="Local validation RMSE (dB)",
           title="Stage-I accuracy and communication")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "fl_rmse_vs_uplink.png")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(rounds, paper, color="#166534", linewidth=1.8, label="Paper payload")
    ax.plot(rounds, wire, color="#2563eb", linewidth=1.5, linestyle="--", label="Serialized payload")
    ax.set(xlabel="Federated round", ylabel="Cumulative uplink (MB)",
           title="Measured Stage-I communication")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "fl_uplink_vs_round.png")
    return paths
