"""Visualization functions for the physics lab.

Color scheme:
- Ground truth: #4C72B0 (blue)
- MLP:          #DD8452 (orange)
- HNN:          #55A868 (green)
- PINN:         #C44E52 (red)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

COLORS = {
    "Ground Truth": "#4C72B0",
    "MLP": "#DD8452",
    "HNN": "#55A868",
    "PINN": "#C44E52",
}


def _to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Pendulum animation
# ---------------------------------------------------------------------------

def animate_pendulum(trajectories, times, labels, L=1.0, fps=30,
                     speedup=1.0, title="Pendulum Animation"):
    """Animate pendulums side by side, one panel per trajectory.

    Parameters
    ----------
    trajectories : list of array-like, each of shape (T, 2)
        Each trajectory has columns [theta, p_theta].
    times : array-like of shape (T,)
    labels : list of str
    L : float
        Pendulum length (for display).
    fps : int
        Frames per second for the animation.
    speedup : float
        Playback speed multiplier.
    title : str

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Call `.to_html5_video()` to embed in marimo.
    """
    times = _to_numpy(times)
    trajs = [_to_numpy(t) for t in trajectories]
    n_panels = len(trajs)

    # Subsample frames to match fps
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    step = max(1, int(1.0 / (fps * dt) * speedup))
    frame_idx = list(range(0, len(times), step))

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    lim = L * 1.3
    bobs = []
    rods = []
    time_texts = []
    trails = []

    for i, (ax, label) in enumerate(zip(axes, labels)):
        color = COLORS.get(label, "#333333")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, color=color, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color="gray", linewidth=0.5)

        # Pivot
        ax.plot(0, 0, "ko", markersize=5, zorder=5)

        rod, = ax.plot([], [], "k-", linewidth=1.5, zorder=3)
        bob = Circle((0, 0), L * 0.06, color=color, zorder=4)
        ax.add_patch(bob)
        trail, = ax.plot([], [], color=color, linewidth=0.8, alpha=0.4, zorder=2)
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                            va="top", fontsize=9, family="monospace")

        rods.append(rod)
        bobs.append(bob)
        trails.append(trail)
        time_texts.append(time_text)

    def _xy(theta, length):
        return length * np.sin(theta), -length * np.cos(theta)

    trail_x = [[] for _ in range(n_panels)]
    trail_y = [[] for _ in range(n_panels)]

    def init():
        for i in range(n_panels):
            rods[i].set_data([], [])
            bobs[i].center = (0, 0)
            trails[i].set_data([], [])
            time_texts[i].set_text("")
            trail_x[i].clear()
            trail_y[i].clear()
        return rods + bobs + trails + time_texts

    def update(frame):
        idx = frame_idx[frame]
        for i in range(n_panels):
            theta = trajs[i][min(idx, len(trajs[i]) - 1), 0]
            x, y = _xy(theta, L)
            rods[i].set_data([0, x], [0, y])
            bobs[i].center = (x, y)
            trail_x[i].append(x)
            trail_y[i].append(y)
            # Keep last 80 trail points
            if len(trail_x[i]) > 80:
                trail_x[i] = trail_x[i][-80:]
                trail_y[i] = trail_y[i][-80:]
            trails[i].set_data(trail_x[i], trail_y[i])
            time_texts[i].set_text(f"t = {times[idx]:.1f}s")
        return rods + bobs + trails + time_texts

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(frame_idx), interval=1000 // fps,
                         blit=True)
    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# Phase portrait
# ---------------------------------------------------------------------------

def plot_phase_portrait(trajectories, labels, title="Phase Portrait"):
    """Plot theta vs p_theta for one or more trajectories.

    Parameters
    ----------
    trajectories : list of array-like, each of shape (T, 2)
    labels : list of str
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for traj, label in zip(trajectories, labels):
        traj = _to_numpy(traj)
        color = COLORS.get(label, None)
        lw = 2.0 if label == "Ground Truth" else 1.5
        alpha = 1.0 if label == "Ground Truth" else 0.85
        ax.plot(traj[:, 0], traj[:, 1], label=label, color=color,
                linewidth=lw, alpha=alpha)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$p_\theta$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Energy drift
# ---------------------------------------------------------------------------

def plot_energy_drift(trajectories, times, hamiltonian_fn, labels,
                      title="Energy Over Time"):
    """Plot H(t) for one or more trajectories.

    Parameters
    ----------
    trajectories : list of array-like, each (T, 2)
    times : array-like of shape (T,)
    hamiltonian_fn : callable
        H(q, p) -> scalar energy.
    labels : list of str
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    times = _to_numpy(times)

    for traj, label in zip(trajectories, labels):
        traj = _to_numpy(traj)
        # Ensure trajectory and times have same length
        n = min(len(traj), len(times))
        t = times[:n]
        q, p = traj[:n, 0], traj[:n, 1]
        energy = hamiltonian_fn(q, p)
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()

        color = COLORS.get(label, None)
        lw = 2.0 if label == "Ground Truth" else 1.5
        ax.plot(t, energy, label=label, color=color, linewidth=lw)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy $H$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Trajectory comparison (theta(t) and p(t) subplots)
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(trajectories, times, labels,
                               title="Trajectory Comparison"):
    """Plot theta(t) and p(t) in two subplots.

    Parameters
    ----------
    trajectories : list of array-like, each (T, 2)
    times : array-like of shape (T,)
    labels : list of str
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    times = _to_numpy(times)

    for traj, label in zip(trajectories, labels):
        traj = _to_numpy(traj)
        n = min(len(traj), len(times))
        t = times[:n]
        color = COLORS.get(label, None)
        lw = 2.0 if label == "Ground Truth" else 1.5

        axes[0].plot(t, traj[:n, 0], label=label, color=color, linewidth=lw)
        axes[1].plot(t, traj[:n, 1], label=label, color=color, linewidth=lw)

    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$p_\theta$")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training data comparison (MLP dense vs PINN sparse)
# ---------------------------------------------------------------------------

def plot_training_data_comparison(dense_states, sparse_t, sparse_theta,
                                  title="Training Data: Dense vs Sparse"):
    """Show the difference between MLP/HNN dense data and PINN sparse data.

    Parameters
    ----------
    dense_states : array-like of shape (N, 2)
        Dense (q, p) pairs used for MLP/HNN.
    sparse_t : array-like of shape (M, 1)
        Sparse time observations for PINN.
    sparse_theta : array-like of shape (M, 1)
        Sparse theta observations for PINN.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    dense = _to_numpy(dense_states)
    axes[0].scatter(dense[:, 0], dense[:, 1], s=1, alpha=0.3, color=COLORS["MLP"])
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$p_\theta$")
    axes[0].set_title(f"MLP/HNN: {len(dense)} dense state-derivative pairs")
    axes[0].grid(True, alpha=0.3)

    sp_t = _to_numpy(sparse_t).flatten()
    sp_theta = _to_numpy(sparse_theta).flatten()
    axes[1].scatter(sp_t, sp_theta, s=50, color=COLORS["PINN"],
                     zorder=5, edgecolors="black", linewidths=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\theta$ (rad)")
    axes[1].set_title(f"PINN: {len(sp_t)} sparse observations")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full 2x2 synthesis comparison
# ---------------------------------------------------------------------------

def plot_comparison(results_dict, ground_truth, times, hamiltonian_fn,
                    title="Model Comparison"):
    """2x2 comparison: trajectories, phase portrait, energy, error.

    Parameters
    ----------
    results_dict : dict
        Maps label -> trajectory array of shape (T, 2).
    ground_truth : array-like of shape (T, 2)
    times : array-like of shape (T,)
    hamiltonian_fn : callable
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    gt = _to_numpy(ground_truth)
    t = _to_numpy(times)
    n_gt = len(gt)

    # --- Top-left: theta(t) ---
    ax = axes[0, 0]
    ax.plot(t[:n_gt], gt[:n_gt, 0], label="Ground Truth",
            color=COLORS["Ground Truth"], linewidth=2)
    for label, traj in results_dict.items():
        traj = _to_numpy(traj)
        n = min(len(traj), len(t))
        ax.plot(t[:n], traj[:n, 0], label=label,
                color=COLORS.get(label), linewidth=1.5)
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.set_title(r"$\theta(t)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Top-right: phase portrait ---
    ax = axes[0, 1]
    ax.plot(gt[:, 0], gt[:, 1], label="Ground Truth",
            color=COLORS["Ground Truth"], linewidth=2)
    for label, traj in results_dict.items():
        traj = _to_numpy(traj)
        ax.plot(traj[:, 0], traj[:, 1], label=label,
                color=COLORS.get(label), linewidth=1.5, alpha=0.85)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p_\theta$")
    ax.set_title("Phase Portrait")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-left: energy ---
    ax = axes[1, 0]
    gt_energy = hamiltonian_fn(gt[:, 0], gt[:, 1])
    if isinstance(gt_energy, torch.Tensor):
        gt_energy = gt_energy.detach().cpu().numpy()
    ax.plot(t[:n_gt], gt_energy[:n_gt], label="Ground Truth",
            color=COLORS["Ground Truth"], linewidth=2)
    for label, traj in results_dict.items():
        traj = _to_numpy(traj)
        n = min(len(traj), len(t))
        energy = hamiltonian_fn(traj[:n, 0], traj[:n, 1])
        if isinstance(energy, torch.Tensor):
            energy = energy.detach().cpu().numpy()
        ax.plot(t[:n], energy, label=label,
                color=COLORS.get(label), linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy $H$")
    ax.set_title("Energy Conservation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Bottom-right: absolute error in theta ---
    ax = axes[1, 1]
    for label, traj in results_dict.items():
        traj = _to_numpy(traj)
        n = min(len(traj), n_gt, len(t))
        error = np.abs(traj[:n, 0] - gt[:n, 0])
        ax.plot(t[:n], error, label=label,
                color=COLORS.get(label), linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|\theta_\mathrm{pred} - \theta_\mathrm{true}|$")
    ax.set_title("Absolute Error")
    ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# PINN loss curves
# ---------------------------------------------------------------------------

def plot_pinn_loss_curves(history, title="PINN Training Loss"):
    """Plot the PINN training loss breakdown (data, physics, total).

    Parameters
    ----------
    history : dict
        Keys: "total", "data", "physics", each mapping to a list of floats.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(history["total"]) + 1)
    ax.plot(epochs, history["total"], label="Total", color="black", linewidth=1.5)
    ax.plot(epochs, history["data"], label="$L_{data}$",
            color=COLORS["PINN"], linewidth=1.2, linestyle="--")
    ax.plot(epochs, history["physics"], label=r"$\lambda \cdot L_{physics}$",
            color=COLORS["HNN"], linewidth=1.2, linestyle=":")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
