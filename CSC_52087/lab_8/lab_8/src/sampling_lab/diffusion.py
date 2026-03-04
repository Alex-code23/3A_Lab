import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from safetensors.torch import save_file, load_file


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def linear_beta_schedule(n_steps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule.

    Returns:
        betas: (n_steps,)
        alphas: (n_steps,)
        alphas_cumprod: (n_steps,)
    """
    betas = torch.linspace(beta_start, beta_end, n_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ScoreNet(nn.Module):
    """Small MLP noise predictor for 2D data.

    Takes a noisy 2D point and a diffusion timestep, predicts the noise.
    """

    def __init__(self, n_steps=200, d_model=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Embedding(n_steps, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.input_proj = nn.Linear(2, d_model)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x, t):
        """
        Args:
            x: Noisy points, shape (batch, 2)
            t: Timestep indices, shape (batch,) — integers in [0, n_steps)

        Returns:
            Predicted noise, shape (batch, 2)
        """
        h = self.input_proj(x) + self.time_embed(t)
        return self.net(h)


# ---------------------------------------------------------------------------
# Forward process (adding noise)
# ---------------------------------------------------------------------------

def q_sample(x_0, t, alphas_cumprod, noise=None):
    """Forward diffusion: sample x_t given x_0.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

    Args:
        x_0: Clean data, shape (batch, 2)
        t: Timestep indices, shape (batch,)
        alphas_cumprod: Cumulative alpha schedule
        noise: Optional pre-drawn noise

    Returns:
        x_t: Noisy data, shape (batch, 2)
        noise: The noise that was added
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    alpha_bar = alphas_cumprod[t].unsqueeze(-1)  # (batch, 1)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise


# ---------------------------------------------------------------------------
# DDPM sampling (provided to students)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample(model, betas, alphas_cumprod, n_samples, device):
    """Full DDPM reverse sampling loop.

    Starting from pure noise, iteratively denoise using the trained model.

    Args:
        model: Trained ScoreNet
        betas: Beta schedule, shape (n_steps,)
        alphas_cumprod: Cumulative alpha schedule, shape (n_steps,)
        n_samples: Number of 2D points to generate
        device: torch device

    Returns:
        samples: Final denoised samples, shape (n_samples, 2)
        trajectory: List of intermediate states (for visualization)
    """
    model.eval()
    n_steps = len(betas)
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    x = torch.randn(n_samples, 2, device=device)
    trajectory = [x.cpu().clone()]

    for t in reversed(range(n_steps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        # Predict noise
        eps_pred = model(x, t_batch)

        # Compute mean of p(x_{t-1} | x_t)
        coeff1 = 1.0 / torch.sqrt(1.0 - beta_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coeff1 * (x - coeff2 * eps_pred)

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * noise
        else:
            x = mean

        # Save trajectory snapshots at regular intervals
        if t % (n_steps // 10) == 0 or t == 0:
            trajectory.append(x.cpu().clone())

    return x.cpu(), trajectory


# ---------------------------------------------------------------------------
# DDIM sampling (student TODO)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model, alphas_cumprod, n_steps_total, n_steps_sample, n_samples, device):
    """DDIM deterministic sampling (accelerated).

    DDIM allows skipping timesteps for faster generation. Instead of going
    through all T steps, we use a subsequence of S steps (S << T).

    The DDIM update formula (deterministic, eta=0):
        x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        x_{t-1}  = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1 - alpha_bar_{t-1}) * eps_pred

    Args:
        model: Trained ScoreNet
        alphas_cumprod: Full cumulative alpha schedule, shape (n_steps_total,)
        n_steps_total: Total number of diffusion steps the model was trained with
        n_steps_sample: Number of DDIM sampling steps to use (fewer = faster)
        n_samples: Number of 2D points to generate
        device: torch device

    Returns:
        samples: Final denoised samples, shape (n_samples, 2)
        trajectory: List of intermediate states
    """
    model.eval()
    alphas_cumprod = alphas_cumprod.to(device)

    # Build a sub-sequence of timesteps (evenly spaced)
    step_size = n_steps_total // n_steps_sample
    timesteps = list(range(0, n_steps_total, step_size))[::-1]  # descending

    x = torch.randn(n_samples, 2, device=device)
    trajectory = [x.cpu().clone()]

    for i, t in enumerate(timesteps):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        alpha_bar_t = alphas_cumprod[t]

        # alpha_bar for the *next* (previous in reverse) timestep
        if i < len(timesteps) - 1:
            t_prev = timesteps[i + 1]
            alpha_bar_prev = alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # Predict noise
        eps_pred = model(x, t_batch)

        # =================================================================
        # TODO: Implement the DDIM update step
        # -----------------------------------------------------------------
        # The DDIM formula (deterministic, eta = 0):
        #
        #   1. Predict the clean data x_0 from the current noisy x_t:
        #      x_0_pred = (x - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        #
        #   2. Compute x at the previous timestep:
        #      x = sqrt(alpha_bar_prev) * x_0_pred
        #          + sqrt(1 - alpha_bar_prev) * eps_pred
        #
        # Replace the line below with your implementation (2 lines):
        # x = x  # placeholder — no update
        # =================================================================
        x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        x = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_pred

        trajectory.append(x.cpu().clone())

    return x.cpu(), trajectory


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_samples(samples, title="Generated Samples", reference=None):
    """Scatter plot of 2D samples, optionally overlaid on reference data.

    Args:
        samples: (n, 2) tensor or array
        title: Plot title
        reference: Optional (n, 2) reference data to show in background

    Returns:
        matplotlib.Figure
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))

    if reference is not None:
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        ax.scatter(reference[:, 0], reference[:, 1], s=5, alpha=0.2, color="#AAAAAA",
                   label="Reference")

    ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="#4C72B0",
               label="Generated")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_trajectory(trajectory, n_snapshots=6, reference=None):
    """Visualize the denoising trajectory as a sequence of snapshots.

    Args:
        trajectory: List of (n, 2) tensors from the sampling process
        n_snapshots: Number of snapshots to display
        reference: Optional (n, 2) reference data for the final panel

    Returns:
        matplotlib.Figure
    """
    # Pick evenly-spaced snapshots
    indices = np.linspace(0, len(trajectory) - 1, n_snapshots, dtype=int)
    snapshots = [trajectory[i] for i in indices]

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))

    for ax, snap, idx in zip(axes, snapshots, indices):
        if isinstance(snap, torch.Tensor):
            snap = snap.detach().cpu().numpy()

        ax.scatter(snap[:, 0], snap[:, 1], s=3, alpha=0.5, color="#4C72B0")

        if reference is not None and idx == indices[-1]:
            ref = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference
            ax.scatter(ref[:, 0], ref[:, 1], s=3, alpha=0.15, color="#DD8452")

        step_label = f"t={len(trajectory) - 1 - idx}" if idx < len(trajectory) - 1 else "t=0"
        ax.set_title(step_label)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Denoising Trajectory", fontsize=13)
    fig.tight_layout()
    return fig


def plot_ddpm_vs_ddim(ddpm_samples, ddim_samples, reference=None):
    """Side-by-side comparison of DDPM and DDIM samples.

    Args:
        ddpm_samples: (n, 2) from ddpm_sample
        ddim_samples: (n, 2) from ddim_sample
        reference: Optional (n, 2) reference data

    Returns:
        matplotlib.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, samples, name in [(ax1, ddpm_samples, "DDPM"), (ax2, ddim_samples, "DDIM")]:
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        if reference is not None:
            ref = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference
            ax.scatter(ref[:, 0], ref[:, 1], s=5, alpha=0.15, color="#AAAAAA", label="Reference")

        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="#4C72B0", label="Generated")
        ax.set_title(name)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend()
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("DDPM vs DDIM Sampling", fontsize=13)
    fig.tight_layout()
    return fig


def plot_steps_degradation(results, reference=None):
    """Show DDIM samples at different numbers of sampling steps.

    Args:
        results: Dict mapping n_steps → samples tensor
        reference: Optional reference data

    Returns:
        matplotlib.Figure
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (steps, samples) in zip(axes, sorted(results.items())):
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        if reference is not None:
            ref = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference
            ax.scatter(ref[:, 0], ref[:, 1], s=5, alpha=0.15, color="#AAAAAA")

        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5, color="#4C72B0")
        ax.set_title(f"DDIM — {steps} steps")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Effect of Reducing Sampling Steps", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------

def make_moons_dataset(n_samples=10000, noise=0.05):
    """Generate a 2D half-moons dataset.

    Returns:
        data: Tensor of shape (n_samples, 2)
    """
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    data = torch.tensor(X, dtype=torch.float32)
    # Center and scale
    data = (data - data.mean(dim=0)) / data.std(dim=0)
    return data


def load_pretrained(checkpoint_dir="checkpoints", device=None):
    """Load a pre-trained ScoreNet and its noise schedule.

    Args:
        checkpoint_dir: Directory containing the checkpoint files
        device: torch device

    Returns:
        model: ScoreNet (in eval mode)
        betas, alphas_cumprod: Schedule tensors
        n_steps: Number of diffusion steps
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint_dir = Path(checkpoint_dir)

    schedule = torch.load(checkpoint_dir / "schedule.pt", weights_only=True)
    n_steps = schedule["n_steps"]
    betas = schedule["betas"]
    alphas_cumprod = schedule["alphas_cumprod"]

    model = ScoreNet(n_steps=n_steps)
    state_dict = load_file(str(checkpoint_dir / "score_net_moons.safetensors"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, betas, alphas_cumprod, n_steps
