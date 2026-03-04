import torch
import numpy as np
import matplotlib.pyplot as plt


def gaussian_mixture_log_prob(x, means, stds, weights=None):
    """Log-probability of a mixture of isotropic Gaussians.

    Args:
        x: Points to evaluate, shape (n, dim) or (dim,)
        means: Mixture component means, shape (k, dim)
        stds: Mixture component stds, shape (k,)
        weights: Mixture weights, shape (k,). Uniform if None.

    Returns:
        Log-probabilities, shape (n,) or scalar
    """
    single = x.dim() == 1
    if single:
        x = x.unsqueeze(0)

    k = means.shape[0]
    dim = means.shape[1]

    if weights is None:
        weights = torch.ones(k) / k

    # (n, 1, dim) - (1, k, dim) → (n, k, dim)
    diff = x.unsqueeze(1) - means.unsqueeze(0)
    # (n, k)
    log_components = (
        -0.5 * (diff ** 2).sum(dim=-1) / stds.unsqueeze(0) ** 2
        - dim / 2 * torch.log(2 * torch.tensor(torch.pi))
        - dim * torch.log(stds.unsqueeze(0))
        + torch.log(weights).unsqueeze(0)
    )
    # Log-sum-exp over components → (n,)
    log_prob = torch.logsumexp(log_components, dim=1)

    return log_prob.squeeze(0) if single else log_prob


def make_default_mixture():
    """Create a default 2D mixture of 4 Gaussians for demonstrations.

    Returns:
        target_log_prob: Callable (x) → log_prob
        means: (4, 2) tensor
        stds: (4,) tensor
    """
    means = torch.tensor([
        [-2.0, -2.0],
        [-2.0,  2.0],
        [ 2.0, -2.0],
        [ 2.0,  2.0],
    ])
    stds = torch.tensor([0.7, 0.7, 0.7, 0.7])

    def target_log_prob(x):
        return gaussian_mixture_log_prob(x, means, stds)

    return target_log_prob, means, stds


def metropolis_hastings(target_log_prob, x_init, n_steps, proposal_std=1.0):
    """Run the Metropolis-Hastings MCMC algorithm.

    At each step:
    1. Propose a new point: x' = x + N(0, proposal_std^2 * I)
    2. Compute acceptance ratio: alpha = p(x') / p(x)
    3. Accept with probability min(1, alpha)

    Args:
        target_log_prob: Callable returning the log-probability of a point
        x_init: Starting point, shape (dim,)
        n_steps: Number of MH steps to run
        proposal_std: Standard deviation of the Gaussian proposal

    Returns:
        chain: Tensor of shape (n_steps, dim) — the full chain
        acceptance_rate: Fraction of accepted proposals
    """
    dim = x_init.shape[0]
    chain = torch.zeros(n_steps, dim)
    chain[0] = x_init.clone()
    accepted = 0

    for i in range(1, n_steps):
        current = chain[i - 1]

        # Propose a new point
        proposal = current + proposal_std * torch.randn(dim)

        # =====================================================================
        # TODO: Implement the Metropolis-Hastings accept/reject step
        # ---------------------------------------------------------------------
        # Steps:
        #   1. Compute log_alpha = target_log_prob(proposal) - target_log_prob(current)
        #   2. Draw u ~ Uniform(0, 1) and take its log: log_u = torch.log(torch.rand(1))
        #   3. If log_u < log_alpha: accept (set chain[i] = proposal, increment accepted)
        #      Otherwise: reject (set chain[i] = current)
        #
        # Hint: Working in log-space avoids numerical overflow.
        #
        # Replace the line below with your implementation:
        # chain[i] = current  # placeholder — always rejects
        # =====================================================================
        log_alpha = target_log_prob(proposal) - target_log_prob(current)
        log_u = torch.log(torch.rand(1))
        if log_u < log_alpha:
            chain[i] = proposal
            accepted += 1
        else:
            chain[i] = current



    acceptance_rate = accepted / max(n_steps - 1, 1)
    return chain, acceptance_rate


def plot_chain_2d(chain, target_log_prob=None, title="MCMC Chain", xlim=(-5, 5), ylim=(-5, 5)):
    """Plot a 2D MCMC chain overlaid on the target density contour.

    Args:
        chain: Tensor of shape (n_steps, 2)
        target_log_prob: Optional callable for contour background
        title: Plot title
        xlim: x-axis limits
        ylim: y-axis limits

    Returns:
        matplotlib.Figure
    """
    if isinstance(chain, torch.Tensor):
        chain = chain.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw target density contour if available
    if target_log_prob is not None:
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 200),
            np.linspace(ylim[0], ylim[1], 200),
        )
        grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
        log_probs = target_log_prob(grid).numpy().reshape(xx.shape)
        ax.contourf(xx, yy, np.exp(log_probs), levels=30, cmap="Blues", alpha=0.5)

    # Plot chain
    ax.plot(chain[:, 0], chain[:, 1], "o-", color="#DD8452", markersize=1,
            linewidth=0.3, alpha=0.6)
    ax.plot(chain[0, 0], chain[0, 1], "go", markersize=10, label="Start", zorder=5)
    ax.plot(chain[-1, 0], chain[-1, 1], "r*", markersize=15, label="End", zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_trace(chain, title="Trace Plot"):
    """Plot trace (time series) of each dimension of the chain.

    Args:
        chain: Tensor of shape (n_steps, dim)
        title: Plot title

    Returns:
        matplotlib.Figure
    """
    if isinstance(chain, torch.Tensor):
        chain = chain.detach().cpu().numpy()

    dim = chain.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(12, 3 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    for d, ax in enumerate(axes):
        ax.plot(chain[:, d], color="#4C72B0", linewidth=0.5, alpha=0.8)
        ax.set_ylabel(f"$x_{d + 1}$")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_autocorrelation(chain, max_lag=200, title="Autocorrelation"):
    """Plot autocorrelation for each dimension of the chain.

    Args:
        chain: Tensor of shape (n_steps, dim)
        max_lag: Maximum lag to compute
        title: Plot title

    Returns:
        matplotlib.Figure
    """
    if isinstance(chain, torch.Tensor):
        chain = chain.detach().cpu().numpy()

    dim = chain.shape[1]
    n = chain.shape[0]
    max_lag = min(max_lag, n // 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for d in range(dim):
        x = chain[:, d]
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-12:
            continue

        acf = np.correlate(x, x, mode="full")
        acf = acf[len(acf) // 2:]  # keep positive lags
        acf = acf / (var * n)

        ax.plot(range(max_lag), acf[:max_lag], label=f"$x_{d + 1}$",
                color=colors[d % len(colors)], linewidth=1.5)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def compare_step_sizes(target_log_prob, step_sizes, x_init, n_steps=5000):
    """Run MH with different step sizes and return chains + acceptance rates.

    Args:
        target_log_prob: Target log-density
        step_sizes: List of proposal standard deviations to try
        x_init: Starting point, shape (dim,)
        n_steps: Number of steps per chain

    Returns:
        List of dicts with keys: step_size, chain, acceptance_rate
    """
    results = []
    for sigma in step_sizes:
        chain, acc_rate = metropolis_hastings(target_log_prob, x_init, n_steps, proposal_std=sigma)
        results.append({
            "step_size": sigma,
            "chain": chain,
            "acceptance_rate": acc_rate,
        })
    return results


def plot_step_size_comparison(results, target_log_prob=None):
    """Plot chains side-by-side for different step sizes.

    Args:
        results: List of dicts from compare_step_sizes
        target_log_prob: Optional callable for contour background

    Returns:
        matplotlib.Figure
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    xlim = (-5, 5)
    ylim = (-5, 5)

    for ax, r in zip(axes, results):
        chain = r["chain"].detach().cpu().numpy() if isinstance(r["chain"], torch.Tensor) else r["chain"]

        if target_log_prob is not None:
            xx, yy = np.meshgrid(
                np.linspace(xlim[0], xlim[1], 150),
                np.linspace(ylim[0], ylim[1], 150),
            )
            grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
            log_probs = target_log_prob(grid).numpy().reshape(xx.shape)
            ax.contourf(xx, yy, np.exp(log_probs), levels=20, cmap="Blues", alpha=0.4)

        ax.plot(chain[:, 0], chain[:, 1], "o", color="#DD8452", markersize=0.5, alpha=0.3)
        ax.set_title(f"$\\sigma = {r['step_size']}$\nacc. rate = {r['acceptance_rate']:.1%}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Metropolis-Hastings: Effect of Step Size", fontsize=13)
    fig.tight_layout()
    return fig
