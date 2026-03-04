import torch
import matplotlib.pyplot as plt


def gaussian_importance_sampling(f, target_log_prob, proposal, n_samples):
    """Estimate E_target[f(x)] using importance sampling.

    Args:
        f: Function to compute expectation of (callable on a batch of samples)
        target_log_prob: Callable returning log-density of the target distribution
        proposal: torch.distributions object with .sample() and .log_prob()
        n_samples: Number of samples to draw

    Returns:
        estimate: Scalar estimate of E[f(x)]
        weights: Normalized importance weights (n_samples,)
        samples: Drawn samples (n_samples, dim)
    """
    samples = proposal.sample((n_samples,))

    log_w = target_log_prob(samples) - proposal.log_prob(samples)
    # Normalize in log-space for numerical stability
    log_w = log_w - log_w.max()
    weights = torch.exp(log_w)
    weights = weights / weights.sum()

    f_values = f(samples)
    estimate = (weights * f_values).sum()

    return estimate, weights, samples


def effective_sample_size(weights):
    """Compute the Effective Sample Size (ESS) from normalized importance weights.

    ESS measures how many "effective" independent samples we have. When all
    weights are equal, ESS = N. When one weight dominates, ESS -> 1.

    The formula is:
        ESS = 1 / sum(w_i^2)

    where w_i are the normalized importance weights (they sum to 1).

    Args:
        weights: Normalized importance weights, shape (n_samples,). Must sum to 1.

    Returns:
        ESS as a scalar (float or tensor)
    """

    # =========================================================================
    # TODO: Implement the Effective Sample Size
    # -------------------------------------------------------------------------
    # The formula is: ESS = 1 / sum(w_i^2)
    #
    # Steps:
    #   1. Square each weight
    #   2. Sum the squared weights
    #   3. Return the reciprocal
    #
    # This should be a single line! Replace the line below:
    # ess = torch.tensor(0.0)
    # =========================================================================
    ess = 1.0 / torch.sum(weights ** 2)
    return ess


def run_importance_sampling_experiment(dims, n_samples=10000, target_offset=3.0):
    """Run importance sampling in increasing dimensions and track ESS collapse.

    Uses a standard normal proposal to estimate E_target[1] (= 1) under a
    shifted Gaussian target. As dimension grows, the proposal and target
    overlap less, causing weight collapse.

    Args:
        dims: List of dimensions to try (e.g., [1, 2, 5, 10, 20, 50])
        n_samples: Number of samples per experiment
        target_offset: How far the target mean is shifted from the origin

    Returns:
        results: List of dicts with keys: dim, estimate, ess, weights
    """
    results = []

    for d in dims:
        proposal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(d),
            covariance_matrix=torch.eye(d),
        )

        target_mean = torch.full((d,), target_offset)

        def target_log_prob(x, mean=target_mean):
            diff = x - mean
            return -0.5 * (diff * diff).sum(dim=-1)

        def f(x):
            return torch.ones(x.shape[0])

        estimate, weights, samples = gaussian_importance_sampling(
            f, target_log_prob, proposal, n_samples,
        )
        ess = effective_sample_size(weights)

        results.append({
            "dim": d,
            "estimate": estimate.item(),
            "ess": ess.item() if isinstance(ess, torch.Tensor) else ess,
            "weights": weights,
        })

    return results


def plot_weight_distribution(weights, title="Importance Weight Distribution"):
    """Visualize the distribution of importance weights.

    Args:
        weights: Normalized importance weights (n_samples,)
        title: Plot title

    Returns:
        matplotlib.Figure
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of weights
    ax1.hist(weights, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Weight value")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title}")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Sorted weights (Lorenz-style)
    sorted_w = sorted(weights, reverse=True)
    cumulative = [sum(sorted_w[:i+1]) for i in range(len(sorted_w))]
    ax2.plot(range(len(cumulative)), cumulative, color="#DD8452", linewidth=2)
    ax2.set_xlabel("Number of top samples")
    ax2.set_ylabel("Cumulative weight")
    ax2.set_title("Cumulative weight (sorted)")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="50%")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_ess_vs_dimension(results):
    """Plot ESS as a function of dimension.

    Args:
        results: List of dicts from run_importance_sampling_experiment

    Returns:
        matplotlib.Figure
    """
    dims = [r["dim"] for r in results]
    ess_values = [r["ess"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, ess_values, "o-", color="#4C72B0", linewidth=2, markersize=8)

    for d, e in zip(dims, ess_values):
        ax.annotate(f"{e:.0f}", (d, e), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Effective Sample Size (ESS)")
    ax.set_title("ESS Collapse with Increasing Dimension")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
