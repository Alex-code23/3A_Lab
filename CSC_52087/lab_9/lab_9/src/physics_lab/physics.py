"""Pure physics utilities and training routines for the pendulum system.

Provides ground-truth ODE solutions via scipy, training data generation,
and training loops for each of the three model types (MLP, HNN, PINN).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.integrate import solve_ivp
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Analytical Hamiltonian
# ---------------------------------------------------------------------------

def pendulum_hamiltonian(q, p, m=1.0, L=1.0, g=9.81):
    """Compute the Hamiltonian H = p^2 / (2mL^2) - mgL cos(q).

    Works with both numpy arrays and torch tensors.
    """
    kinetic = p ** 2 / (2 * m * L ** 2)
    potential = -m * g * L * np.cos(q) if isinstance(q, (float, np.ndarray)) else -m * g * L * torch.cos(q)
    return kinetic + potential


# ---------------------------------------------------------------------------
# Hamilton's equations for solve_ivp
# ---------------------------------------------------------------------------

def pendulum_dynamics(t, state, m=1.0, L=1.0, g=9.81):
    """Hamilton's equations: dq/dt = dH/dp, dp/dt = -dH/dq.

    Parameters
    ----------
    t : float
        Time (unused, system is autonomous).
    state : array-like
        [q, p] — generalized coordinate and momentum.
    m, L, g : float
        Pendulum mass, length, gravitational acceleration.

    Returns
    -------
    list
        [dq/dt, dp/dt]
    """
    q, p = state
    dq_dt = p / (m * L ** 2)          # dH/dp
    dp_dt = -m * g * L * np.sin(q)    # -dH/dq
    return [dq_dt, dp_dt]


# ---------------------------------------------------------------------------
# Ground-truth trajectory generation
# ---------------------------------------------------------------------------

def generate_trajectory(q0=1.0, p0=0.0, t_span=(0, 20), n_points=2000,
                        m=1.0, L=1.0, g=9.81):
    """Generate a ground-truth pendulum trajectory using RK45.

    Parameters
    ----------
    q0, p0 : float
        Initial angle (rad) and angular momentum.
    t_span : tuple
        (t_start, t_end).
    n_points : int
        Number of evenly spaced evaluation points.
    m, L, g : float
        Physical parameters.

    Returns
    -------
    t : ndarray of shape (n_points,)
    states : ndarray of shape (n_points, 2) — columns are [q, p]
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        pendulum_dynamics, t_span, [q0, p0],
        args=(m, L, g),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10, atol=1e-12,
    )
    states = np.stack([sol.y[0], sol.y[1]], axis=1)  # (n_points, 2)
    return sol.t, states


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_training_data(n_trajectories=20, n_points_per=100,
                           q_range=(-np.pi, np.pi), p_range=(-5, 5),
                           m=1.0, L=1.0, g=9.81, seed=42):
    """Generate dense (state, derivative) pairs for MLP/HNN training.

    Samples random initial conditions and integrates short trajectories
    to produce (q, p) -> (dq/dt, dp/dt) pairs.

    Returns
    -------
    states : Tensor of shape (N, 2)
    derivatives : Tensor of shape (N, 2)
    """
    rng = np.random.RandomState(seed)
    all_states = []
    all_derivs = []

    for _ in range(n_trajectories):
        q0 = rng.uniform(*q_range)
        p0 = rng.uniform(*p_range)
        t, states = generate_trajectory(
            q0, p0, t_span=(0, 2), n_points=n_points_per,
            m=m, L=L, g=g,
        )
        # Compute exact derivatives at each point
        derivs = np.array([pendulum_dynamics(0, s, m, L, g) for s in states])
        all_states.append(states)
        all_derivs.append(derivs)

    all_states = np.concatenate(all_states, axis=0)
    all_derivs = np.concatenate(all_derivs, axis=0)

    return (
        torch.tensor(all_states, dtype=torch.float32),
        torch.tensor(all_derivs, dtype=torch.float32),
    )


def generate_pinn_training_data(n_obs=10, n_colloc=500,
                                q0=1.0, p0=0.0, t_span=(0, 2),
                                m=1.0, L=1.0, g=9.81, seed=42):
    """Generate sparse observations + dense collocation points for PINN.

    Parameters
    ----------
    n_obs : int
        Number of sparse (t, theta) observation points.
    n_colloc : int
        Number of collocation points for physics residual.
    q0, p0 : float
        Initial condition for the observed trajectory.
    t_span : tuple
        Time interval for observations.

    Returns
    -------
    t_obs : Tensor of shape (n_obs, 1)
    theta_obs : Tensor of shape (n_obs, 1)
    t_colloc : Tensor of shape (n_colloc, 1)
    """
    rng = np.random.RandomState(seed)

    # Sparse observations from ground truth
    t_full, states_full = generate_trajectory(
        q0, p0, t_span=t_span, n_points=1000, m=m, L=L, g=g,
    )
    obs_idx = np.sort(rng.choice(len(t_full), size=n_obs, replace=False))
    t_obs = torch.tensor(t_full[obs_idx], dtype=torch.float32).unsqueeze(1)
    theta_obs = torch.tensor(states_full[obs_idx, 0], dtype=torch.float32).unsqueeze(1)

    # Dense collocation points (uniformly sampled in t_span)
    t_colloc = torch.tensor(
        rng.uniform(t_span[0], t_span[1], size=(n_colloc, 1)),
        dtype=torch.float32,
    )

    return t_obs, theta_obs, t_colloc


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_mlp(model, states, derivatives, n_epochs=2000, lr=1e-3,
              device="cpu", verbose=True):
    """Train the baseline MLP on (state -> derivative) pairs with MSE loss."""
    model = model.to(device)
    model.train()
    states = states.to(device)
    derivatives = derivatives.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(n_epochs):
        pred = model(states)
        loss = nn.functional.mse_loss(pred, derivatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  MLP epoch {epoch+1}/{n_epochs} — loss: {loss.item():.6f}")

    return history


def train_hnn(model, states, derivatives, n_epochs=2000, lr=1e-3,
              device="cpu", verbose=True):
    """Train the HNN by matching autograd-derived dynamics to true derivatives."""
    model = model.to(device)
    model.train()
    states = states.to(device).requires_grad_(True)
    derivatives = derivatives.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(n_epochs):
        pred = model(states)
        loss = nn.functional.mse_loss(pred, derivatives)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Detach and re-enable grad for next iteration
        states = states.detach().requires_grad_(True)

        history.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(f"  HNN epoch {epoch+1}/{n_epochs} — loss: {loss.item():.6f}")

    return history


def train_pinn(model, t_obs, theta_obs, t_colloc, n_epochs=5000, lr=1e-3,
               lambda_physics=1.0, m=1.0, L=1.0, g=9.81,
               device="cpu", verbose=True):
    """Train the PINN with data loss + physics residual loss.

    L = L_data + lambda * L_physics

    where L_physics enforces d^2theta/dt^2 + (g/L)*sin(theta) = 0
    at the collocation points.
    """
    model = model.to(device)
    model.train()
    t_obs = t_obs.to(device)
    theta_obs = theta_obs.to(device)
    t_colloc = t_colloc.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"total": [], "data": [], "physics": []}

    for epoch in range(n_epochs):
        # --- Data loss ---
        theta_pred_obs = model(t_obs)
        loss_data = nn.functional.mse_loss(theta_pred_obs, theta_obs)

        # --- Physics loss at collocation points ---
        t_c = t_colloc.detach().requires_grad_(True)
        theta_c = model(t_c)

        # First derivative dtheta/dt
        dtheta_dt = torch.autograd.grad(
            theta_c, t_c, grad_outputs=torch.ones_like(theta_c),
            create_graph=True,
        )[0]

        # Second derivative d^2theta/dt^2
        d2theta_dt2 = torch.autograd.grad(
            dtheta_dt, t_c, grad_outputs=torch.ones_like(dtheta_dt),
            create_graph=True,
        )[0]

        # ODE residual: d^2theta/dt^2 + (g/L)*sin(theta) = 0
        residual = d2theta_dt2 + (g / L) * torch.sin(theta_c)
        loss_physics = (residual ** 2).mean()

        # --- Combined loss ---
        loss = loss_data + lambda_physics * loss_physics

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"  PINN epoch {epoch+1}/{n_epochs} — "
                  f"total: {loss.item():.6f}, "
                  f"data: {loss_data.item():.6f}, "
                  f"physics: {loss_physics.item():.6f}")

    return history


# ---------------------------------------------------------------------------
# Train-and-save-all (instructor use)
# ---------------------------------------------------------------------------

def train_and_save_all(save_dir="checkpoints", device="cpu"):
    """Train all three models and save checkpoints + config.

    This is the instructor-facing entry point for pre-training.
    """
    from physics_lab.models import BaselineMLP, HamiltonianNN, PINN

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Shared physical parameters
    m, L, g = 1.0, 1.0, 9.81
    q0, p0 = 1.0, 0.0
    hidden_dim = 64
    n_layers = 3

    print("Generating MLP/HNN training data...")
    states, derivatives = generate_training_data(
        n_trajectories=20, n_points_per=100, m=m, L=L, g=g,
    )

    print("Generating PINN training data...")
    t_obs, theta_obs, t_colloc = generate_pinn_training_data(
        n_obs=10, n_colloc=500, q0=q0, p0=p0, m=m, L=L, g=g,
    )

    # --- Train MLP ---
    print("\nTraining BaselineMLP...")
    mlp = BaselineMLP(hidden_dim=hidden_dim, n_layers=n_layers)
    train_mlp(mlp, states, derivatives, n_epochs=2000, lr=1e-3, device=device)
    mlp.cpu().eval()
    save_file(mlp.state_dict(), str(save_dir / "mlp.safetensors"))
    print("  Saved mlp.safetensors")

    # --- Train HNN ---
    print("\nTraining HamiltonianNN...")
    hnn = HamiltonianNN(hidden_dim=hidden_dim, n_layers=n_layers)
    train_hnn(hnn, states, derivatives, n_epochs=2000, lr=1e-3, device=device)
    hnn.cpu().eval()
    save_file(hnn.state_dict(), str(save_dir / "hnn.safetensors"))
    print("  Saved hnn.safetensors")

    # --- Train PINN ---
    print("\nTraining PINN...")
    pinn = PINN(hidden_dim=hidden_dim, n_layers=4)
    train_pinn(
        pinn, t_obs, theta_obs, t_colloc,
        n_epochs=5000, lr=1e-3, lambda_physics=1.0,
        m=m, L=L, g=g, device=device,
    )
    pinn.cpu().eval()
    save_file(pinn.state_dict(), str(save_dir / "pinn.safetensors"))
    print("  Saved pinn.safetensors")

    # --- Save training config ---
    config = {
        "m": m, "L": L, "g": g,
        "q0": q0, "p0": p0,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
    }
    torch.save(config, str(save_dir / "training_config.pt"))
    print(f"\n  Saved training_config.pt")
    print("All checkpoints saved.")
