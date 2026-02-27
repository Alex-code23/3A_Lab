"""Neural network architectures for learning pendulum dynamics.

Three paradigms:
- BaselineMLP: (q, p) -> (dq/dt, dp/dt) — pure data-fitting
- HamiltonianNN: learns scalar H(q, p), derives dynamics via autograd
- PINN: t -> theta(t), trained with physics-informed loss

Also provides integration utilities (Euler, symplectic, rollout).
"""

import torch
import torch.nn as nn
from pathlib import Path
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Baseline MLP
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """Direct mapping (q, p) -> (dq/dt, dp/dt).

    A standard MLP with Tanh activations — no physics inductive bias.
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 2)
            State [q, p].

        Returns
        -------
        Tensor of shape (batch, 2)
            Predicted [dq/dt, dp/dt].
        """
        return self.network(x)


# ---------------------------------------------------------------------------
# Hamiltonian Neural Network
# ---------------------------------------------------------------------------

class HamiltonianNN(nn.Module):
    """Learn a scalar Hamiltonian H(q, p) and derive dynamics via autograd.

    The key insight: instead of learning the dynamics directly, we learn
    the conserved quantity H. The dynamics follow from Hamilton's equations:
        dq/dt =  dH/dp
        dp/dt = -dH/dq
    This *structurally* enforces energy conservation.
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Compute dynamics from the learned Hamiltonian.

        Parameters
        ----------
        x : Tensor of shape (batch, 2), requires_grad=True
            State [q, p].

        Returns
        -------
        Tensor of shape (batch, 2)
            Predicted [dq/dt, dp/dt] derived via Hamilton's equations.
        """
        # Ensure we can differentiate through x
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)

        # (1) Learn a scalar Hamiltonian
        H = self.network(x)

        # (2) Derive dynamics via autograd
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dq_dt = dH[:, 1:2]    # dH/dp
        dp_dt = -dH[:, 0:1]   # -dH/dq

        return torch.cat([dq_dt, dp_dt], dim=1)

    def hamiltonian(self, x):
        """Evaluate the learned Hamiltonian (scalar) without computing dynamics."""
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        return self.network(x)


# ---------------------------------------------------------------------------
# Physics-Informed Neural Network (PINN)
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Map time t -> theta(t) directly.

    Trained with a combined loss: sparse data observations + physics
    residual (ODE) at collocation points. Uses 4 hidden layers for
    sufficient expressivity.
    """

    def __init__(self, hidden_dim=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        """Predict theta at time t.

        Parameters
        ----------
        t : Tensor of shape (batch, 1)

        Returns
        -------
        Tensor of shape (batch, 1)
            Predicted theta(t).
        """
        return self.network(t)


# ---------------------------------------------------------------------------
# Integration utilities
# ---------------------------------------------------------------------------

def euler_step(model, state, dt):
    """Standard (explicit) Euler integration step.

    Parameters
    ----------
    model : BaselineMLP or HamiltonianNN
        Model that maps state -> (dq/dt, dp/dt).
    state : Tensor of shape (2,)
        Current [q, p].
    dt : float
        Time step.

    Returns
    -------
    Tensor of shape (2,)
        Updated state.
    """
    x = state.unsqueeze(0).requires_grad_(True)
    deriv = model(x).squeeze(0).detach()
    return state + dt * deriv


def symplectic_step(model, state, dt):
    """Symplectic (semi-implicit) Euler integration step.

    Unlike standard Euler, symplectic Euler preserves the symplectic
    structure of Hamiltonian systems, leading to bounded energy error
    over long time horizons.

    The key idea: update p first using the current q, then update q
    using the *new* p. This asymmetry is what preserves the structure.

    Parameters
    ----------
    model : BaselineMLP or HamiltonianNN
        Model that maps state -> (dq/dt, dp/dt).
    state : Tensor of shape (2,)
        Current [q, p].
    dt : float
        Time step.

    Returns
    -------
    Tensor of shape (2,)
        Updated state.
    """
    # =========================================================================
    # TODO: Implement one step of symplectic Euler integration
    # -------------------------------------------------------------------------
    # Standard Euler updates q and p simultaneously using derivatives at the
    # current state. Symplectic Euler is different:
    #
    # Step 1: Compute derivatives at current state (q, p)
    #         x = state.unsqueeze(0).requires_grad_(True)
    #         deriv = model(x).squeeze(0).detach()
    #
    # Step 2: Update p first:  p_new = p + dt * dp_dt
    #
    # Step 3: Recompute derivatives at the intermediate state (q, p_new)
    #
    # Step 4: Update q using the NEW derivatives:  q_new = q + dt * dq_dt_new
    #
    # Return: torch.stack([q_new, p_new])
    #
    # Hint: The order matters! Updating p before q is what makes this
    #       symplectic rather than just a different flavor of Euler.
    #
    # Replace the line below with your implementation:
    # return euler_step(model, state, dt)
    # =========================================================================
    # Step 1: Compute derivatives at current state
    x = state.unsqueeze(0).requires_grad_(True)
    deriv = model(x).squeeze(0).detach()
    
    dq_dt, dp_dt = deriv[0], deriv[1]
    q, p = state[0], state[1]

    # Step 2: Update p first
    p_new = p + dt * dp_dt
    # Step 3: Recompute derivatives at the intermediate state (q, p_new)
    intermediate_state = torch.tensor([q, p_new], device=state.device).unsqueeze(0).requires_grad_(True)
    new_deriv = model(intermediate_state).squeeze(0).detach()
    dq_dt_new = new_deriv[0]
    # Step 4: Update q using the NEW derivatives
    q_new = q + dt * dq_dt_new
    return torch.stack([q_new, p_new])



def rollout(model, initial_state, dt, n_steps, step_fn=euler_step):
    """Integrate a trajectory using the given step function.

    Parameters
    ----------
    model : nn.Module
        Dynamics model (MLP or HNN).
    initial_state : Tensor of shape (2,)
        Starting [q, p].
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.
    step_fn : callable
        Integration function (euler_step or symplectic_step).

    Returns
    -------
    Tensor of shape (n_steps + 1, 2)
        Full trajectory including initial state.
    """
    # Move state to the same device as the model
    device = next(model.parameters()).device
    state = initial_state.detach().clone().to(device)
    trajectory = [state.clone()]

    # Always enable grad: HNN forward pass needs autograd for torch.autograd.grad.
    # Step functions already detach their outputs, so no gradient accumulation.
    with torch.enable_grad():
        for _ in range(n_steps):
            state = step_fn(model, state, dt)
            trajectory.append(state.detach().clone())

    return torch.stack(trajectory)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(path, model_class, device="cpu", **kwargs):
    """Load a model from a safetensors checkpoint.

    Parameters
    ----------
    path : str or Path
        Path to the .safetensors file.
    model_class : type
        One of BaselineMLP, HamiltonianNN, PINN.
    device : str or torch.device
        Target device.
    **kwargs
        Keyword arguments forwarded to model_class constructor.

    Returns
    -------
    nn.Module
        Loaded model in eval mode on the specified device.
    """
    model = model_class(**kwargs)
    state_dict = load_file(str(path))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
