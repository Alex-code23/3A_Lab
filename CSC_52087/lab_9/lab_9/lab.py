import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell
def imports():
    import torch
    import numpy as np
    import os

    from physics_lab.utils import get_device, set_seed
    from physics_lab.physics import (
        pendulum_hamiltonian,
        pendulum_dynamics,
        generate_trajectory,
        generate_training_data,
        generate_pinn_training_data,
        train_pinn,
    )
    from physics_lab.models import (
        BaselineMLP,
        HamiltonianNN,
        PINN,
        euler_step,
        symplectic_step,
        rollout,
        load_model,
        count_parameters,
    )
    from physics_lab.visualization import (
        animate_pendulum,
        plot_phase_portrait,
        plot_energy_drift,
        plot_trajectory_comparison,
        plot_training_data_comparison,
        plot_comparison,
        plot_pinn_loss_curves,
    )

    set_seed(42)
    device = get_device()

    mo.output.replace(mo.md(f"""
    ## Environment Setup

    | Component | Status |
    |-----------|--------|
    | **Device** | `{device}` |
    | **PyTorch** | `{torch.__version__}` |
    """))
    return (
        BaselineMLP,
        HamiltonianNN,
        PINN,
        animate_pendulum,
        count_parameters,
        device,
        euler_step,
        generate_pinn_training_data,
        generate_trajectory,
        generate_training_data,
        load_model,
        np,
        os,
        pendulum_hamiltonian,
        pendulum_dynamics,
        plot_comparison,
        plot_energy_drift,
        plot_phase_portrait,
        plot_pinn_loss_curves,
        plot_trajectory_comparison,
        plot_training_data_comparison,
        rollout,
        symplectic_step,
        torch,
        train_pinn,
    )


@app.cell(hide_code=True)
def title():
    mo.output.replace(mo.md("""
    # Lab 9 — World Models for Physical Systems

    **Advanced Deep Learning** — 2025-2026

    - **Lecture:** Prof. Ye Zhu
    - **Lab:** Dr Guillaume Lachaud

    ---

    One system. Three paradigms. One question: *what makes a trustworthy world model?*

    | Model | Physics role | Conservation |
    |-------|-------------|-------------|
    | **MLP** | None — pure data | Not enforced |
    | **HNN** | Structural inductive bias | Enforced by architecture |
    | **PINN** | Regularization via ODE residual | Approximate, via loss |

    You will probe each model on a simple pendulum, observe where they succeed
    and fail, and discover that **conservation must be enforced structurally or
    it will be violated**.
    """))
    return


# =========================================================================
# PART 0: PHYSICS & ARCHITECTURE RECAP
# =========================================================================


@app.cell(hide_code=True)
def part0_recap():
    mo.output.replace(mo.md(r"""
    ---
    # Part 0: Physics & Architecture Recap

    > See also the **Hamiltonian Primer** document for a fuller derivation.

    ## Hamiltonian Mechanics — Key Ideas

    For a system with generalized coordinate $q$ and conjugate momentum $p$, the
    **Hamiltonian** $H(q, p) = T + V$ is the total energy. The equations of motion are:

    $$\dot{q} = \frac{\partial H}{\partial p}, \qquad \dot{p} = -\frac{\partial H}{\partial q}$$

    **Key property:** Along any trajectory, $\frac{dH}{dt} = \frac{\partial H}{\partial q}\dot{q} + \frac{\partial H}{\partial p}\dot{p} = 0$. Energy is conserved.

    ## The Simple Pendulum

    For a pendulum of mass $m$, length $L$, under gravity $g$:

    $$H(\theta, p_\theta) = \frac{p_\theta^2}{2mL^2} - mgL\cos\theta$$

    where $\theta$ is the angle and $p_\theta = mL^2\dot\theta$ is the angular momentum.
    """))
    return


@app.cell
def part0_verify(generate_trajectory, np, pendulum_hamiltonian, torch):
    _t, _states = generate_trajectory(q0=1.0, p0=0.0, t_span=(0, 10), n_points=1000)
    _q, _p = _states[:, 0], _states[:, 1]
    _H = pendulum_hamiltonian(_q, _p)
    _H_drift = np.max(_H) - np.min(_H)

    if _H_drift < 1e-6:
        _result = mo.callout(
            mo.md(f"**Energy conservation verified!**\n\n"
                   f"Max energy drift over 10s: {_H_drift:.2e}\n\n"
                   f"The ODE solver preserves the Hamiltonian — this is our ground truth."),
            kind="success",
        )
    else:
        _result = mo.callout(
            mo.md(f"**Warning:** Energy drift = {_H_drift:.2e} (expected < 1e-6)"),
            kind="warn",
        )

    mo.output.replace(mo.vstack([
        mo.md("### Ground Truth Verification"),
        _result,
    ]))
    return


@app.cell(hide_code=True)
def part0_paradigms():
    mo.output.replace(mo.md(r"""
    ## Three Paradigms, One System

    All three models learn pendulum dynamics, but encode physics differently:

    **MLP** (Baseline)
    - Input: $(\theta, p_\theta)$ → Output: $(\dot\theta, \dot{p}_\theta)$
    - Trained on dense trajectory data — no physics knowledge
    - Integration: standard Euler

    **HNN** (Hamiltonian Neural Network)
    - Learns a scalar $\hat{H}(\theta, p_\theta)$
    - Derives $\dot\theta = \partial\hat{H}/\partial p$, $\dot{p} = -\partial\hat{H}/\partial\theta$ via `autograd`
    - Conservation is **structural** — enforced by architecture

    **PINN** (Physics-Informed Neural Network)
    - Input: $t$ → Output: $\theta(t)$ directly
    - Trained on sparse observations + ODE residual penalty
    - Physics is **approximate** — enforced as a loss term
    """))
    return


@app.cell
def part0_prediction():
    prediction = mo.ui.text_area(
        label="Before running any experiment, rank the three models (MLP, HNN, PINN) "
              "for long-horizon extrapolation accuracy. Justify your ranking. "
              "We will revisit this at the end.",
        full_width=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Your Prediction"),
        prediction,
    ]))
    return (prediction,)


# =========================================================================
# PART 1A: MLP BASELINE
# =========================================================================


@app.cell(hide_code=True)
def part1a_intro():
    mo.output.replace(mo.md("""
    ---
    # Part 1 — Probing the Three Models

    ## Sub-part A: MLP Baseline (~20 min)

    The MLP learns a direct mapping $(\theta, p_\\theta) \\to (\\dot\\theta, \\dot{p}_\\theta)$ from
    dense trajectory data. It has **no physics knowledge** — just function approximation.
    """))
    return


@app.cell
def part1a_load(BaselineMLP, count_parameters, device, load_model, os, torch):
    _ckpt = "checkpoints/mlp.safetensors"
    _config_path = "checkpoints/training_config.pt"

    if not os.path.exists(_ckpt) or not os.path.exists(_config_path):
        mo.output.replace(mo.callout(
            mo.md("**Checkpoints not found** in `checkpoints/`.\n\n"
                   "Ask your instructor for the checkpoint files."),
            kind="danger",
        ))
        mo.stop(True)

    config = torch.load(_config_path, weights_only=False)
    mlp = load_model(_ckpt, BaselineMLP, device=device,
                     hidden_dim=config["hidden_dim"], n_layers=config["n_layers"])

    mo.output.replace(mo.md(f"""
    ### MLP Loaded

    | Property | Value |
    |----------|-------|
    | **Parameters** | {count_parameters(mlp):,} |
    | **Architecture** | {config['n_layers']}-layer MLP, hidden_dim={config['hidden_dim']} |
    | **Activation** | Tanh |
    """))
    return config, mlp


@app.cell
def part1a_short(config, euler_step, generate_trajectory, mlp, np,
                 pendulum_hamiltonian, plot_phase_portrait,
                 plot_trajectory_comparison, rollout, torch):
    _t_short, _gt_short = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 2), n_points=200,
        m=config["m"], L=config["L"], g=config["g"],
    )

    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _mlp_short = rollout(mlp, _init, dt=0.01, n_steps=200, step_fn=euler_step)

    _fig_traj = plot_trajectory_comparison(
        [_gt_short, _mlp_short], _t_short,
        ["Ground Truth", "MLP"],
        title="MLP — Short Horizon (2s)",
    )
    _fig_phase = plot_phase_portrait(
        [_gt_short, _mlp_short],
        ["Ground Truth", "MLP"],
        title="MLP — Phase Portrait (2s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### MLP: Short Horizon (0–2s)"),
        _fig_traj,
        _fig_phase,
    ]))
    return


@app.cell
def part1a_long(config, euler_step, generate_trajectory, mlp, np,
                pendulum_hamiltonian, plot_energy_drift, plot_phase_portrait,
                plot_trajectory_comparison, rollout, torch):
    _t_long, _gt_long = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 20), n_points=2000,
        m=config["m"], L=config["L"], g=config["g"],
    )

    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _mlp_long = rollout(mlp, _init, dt=0.01, n_steps=2000, step_fn=euler_step)

    _fig_traj = plot_trajectory_comparison(
        [_gt_long, _mlp_long], _t_long,
        ["Ground Truth", "MLP"],
        title="MLP — Long Horizon (20s)",
    )
    _fig_phase = plot_phase_portrait(
        [_gt_long, _mlp_long],
        ["Ground Truth", "MLP"],
        title="MLP — Phase Portrait (20s)",
    )

    _H_fn = lambda q, p: pendulum_hamiltonian(q, p, m=config["m"], L=config["L"], g=config["g"])
    _fig_energy = plot_energy_drift(
        [_gt_long, _mlp_long], _t_long, _H_fn,
        ["Ground Truth", "MLP"],
        title="MLP — Energy Drift (20s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### MLP: Long Horizon (0–20s)"),
        _fig_traj,
        _fig_phase,
        _fig_energy,
    ]))
    return


@app.cell
def part1a_animation(animate_pendulum, config, euler_step, generate_trajectory,
                     mlp, rollout, torch):
    _t, _gt = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 10), n_points=1000,
        m=config["m"], L=config["L"], g=config["g"],
    )
    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _mlp_traj = rollout(mlp, _init, dt=0.01, n_steps=1000, step_fn=euler_step)

    _anim = animate_pendulum(
        [_gt, _mlp_traj], _t,
        ["Ground Truth", "MLP"],
        L=config["L"], fps=30, speedup=1.0,
        title="MLP vs Ground Truth (10s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### Animation: MLP vs Ground Truth"),
        mo.md("Watch the MLP pendulum — it gradually swings faster as energy is created."),
        mo.Html(_anim.to_html5_video()),
    ]))
    return


@app.cell
def part1a_questions():
    a1 = mo.ui.text_area(
        label="A1: Look at the energy plot. What physical law is being violated?",
        full_width=True,
    )
    a2 = mo.ui.text_area(
        label="A2: At what point in time does the MLP start failing noticeably? "
              "Is this a model capacity problem or a structural problem?",
        full_width=True,
    )
    a3 = mo.ui.text_area(
        label="A3: Could you fix this by training on more data or using a bigger MLP? "
              "Why or why not?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Questions — MLP"),
        a1, a2, a3,
    ]))
    return a1, a2, a3


# =========================================================================
# PART 1B: HAMILTONIAN NEURAL NETWORK
# =========================================================================


@app.cell(hide_code=True)
def part1b_intro():
    mo.output.replace(mo.md("""
    ---
    ## Sub-part B: Hamiltonian Neural Network (~25 min)

    Instead of learning dynamics directly, the HNN learns a **scalar Hamiltonian**
    $\\hat{H}(q, p)$ and derives dynamics via automatic differentiation.

    This structurally enforces $dH/dt = 0$ — conservation is not learned, it is
    **guaranteed by the architecture**.
    """))
    return


@app.cell
def part1b_load(HamiltonianNN, config, count_parameters, device, load_model, os):
    _ckpt = "checkpoints/hnn.safetensors"

    if not os.path.exists(_ckpt):
        mo.output.replace(mo.callout(
            mo.md("**HNN checkpoint not found.** Ask your instructor."),
            kind="danger",
        ))
        mo.stop(True)

    hnn = load_model(_ckpt, HamiltonianNN, device=device,
                     hidden_dim=config["hidden_dim"], n_layers=config["n_layers"])

    mo.output.replace(mo.md(f"""
    ### HNN Loaded

    | Property | Value |
    |----------|-------|
    | **Parameters** | {count_parameters(hnn):,} |
    | **Output** | Scalar $\\hat{{H}}(q, p)$ → dynamics via autograd |
    """))
    return (hnn,)


@app.cell(hide_code=True)
def part1b_annotate():
    mo.output.replace(mo.md(r"""
    ### HNN Forward Pass — Read and Annotate

    Study this forward pass carefully. The comments highlight the key design choices.

    ```python
    def forward(self, x):
        # x has shape (batch, 2) with columns [q, p]

        # (1) Learn a scalar Hamiltonian
        H = self.network(x)          # shape: (batch, 1)

        # (2) Derive dynamics via autograd — NOT a second network!
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        dq_dt =  dH[:, 1:2]          # ∂H/∂p  — Hamilton's first equation
        dp_dt = -dH[:, 0:1]          # -∂H/∂q — Hamilton's second equation

        return torch.cat([dq_dt, dp_dt], dim=1)
    ```

    **Think about these questions** (no need to write answers, but discuss with your neighbor):

    1. Why does line (2) use `autograd` rather than a second network?
    2. Why is `dp_dt` **negative** `dH/dq`?
    3. Write out $dH/dt$ using the chain rule. What does it equal, and why?
    """))
    return


@app.cell
def part1b_test(config, hnn, symplectic_step, euler_step, rollout, torch):
    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)

    # Test with symplectic_step
    _sym_traj = rollout(hnn, _init, dt=0.01, n_steps=10, step_fn=symplectic_step)
    _eul_traj = rollout(hnn, _init, dt=0.01, n_steps=10, step_fn=euler_step)

    # If symplectic_step just delegates to euler_step, trajectories will be identical
    _diff = (_sym_traj - _eul_traj).abs().max().item()

    if _diff > 1e-6:
        _result = mo.callout(
            mo.md("**Symplectic step test passed!**\n\n"
                   "Your symplectic integrator produces different results from "
                   "standard Euler — this is expected."),
            kind="success",
        )
    else:
        _result = mo.callout(
            mo.md("**Symplectic step not implemented yet.**\n\n"
                   "Your `symplectic_step` currently returns the same result as "
                   "`euler_step`. Open `src/physics_lab/models.py` and complete "
                   "the TODO in `symplectic_step`.\n\n"
                   "You can still continue with the lab using standard Euler — "
                   "the results will be slightly less accurate on long horizons."),
            kind="warn",
        )

    mo.output.replace(mo.vstack([
        mo.md("### Test Your Symplectic Step Implementation"),
        _result,
    ]))
    return


@app.cell
def part1b_short(config, generate_trajectory, hnn, plot_phase_portrait,
                 plot_trajectory_comparison, rollout, symplectic_step, torch):
    _t_short, _gt_short = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 2), n_points=200,
        m=config["m"], L=config["L"], g=config["g"],
    )

    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _hnn_short = rollout(hnn, _init, dt=0.01, n_steps=200, step_fn=symplectic_step)

    _fig_traj = plot_trajectory_comparison(
        [_gt_short, _hnn_short], _t_short,
        ["Ground Truth", "HNN"],
        title="HNN — Short Horizon (2s)",
    )
    _fig_phase = plot_phase_portrait(
        [_gt_short, _hnn_short],
        ["Ground Truth", "HNN"],
        title="HNN — Phase Portrait (2s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### HNN: Short Horizon (0–2s)"),
        _fig_traj,
        _fig_phase,
    ]))
    return


@app.cell
def part1b_long(config, generate_trajectory, hnn, np,
                pendulum_hamiltonian, plot_energy_drift, plot_phase_portrait,
                plot_trajectory_comparison, rollout, symplectic_step, torch):
    _t_long, _gt_long = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 20), n_points=2000,
        m=config["m"], L=config["L"], g=config["g"],
    )

    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _hnn_long = rollout(hnn, _init, dt=0.01, n_steps=2000, step_fn=symplectic_step)

    _fig_traj = plot_trajectory_comparison(
        [_gt_long, _hnn_long], _t_long,
        ["Ground Truth", "HNN"],
        title="HNN — Long Horizon (20s)",
    )
    _fig_phase = plot_phase_portrait(
        [_gt_long, _hnn_long],
        ["Ground Truth", "HNN"],
        title="HNN — Phase Portrait (20s)",
    )

    _H_fn = lambda q, p: pendulum_hamiltonian(q, p, m=config["m"], L=config["L"], g=config["g"])
    _fig_energy = plot_energy_drift(
        [_gt_long, _hnn_long], _t_long, _H_fn,
        ["Ground Truth", "HNN"],
        title="HNN — Energy Conservation (20s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### HNN: Long Horizon (0–20s)"),
        mo.md("Compare the energy plot to the MLP — this is the key result."),
        _fig_traj,
        _fig_phase,
        _fig_energy,
    ]))
    return


@app.cell
def part1b_animation(animate_pendulum, config, generate_trajectory,
                     hnn, rollout, symplectic_step, torch):
    _t, _gt = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 10), n_points=1000,
        m=config["m"], L=config["L"], g=config["g"],
    )
    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _hnn_traj = rollout(hnn, _init, dt=0.01, n_steps=1000, step_fn=symplectic_step)

    _anim = animate_pendulum(
        [_gt, _hnn_traj], _t,
        ["Ground Truth", "HNN"],
        L=config["L"], fps=30, speedup=1.0,
        title="HNN vs Ground Truth (10s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### Animation: HNN vs Ground Truth"),
        mo.md("The HNN pendulum stays in sync — energy conservation keeps the orbit closed."),
        mo.Html(_anim.to_html5_video()),
    ]))
    return


@app.cell
def part1b_questions():
    b1 = mo.ui.text_area(
        label="B1: Compare the phase portrait to the MLP. What changed? "
              "The HNN and MLP were trained on identical data — what's different?",
        full_width=True,
    )
    b2 = mo.ui.text_area(
        label="B2: The HNN conserves energy by construction. Does this mean it "
              "predicts the correct trajectory? Can you think of a case where "
              "conservation holds but the trajectory is wrong?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Questions — HNN"),
        b1, b2,
    ]))
    return b1, b2


# =========================================================================
# PART 1C: PINN
# =========================================================================


@app.cell(hide_code=True)
def part1c_intro():
    mo.output.replace(mo.md(r"""
    ---
    ## Sub-part C: Physics-Informed Neural Network (~25 min)

    The PINN learns $\theta(t)$ directly from sparse observations, regularized
    by the pendulum ODE:

    $$\mathcal{L} = \underbrace{\mathcal{L}_\text{data}}_{\text{sparse observations}} + \lambda \underbrace{\mathcal{L}_\text{physics}}_{\ddot\theta + \frac{g}{L}\sin\theta = 0}$$

    Unlike MLP/HNN, the PINN doesn't need dense trajectory data — just a few
    scattered $(t, \theta)$ points plus the governing equation.
    """))
    return


@app.cell
def part1c_load(PINN, config, count_parameters, device, load_model, os):
    _ckpt = "checkpoints/pinn.safetensors"

    if not os.path.exists(_ckpt):
        mo.output.replace(mo.callout(
            mo.md("**PINN checkpoint not found.** Ask your instructor."),
            kind="danger",
        ))
        mo.stop(True)

    pinn = load_model(_ckpt, PINN, device=device,
                      hidden_dim=config["hidden_dim"], n_layers=4)

    mo.output.replace(mo.md(f"""
    ### PINN Loaded

    | Property | Value |
    |----------|-------|
    | **Parameters** | {count_parameters(pinn):,} |
    | **Input** | $t$ |
    | **Output** | $\\theta(t)$ |
    | **Physics** | ODE residual as loss penalty |
    """))
    return (pinn,)


@app.cell
def part1c_data_contrast(config, generate_pinn_training_data,
                         generate_training_data, plot_training_data_comparison):
    _states, _derivs = generate_training_data(
        n_trajectories=20, n_points_per=100,
        m=config["m"], L=config["L"], g=config["g"],
    )
    _t_obs, _theta_obs, _t_colloc = generate_pinn_training_data(
        n_obs=10, n_colloc=500,
        q0=config["q0"], p0=config["p0"],
        m=config["m"], L=config["L"], g=config["g"],
    )

    _fig = plot_training_data_comparison(_states, _t_obs, _theta_obs)

    mo.output.replace(mo.vstack([
        mo.md("### Training Data Contrast"),
        mo.md("The MLP/HNN were trained on **2,000 dense state-derivative pairs**. "
              "The PINN was trained on just **10 sparse observations** plus "
              "500 collocation points where the ODE residual is penalized."),
        _fig,
    ]))
    return


@app.cell
def part1c_predictions(config, device, generate_trajectory, np, pinn,
                       pendulum_hamiltonian, plot_energy_drift,
                       plot_trajectory_comparison, torch):
    # Generate ground truth for both horizons
    _t_short, _gt_short = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 2), n_points=200,
        m=config["m"], L=config["L"], g=config["g"],
    )
    _t_long, _gt_long = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 20), n_points=2000,
        m=config["m"], L=config["L"], g=config["g"],
    )

    # PINN predicts theta(t) directly — need to also get p via autograd
    def _pinn_trajectory(t_array, pinn_model, m, L):
        """Get (theta, p) trajectory from PINN."""
        _dev = next(pinn_model.parameters()).device
        t_tensor = torch.tensor(t_array, dtype=torch.float32, device=_dev).unsqueeze(1).requires_grad_(True)
        theta = pinn_model(t_tensor)
        dtheta_dt = torch.autograd.grad(
            theta, t_tensor, grad_outputs=torch.ones_like(theta),
            create_graph=False,
        )[0]
        p = m * L ** 2 * dtheta_dt  # p = mL^2 * dtheta/dt
        traj = torch.cat([theta, p], dim=1).detach().cpu()
        return traj

    _pinn_short = _pinn_trajectory(_t_short, pinn, config["m"], config["L"])
    _pinn_long = _pinn_trajectory(_t_long, pinn, config["m"], config["L"])

    _fig_short = plot_trajectory_comparison(
        [_gt_short, _pinn_short], _t_short,
        ["Ground Truth", "PINN"],
        title="PINN — Short Horizon (2s)",
    )
    _fig_long = plot_trajectory_comparison(
        [_gt_long, _pinn_long], _t_long,
        ["Ground Truth", "PINN"],
        title="PINN — Long Horizon (20s)",
    )

    _H_fn = lambda q, p: pendulum_hamiltonian(q, p, m=config["m"], L=config["L"], g=config["g"])
    _fig_energy = plot_energy_drift(
        [_gt_long, _pinn_long], _t_long, _H_fn,
        ["Ground Truth", "PINN"],
        title="PINN — Energy (20s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### PINN Predictions"),
        mo.md("**Short horizon (2s)** — within the training window:"),
        _fig_short,
        mo.md("**Long horizon (20s)** — 10x beyond the training window:"),
        _fig_long,
        _fig_energy,
        mo.callout(
            mo.md("The PINN was trained on $t \\in [0, 2]$ only. Beyond that window, "
                   "it has **no data and no physics guidance** — the collocation points "
                   "also stop at $t = 2$. The Tanh activations saturate and the output "
                   "freezes to a constant. This is a fundamentally different failure mode "
                   "from the MLP: the MLP creates energy, while the PINN simply stops predicting."),
            kind="info",
        ),
    ]))
    return


@app.cell
def part1c_animation(animate_pendulum, config, generate_trajectory, pinn, torch):
    _t, _gt = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 10), n_points=1000,
        m=config["m"], L=config["L"], g=config["g"],
    )

    # PINN: t -> theta(t), derive p via autograd
    _pinn_dev = next(pinn.parameters()).device
    _t_tensor = torch.tensor(_t, dtype=torch.float32, device=_pinn_dev).unsqueeze(1).requires_grad_(True)
    _theta = pinn(_t_tensor)
    _dtheta = torch.autograd.grad(
        _theta, _t_tensor, grad_outputs=torch.ones_like(_theta),
        create_graph=False,
    )[0]
    _p_pinn = config["m"] * config["L"] ** 2 * _dtheta
    _pinn_traj = torch.cat([_theta, _p_pinn], dim=1).detach().cpu()

    _anim = animate_pendulum(
        [_gt, _pinn_traj], _t,
        ["Ground Truth", "PINN"],
        L=config["L"], fps=30, speedup=1.0,
        title="PINN vs Ground Truth (10s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### Animation: PINN vs Ground Truth"),
        mo.md("Watch the PINN pendulum — it tracks well until $t \\approx 2$s "
              "(the training window), then freezes as the network output saturates."),
        mo.Html(_anim.to_html5_video()),
    ]))
    return


@app.cell
def part1c_questions():
    c1 = mo.ui.text_area(
        label="C1: The PINN never saw a full trajectory. What did it use instead "
              "of data to reconstruct the dynamics?",
        full_width=True,
    )
    c2 = mo.ui.text_area(
        label="C2: Compare PINN and MLP accuracy on the short horizon with only "
              "10 training points. Which wins? Why?",
        full_width=True,
    )
    c3 = mo.ui.text_area(
        label="C3: Does the PINN conserve energy? Is this structural like the HNN, "
              "or something else?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Questions — PINN"),
        c1, c2, c3,
    ]))
    return c1, c2, c3


@app.cell
def part1c_retrain_ui():
    retrain_checkbox = mo.ui.checkbox(label="I want to retrain the PINN with a different λ")
    lambda_slider = mo.ui.slider(
        start=-2, stop=2, step=0.5, value=0.0,
        label="log₁₀(λ)",
        show_value=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Optional: Retrain PINN with Different λ"),
        retrain_checkbox,
        mo.md("Check the box above to explore how the physics weight λ affects training."),
    ]))
    return retrain_checkbox, lambda_slider


@app.cell
def part1c_retrain(PINN, config, generate_pinn_training_data,
                   plot_pinn_loss_curves, train_pinn, device,
                   retrain_checkbox, lambda_slider):
    if not retrain_checkbox.value:
        mo.stop(True)

    _lam = 10 ** lambda_slider.value

    _t_obs, _theta_obs, _t_colloc = generate_pinn_training_data(
        n_obs=10, n_colloc=500,
        q0=config["q0"], p0=config["p0"],
        m=config["m"], L=config["L"], g=config["g"],
    )

    _pinn_new = PINN(hidden_dim=config["hidden_dim"], n_layers=4)
    _history = train_pinn(
        _pinn_new, _t_obs, _theta_obs, _t_colloc,
        n_epochs=2000, lr=1e-3, lambda_physics=_lam,
        m=config["m"], L=config["L"], g=config["g"],
        device=str(device), verbose=False,
    )

    _fig = plot_pinn_loss_curves(_history, title=f"PINN Training (λ = {_lam:.2f})")

    mo.output.replace(mo.vstack([
        lambda_slider,
        mo.md(f"**Retraining with λ = {_lam:.2f}** (2000 epochs)"),
        _fig,
        mo.md("Try different values of λ. What happens at λ = 0? At λ = 100?"),
    ]))
    return


# =========================================================================
# PART 2: SYNTHESIS
# =========================================================================


@app.cell(hide_code=True)
def part2_intro():
    mo.output.replace(mo.md("""
    ---
    # Part 2: Synthesis — All Three Side by Side
    """))
    return


@app.cell
def part2_comparison(config, euler_step, generate_trajectory, hnn, mlp, np,
                     pendulum_hamiltonian, pinn, plot_comparison,
                     rollout, symplectic_step, torch):
    _t, _gt = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 20), n_points=2000,
        m=config["m"], L=config["L"], g=config["g"],
    )

    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)

    # MLP rollout
    _mlp_traj = rollout(mlp, _init, dt=0.01, n_steps=2000, step_fn=euler_step)

    # HNN rollout
    _hnn_traj = rollout(hnn, _init, dt=0.01, n_steps=2000, step_fn=symplectic_step)

    # PINN trajectory (move to model device, then back to CPU for plotting)
    _pinn_dev = next(pinn.parameters()).device
    _t_tensor = torch.tensor(_t, dtype=torch.float32, device=_pinn_dev).unsqueeze(1).requires_grad_(True)
    _theta = pinn(_t_tensor)
    _dtheta = torch.autograd.grad(
        _theta, _t_tensor, grad_outputs=torch.ones_like(_theta),
        create_graph=False,
    )[0]
    _p_pinn = config["m"] * config["L"] ** 2 * _dtheta
    _pinn_traj = torch.cat([_theta, _p_pinn], dim=1).detach().cpu()

    _H_fn = lambda q, p: pendulum_hamiltonian(q, p, m=config["m"], L=config["L"], g=config["g"])

    _fig = plot_comparison(
        {"MLP": _mlp_traj, "HNN": _hnn_traj, "PINN": _pinn_traj},
        _gt, _t, _H_fn,
        title="20s Horizon — MLP vs HNN vs PINN",
    )

    mo.output.replace(mo.vstack([
        mo.md("### Full Comparison (20s Horizon)"),
        _fig,
    ]))
    return


@app.cell
def part2_animation(animate_pendulum, config, euler_step, generate_trajectory,
                    hnn, mlp, pinn, rollout, symplectic_step, torch):
    _t, _gt = generate_trajectory(
        q0=config["q0"], p0=config["p0"], t_span=(0, 10), n_points=1000,
        m=config["m"], L=config["L"], g=config["g"],
    )
    _init = torch.tensor([config["q0"], config["p0"]], dtype=torch.float32)
    _mlp_traj = rollout(mlp, _init, dt=0.01, n_steps=1000, step_fn=euler_step)
    _hnn_traj = rollout(hnn, _init, dt=0.01, n_steps=1000, step_fn=symplectic_step)

    # PINN: t -> theta(t), derive p via autograd
    _pinn_dev = next(pinn.parameters()).device
    _t_tensor = torch.tensor(_t, dtype=torch.float32, device=_pinn_dev).unsqueeze(1).requires_grad_(True)
    _theta = pinn(_t_tensor)
    _dtheta = torch.autograd.grad(
        _theta, _t_tensor, grad_outputs=torch.ones_like(_theta),
        create_graph=False,
    )[0]
    _p_pinn = config["m"] * config["L"] ** 2 * _dtheta
    _pinn_traj = torch.cat([_theta, _p_pinn], dim=1).detach().cpu()

    _anim = animate_pendulum(
        [_gt, _mlp_traj, _hnn_traj, _pinn_traj], _t,
        ["Ground Truth", "MLP", "HNN", "PINN"],
        L=config["L"], fps=30, speedup=1.0,
        title="Ground Truth vs MLP vs HNN vs PINN (10s)",
    )

    mo.output.replace(mo.vstack([
        mo.md("### Animation: All Models Side by Side"),
        mo.md("Three failure modes: the MLP spirals out (energy created), "
              "the PINN freezes mid-swing (trained on $t \\in [0, 2]$ only), "
              "and the HNN stays in sync (conservation enforced structurally)."),
        mo.Html(_anim.to_html5_video()),
    ]))
    return


@app.cell
def part2_questions():
    s1 = mo.ui.text_area(
        label="S1: Revisit your prediction from Part 0. Were you right? "
              "What surprised you?",
        full_width=True,
    )
    s2 = mo.ui.text_area(
        label="S2: Rank the three models as world models for a robot controller "
              "that must run for 60 seconds. Justify your ranking.",
        full_width=True,
    )
    s3 = mo.ui.text_area(
        label="S3: HNN enforces physics structurally. PINN enforces it as a loss. "
              "What are the trade-offs of each approach? When would you choose "
              "one over the other?",
        full_width=True,
    )
    s4 = mo.ui.text_area(
        label="S4: If you had abundant data but needed long-horizon predictions, "
              "which would you choose? What if data was scarce?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([
        mo.md("### Synthesis Questions"),
        s1, s2, s3, s4,
    ]))
    return s1, s2, s3, s4


# =========================================================================
# PART 3: EXTENSION STUB
# =========================================================================


@app.cell(hide_code=True)
def part3_stub():
    mo.output.replace(mo.md(r"""
    ---
    # Part 3: Extension — Double Pendulum (Optional)

    > This section is a stub for a future extension.

    The double pendulum is a **chaotic** system: tiny perturbations in initial
    conditions grow exponentially. This means:

    - Even the HNN, which conserves energy perfectly, will **diverge from the
      true trajectory** after a short time
    - Energy conservation is **necessary but not sufficient** for accurate
      long-horizon prediction in chaotic regimes

    **Questions to think about:**

    - If conservation holds but trajectories diverge, what metric should we
      use to evaluate a world model on chaotic systems?
    - What additional inductive biases might help? (Symplectic structure?
      Lyapunov exponent awareness? Ensemble methods?)

    This is an open research question — there is no single right answer.
    """))
    return


if __name__ == "__main__":
    app.run()
