import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell
def imports():
    import torch
    import os

    from sampling_lab.utils import get_device, set_seed
    from sampling_lab.importance import (
        gaussian_importance_sampling,
        effective_sample_size,
        run_importance_sampling_experiment,
        plot_weight_distribution,
        plot_ess_vs_dimension,
    )
    from sampling_lab.mcmc import (
        make_default_mixture,
        metropolis_hastings,
        compare_step_sizes,
        plot_chain_2d,
        plot_trace,
        plot_autocorrelation,
        plot_step_size_comparison,
    )
    from sampling_lab.diffusion import (
        load_pretrained,
        make_moons_dataset,
        ddpm_sample,
        ddim_sample,
        plot_samples,
        plot_trajectory,
        plot_ddpm_vs_ddim,
        plot_steps_degradation,
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
        compare_step_sizes,
        ddim_sample,
        ddpm_sample,
        device,
        effective_sample_size,
        gaussian_importance_sampling,
        load_pretrained,
        make_default_mixture,
        make_moons_dataset,
        metropolis_hastings,
        os,
        plot_autocorrelation,
        plot_chain_2d,
        plot_ddpm_vs_ddim,
        plot_ess_vs_dimension,
        plot_samples,
        plot_step_size_comparison,
        plot_steps_degradation,
        plot_trace,
        plot_trajectory,
        plot_weight_distribution,
        run_importance_sampling_experiment,
        torch,
    )


@app.cell(hide_code=True)
def title():
    mo.output.replace(mo.md("""
    # Lab 8 — Sampling in Deep Learning

    **Advanced Deep Learning** — 2025-2026

    - **Lecture:** Prof. Ye Zhu
    - **Lab:** Dr Guillaume Lachaud

    ---

    This lab explores three fundamental sampling strategies, from classical to modern:

    1. **Importance Sampling** — weight-based estimation and its high-dimensional collapse
    2. **MCMC / Metropolis-Hastings** — iterative correction with convergence diagnostics
    3. **Diffusion Models** — learned denoising as a sampling mechanism
    """))
    return


# =========================================================================
# ACT 1: IMPORTANCE SAMPLING
# =========================================================================


@app.cell(hide_code=True)
def act1_intro():
    mo.output.replace(mo.md("""
    ---
    # Act 1: Importance Sampling (~30 min)

    ## The Idea

    We want to estimate $\\mathbb{E}_{p}[f(x)]$ but can't sample directly from $p$.
    Instead, we sample from a **proposal** distribution $q$ and re-weight:

    $$\\mathbb{E}_{p}[f(x)] = \\mathbb{E}_{q}\\left[f(x) \\cdot \\frac{p(x)}{q(x)}\\right] \\approx \\frac{1}{N}\\sum_{i=1}^{N} f(x_i) \\cdot w_i, \\quad x_i \\sim q$$

    where $w_i = p(x_i) / q(x_i)$ are the **importance weights**.

    ## The Problem

    When $p$ and $q$ don't overlap well, most weights become negligible and a few dominate.
    This **weight collapse** gets exponentially worse in high dimensions.

    ## Effective Sample Size (ESS)

    The **ESS** measures how many "useful" samples we actually have:

    $$\\text{ESS} = \\frac{1}{\\sum_{i=1}^N \\hat{w}_i^2}$$

    where $\\hat{w}_i$ are normalized weights (summing to 1). If ESS $\\approx N$, all samples contribute equally. If ESS $\\approx 1$, one sample dominates.

    ### Your task

    Open `src/sampling_lab/importance.py` and implement the `effective_sample_size` function.
    """))
    return


@app.cell
def act1_test_ess(effective_sample_size, torch):
    _uniform_w = torch.ones(1000) / 1000
    _ess_uniform = effective_sample_size(_uniform_w)

    _peaked_w = torch.zeros(1000)
    _peaked_w[0] = 1.0
    _ess_peaked = effective_sample_size(_peaked_w)

    _ess_u = _ess_uniform.item() if isinstance(_ess_uniform, torch.Tensor) else _ess_uniform
    _ess_p = _ess_peaked.item() if isinstance(_ess_peaked, torch.Tensor) else _ess_peaked

    if abs(_ess_u - 1000) < 1 and abs(_ess_p - 1) < 0.1:
        _result = mo.callout(
            mo.md(f"**ESS test passed!**\n\n"
                   f"- Uniform weights → ESS = {_ess_u:.0f} (expected 1000)\n"
                   f"- Single-peaked weights → ESS = {_ess_p:.1f} (expected 1)"),
            kind="success",
        )
    elif _ess_u == 0 and _ess_p == 0:
        _result = mo.callout(
            mo.md("**ESS is 0.0** — You haven't implemented `effective_sample_size` yet.\n\n"
                   "Open `src/sampling_lab/importance.py` and complete the TODO."),
            kind="warn",
        )
    else:
        _result = mo.callout(
            mo.md(f"**ESS test failed.**\n\n"
                   f"- Uniform weights → ESS = {_ess_u:.1f} (expected 1000)\n"
                   f"- Single-peaked weights → ESS = {_ess_p:.1f} (expected 1)\n\n"
                   "Check your implementation."),
            kind="danger",
        )

    mo.output.replace(mo.vstack([mo.md("### Test Your ESS Implementation"), _result]))
    return


@app.cell(hide_code=True)
def act1_explorer_header():
    mo.output.replace(mo.md("""
    ### Interactive Explorer

    Use the sliders below to explore how **dimension** and **target offset** affect
    the importance weights. The proposal is always $\\mathcal{N}(0, I)$ and the target
    is $\\mathcal{N}(\\mu \\cdot \\mathbf{1}, I)$ where $\\mu$ is the offset.
    """))
    return


@app.cell
def act1_explorer_controls():
    dim_slider = mo.ui.slider(
        start=1, stop=50, step=1, value=1,
        label="Dimension",
        show_value=True,
    )
    offset_slider = mo.ui.slider(
        start=0.5, stop=5.0, step=0.5, value=2.0,
        label="Target offset ($\\mu$)",
        show_value=True,
    )
    mo.output.replace(mo.hstack([dim_slider, offset_slider], gap=2))
    return dim_slider, offset_slider


@app.cell
def act1_explorer_run(
    dim_slider,
    effective_sample_size,
    gaussian_importance_sampling,
    offset_slider,
    plot_weight_distribution,
    torch,
):
    _d = dim_slider.value
    _mu = offset_slider.value

    if _d == 1:
        _proposal = torch.distributions.Normal(loc=0.0, scale=1.0)
        def _target_log_prob(x):
            return torch.distributions.Normal(loc=_mu, scale=1.0).log_prob(x)
    else:
        _proposal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(_d), covariance_matrix=torch.eye(_d),
        )
        _mean = torch.full((_d,), _mu)
        def _target_log_prob(x, mean=_mean):
            diff = x - mean
            return -0.5 * (diff * diff).sum(dim=-1)

    def _f(x):
        return torch.ones(x.shape[0]) if x.dim() > 1 else torch.ones_like(x)

    _est, _w, _ = gaussian_importance_sampling(_f, _target_log_prob, _proposal, n_samples=10000)
    _ess = effective_sample_size(_w)
    _ess_val = _ess.item() if isinstance(_ess, torch.Tensor) else _ess

    _fig = plot_weight_distribution(
        _w, title=f"Importance Weights — dim={_d}, offset={_mu}",
    )

    _ess_pct = _ess_val / 10000 * 100
    mo.output.replace(mo.vstack([
        mo.md(f"**ESS = {_ess_val:.0f}** / 10,000 ({_ess_pct:.1f}%) — Estimate = {_est.item():.3f} (true = 1.0)"),
        _fig,
    ]))
    return


@app.cell(hide_code=True)
def act1_sweep_header():
    mo.output.replace(mo.md("""
    ### Summary: ESS vs Dimension

    The plot below runs a full sweep to show the collapse trend.
    """))
    return


@app.cell
def act1_dimension_sweep(os, plot_ess_vs_dimension, run_importance_sampling_experiment):
    os.makedirs("plots", exist_ok=True)
    _results = run_importance_sampling_experiment(
        dims=[1, 2, 5, 10, 20, 50],
        n_samples=10000,
        target_offset=3.0,
    )
    _fig = plot_ess_vs_dimension(_results)
    _fig.savefig("plots/act1_ess_vs_dim.pdf", bbox_inches="tight")
    mo.output.replace(_fig)
    return


@app.cell
def act1_question():
    q1 = mo.ui.text_area(
        label="Q1: Why does importance sampling collapse in high dimensions? What role does ESS play as a diagnostic?",
        full_width=True,
    )
    mo.output.replace(q1)
    return (q1,)


# =========================================================================
# ACT 2: MCMC / METROPOLIS-HASTINGS
# =========================================================================


@app.cell(hide_code=True)
def act2_intro():
    mo.output.replace(mo.md("""
    ---
    # Act 2: MCMC / Metropolis-Hastings (~35 min)

    ## The Idea

    Instead of weighting samples from a fixed proposal, we build a **Markov chain**
    whose stationary distribution is the target $p(x)$.

    The **Metropolis-Hastings** algorithm:

    1. Start at some point $x_0$
    2. Propose: $x' = x_t + \\epsilon$, where $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I)$
    3. Compute acceptance ratio: $\\alpha = \\min\\left(1, \\frac{p(x')}{p(x_t)}\\right)$
    4. Accept $x_{t+1} = x'$ with probability $\\alpha$, otherwise $x_{t+1} = x_t$
    5. Repeat

    The chain converges to sampling from $p$ regardless of the starting point (in theory).

    ## Key Diagnostic: Step Size

    - **Too small** $\\sigma$: high acceptance rate but slow exploration (chain "sticks")
    - **Too large** $\\sigma$: low acceptance rate, many rejections, also slow
    - **Sweet spot**: ~25-50% acceptance rate (for high-dimensional targets)

    ### Your task

    Open `src/sampling_lab/mcmc.py` and implement the accept/reject step in `metropolis_hastings`.
    """))
    return


@app.cell
def act2_test(make_default_mixture, metropolis_hastings, torch):
    _target_log_prob, _, _ = make_default_mixture()
    _x_init = torch.zeros(2)

    _chain, _acc = metropolis_hastings(
        _target_log_prob, _x_init, n_steps=2000, proposal_std=0.8,
    )

    if _acc == 0.0:
        _result = mo.callout(
            mo.md("**Acceptance rate is 0%** — all proposals are rejected.\n\n"
                   "You haven't implemented the MH accept/reject step yet.\n"
                   "Open `src/sampling_lab/mcmc.py` and complete the TODO."),
            kind="warn",
        )
    elif _acc < 0.05:
        _result = mo.callout(
            mo.md(f"**Acceptance rate = {_acc:.1%}** — very low.\n\n"
                   "Check your implementation — the accept/reject logic may be inverted."),
            kind="danger",
        )
    else:
        _result = mo.callout(
            mo.md(f"**Acceptance rate = {_acc:.1%}** — chain is running!"),
            kind="success",
        )

    mo.output.replace(mo.vstack([mo.md("### Test Your MH Implementation"), _result]))
    return


@app.cell(hide_code=True)
def act2_explorer_header():
    mo.output.replace(mo.md("""
    ### Interactive Explorer

    Drag the step-size slider and watch how the chain, trace plot, and autocorrelation change.
    Look for the sweet spot between too-cautious and too-aggressive proposals.
    """))
    return


@app.cell
def act2_explorer_control():
    sigma_slider = mo.ui.slider(
        start=0.05, stop=10.0, step=0.05, value=0.8,
        label="Proposal step size ($\\sigma$)",
        show_value=True,
    )
    mo.output.replace(sigma_slider)
    return (sigma_slider,)


@app.cell
def act2_explorer_run(
    make_default_mixture,
    metropolis_hastings,
    plot_chain_2d,
    plot_trace,
    plot_autocorrelation,
    sigma_slider,
    torch,
):
    _target_log_prob, _, _ = make_default_mixture()
    _x_init = torch.zeros(2)
    _sigma = sigma_slider.value

    _chain, _acc = metropolis_hastings(
        _target_log_prob, _x_init, n_steps=5000, proposal_std=_sigma,
    )

    _fig_chain = plot_chain_2d(
        _chain, target_log_prob=_target_log_prob,
        title=f"MH Chain — $\\sigma={_sigma:.2f}$, acc. rate={_acc:.1%}",
    )
    _fig_trace = plot_trace(_chain, title=f"Trace Plot ($\\sigma={_sigma:.2f}$)")
    _fig_acf = plot_autocorrelation(_chain, max_lag=200,
                                     title=f"Autocorrelation ($\\sigma={_sigma:.2f}$)")

    mo.output.replace(mo.vstack([
        mo.md(f"**Acceptance rate: {_acc:.1%}**"),
        _fig_chain,
        _fig_trace,
        _fig_acf,
    ]))
    return


@app.cell(hide_code=True)
def act2_comparison_header():
    mo.output.replace(mo.md("""
    ### Summary: Step Size Comparison

    Side-by-side view of four different step sizes for quick comparison.
    """))
    return


@app.cell
def act2_step_comparison(compare_step_sizes, make_default_mixture, os, plot_step_size_comparison, torch):
    os.makedirs("plots", exist_ok=True)
    _target_log_prob, _, _ = make_default_mixture()
    _x_init = torch.zeros(2)

    _results = compare_step_sizes(
        _target_log_prob,
        step_sizes=[0.05, 0.5, 2.0, 10.0],
        x_init=_x_init,
        n_steps=5000,
    )
    _fig = plot_step_size_comparison(_results, target_log_prob=_target_log_prob)
    _fig.savefig("plots/act2_step_sizes.pdf", bbox_inches="tight")

    _acc_table = "| Step size ($\\sigma$) | Acceptance rate |\n|---|---|\n"
    for _r in _results:
        _acc_table += f"| {_r['step_size']} | {_r['acceptance_rate']:.1%} |\n"

    mo.output.replace(mo.vstack([_fig, mo.md(_acc_table)]))
    return


@app.cell
def act2_question():
    q2 = mo.ui.text_area(
        label="Q2: What is the relationship between step size, acceptance rate, and mixing quality? How would you choose a good step size in practice?",
        full_width=True,
    )
    mo.output.replace(q2)
    return (q2,)


# =========================================================================
# ACT 3: DIFFUSION SAMPLING
# =========================================================================


@app.cell(hide_code=True)
def act3_intro():
    mo.output.replace(mo.md("""
    ---
    # Act 3: Diffusion Sampling (~45 min)

    ## The Idea

    Diffusion models define a **forward process** that gradually adds noise to data:

    $$q(x_t | x_0) = \\mathcal{N}\\left(x_t; \\sqrt{\\bar{\\alpha}_t}\\, x_0,\\; (1 - \\bar{\\alpha}_t)\\, I\\right)$$

    and learn a **reverse process** that denoises step by step:

    $$p_\\theta(x_{t-1} | x_t) \\approx \\mathcal{N}\\left(x_{t-1}; \\mu_\\theta(x_t, t),\\; \\sigma_t^2 I\\right)$$

    A neural network $\\epsilon_\\theta(x_t, t)$ is trained to predict the noise added at each step.

    ## DDPM vs DDIM

    - **DDPM** (Denoising Diffusion Probabilistic Models): stochastic reverse process, goes through all $T$ steps
    - **DDIM** (Denoising Diffusion Implicit Models): deterministic reverse process, can **skip steps** for faster sampling

    The DDIM update (with $\\eta = 0$, deterministic):

    $$\\hat{x}_0 = \\frac{x_t - \\sqrt{1 - \\bar{\\alpha}_t}\\, \\epsilon_\\theta(x_t, t)}{\\sqrt{\\bar{\\alpha}_t}}$$

    $$x_{t-1} = \\sqrt{\\bar{\\alpha}_{t-1}}\\, \\hat{x}_0 + \\sqrt{1 - \\bar{\\alpha}_{t-1}}\\, \\epsilon_\\theta(x_t, t)$$

    ### Your task

    Open `src/sampling_lab/diffusion.py` and implement the DDIM update step in `ddim_sample`.

    > **Note:** You need a pre-trained model at `checkpoints/score_net_moons.safetensors`.
    > If you don't have it, ask your instructor.
    """))
    return


@app.cell
def act3_load_model(device, load_pretrained, make_moons_dataset, os):
    _ckpt_dir = "checkpoints"
    _model_ok = (
        os.path.exists(os.path.join(_ckpt_dir, "score_net_moons.safetensors"))
        and os.path.exists(os.path.join(_ckpt_dir, "schedule.pt"))
    )

    if not _model_ok:
        mo.output.replace(mo.callout(
            mo.md("**Pre-trained model not found** in `checkpoints/`.\n\n"
                   "Ask your instructor for the checkpoint files:\n"
                   "- `score_net_moons.safetensors`\n"
                   "- `schedule.pt`"),
            kind="danger",
        ))
        mo.stop(True)

    diff_model, diff_betas, diff_alphas_cumprod, diff_n_steps = load_pretrained(
        checkpoint_dir=_ckpt_dir, device=device,
    )
    reference_data = make_moons_dataset(n_samples=2000)

    mo.output.replace(mo.md(f"""
    ### Model Loaded

    | Component | Value |
    |-----------|-------|
    | **Diffusion steps** | {diff_n_steps} |
    | **Model parameters** | {sum(p.numel() for p in diff_model.parameters()):,} |
    | **Device** | `{device}` |
    | **Training data** | Two Moons (2D) |
    """))
    return diff_alphas_cumprod, diff_betas, diff_model, diff_n_steps, reference_data


@app.cell
def act3_reference_data(plot_samples, reference_data):
    _fig = plot_samples(reference_data, title="Reference Data (Two Moons)")
    mo.output.replace(mo.vstack([mo.md("### Training Data Distribution"), _fig]))
    return


@app.cell(hide_code=True)
def act3_ddpm_header():
    mo.output.replace(mo.md("""
    ### DDPM Sampling

    First, let's generate samples using the full DDPM reverse process (all T steps).
    """))
    return


@app.cell
def act3_ddpm_run(ddpm_sample, device, diff_betas, diff_alphas_cumprod, diff_model, os, plot_trajectory, reference_data):
    os.makedirs("plots", exist_ok=True)
    ddpm_samples, ddpm_traj = ddpm_sample(
        diff_model, diff_betas, diff_alphas_cumprod,
        n_samples=1000, device=device,
    )
    _fig = plot_trajectory(ddpm_traj, n_snapshots=6, reference=reference_data)
    _fig.savefig("plots/act3_trajectory.pdf", bbox_inches="tight")
    mo.output.replace(mo.vstack([mo.md("### DDPM Denoising Trajectory"), _fig]))
    return ddpm_samples, ddpm_traj


@app.cell
def act3_ddim_test(ddim_sample, device, diff_alphas_cumprod, diff_model, diff_n_steps):
    _samples, _traj = ddim_sample(
        diff_model, diff_alphas_cumprod,
        n_steps_total=diff_n_steps,
        n_steps_sample=50,
        n_samples=500,
        device=device,
    )

    _first = _traj[0] if len(_traj) > 0 else None
    _last = _traj[-1] if len(_traj) > 0 else None
    if _first is not None and _last is not None:
        _diff = (_first - _last).abs().mean().item()
        if _diff < 0.01:
            _result = mo.callout(
                mo.md("**DDIM samples look like pure noise** — the update step isn't implemented.\n\n"
                       "Open `src/sampling_lab/diffusion.py` and complete the TODO in `ddim_sample`."),
                kind="warn",
            )
        else:
            _result = mo.callout(
                mo.md(f"**DDIM sampling is working** — using 50 steps instead of {diff_n_steps}."),
                kind="success",
            )
    else:
        _result = mo.callout(mo.md("Could not verify DDIM output."), kind="warn")

    mo.output.replace(mo.vstack([mo.md("### Test Your DDIM Implementation"), _result]))
    return


@app.cell
def act3_comparison(ddpm_sample, ddim_sample, device, diff_betas, diff_alphas_cumprod, diff_model, diff_n_steps, os, plot_ddpm_vs_ddim, reference_data):
    os.makedirs("plots", exist_ok=True)
    _ddpm, _ = ddpm_sample(
        diff_model, diff_betas, diff_alphas_cumprod,
        n_samples=1000, device=device,
    )
    _ddim, _ = ddim_sample(
        diff_model, diff_alphas_cumprod,
        n_steps_total=diff_n_steps, n_steps_sample=50,
        n_samples=1000, device=device,
    )
    _fig = plot_ddpm_vs_ddim(_ddpm, _ddim, reference=reference_data)
    _fig.savefig("plots/act3_ddpm_vs_ddim.pdf", bbox_inches="tight")
    mo.output.replace(mo.vstack([mo.md("### DDPM vs DDIM (50 steps)"), _fig]))
    return


@app.cell(hide_code=True)
def act3_steps_header():
    mo.output.replace(mo.md("""
    ### Interactive Explorer: DDIM Step Count

    Drag the slider to see how reducing the number of DDIM steps degrades sample quality.
    Fewer steps = faster generation, but at what cost?
    """))
    return


@app.cell
def act3_steps_control():
    steps_slider = mo.ui.slider(
        start=2, stop=200, step=1, value=50,
        label="DDIM sampling steps",
        show_value=True,
    )
    mo.output.replace(steps_slider)
    return (steps_slider,)


@app.cell
def act3_steps_run(
    ddim_sample,
    device,
    diff_alphas_cumprod,
    diff_model,
    diff_n_steps,
    plot_samples,
    reference_data,
    steps_slider,
):
    _n = steps_slider.value
    _samples, _ = ddim_sample(
        diff_model, diff_alphas_cumprod,
        n_steps_total=diff_n_steps,
        n_steps_sample=_n,
        n_samples=1000,
        device=device,
    )
    _fig = plot_samples(_samples, title=f"DDIM — {_n} steps", reference=reference_data)
    mo.output.replace(_fig)
    return


@app.cell
def act3_question():
    q3 = mo.ui.text_area(
        label="Q3: Compare DDPM and DDIM. What are the trade-offs between sample quality and speed? How few DDIM steps can you use before quality visibly degrades?",
        full_width=True,
    )
    mo.output.replace(q3)
    return (q3,)


# =========================================================================
# SYNTHESIS
# =========================================================================


@app.cell
def synthesis_intro():
    mo.output.replace(mo.md("""
    ---
    # Synthesis

    Reflect on all three sampling methods you've explored.
    """))
    return


@app.cell
def synthesis_question():
    q4 = mo.ui.text_area(
        label="Q4: Importance sampling collapses in high dimensions, yet diffusion models generate high-dimensional images. What is the key insight that makes diffusion work where IS fails?",
        full_width=True,
    )
    mo.output.replace(q4)
    return (q4,)


if __name__ == "__main__":
    app.run()
