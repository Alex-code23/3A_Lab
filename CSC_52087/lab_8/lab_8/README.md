# Lab 8 — Sampling in Deep Learning

**Advanced Deep Learning** — 2025-2026

- **Lecture:** Prof. Ye Zhu
- **Lab:** Dr Guillaume Lachaud

---

In this lab you will explore three fundamental sampling strategies — from classical Monte Carlo to modern generative models — and discover why some methods fail in high dimensions while others succeed.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

From the `lab_8/` directory:

```bash
uv sync                     # Install dependencies
uv run marimo edit lab.py   # Open the lab notebook
```

> **Note:** You need the pre-trained diffusion model checkpoint in `checkpoints/`.
> If it's missing, ask your instructor.

## Lab Structure

### Act 1 — Importance Sampling (~30 min)

Estimate expectations by re-weighting samples from a proposal distribution. You will:

- See importance sampling work well in 1D
- Use interactive sliders to crank up the dimension and watch the weights collapse
- Observe the Effective Sample Size (ESS) drop exponentially

### Act 2 — MCMC / Metropolis-Hastings (~35 min)

Build a Markov chain that samples from a 2D mixture of Gaussians. You will:

- Run the MH algorithm on a multimodal target
- Use an interactive slider to tune the proposal step size
- Observe trace plots and autocorrelation diagnostics in real-time
- Find the sweet spot between too-cautious and too-aggressive proposals

### Act 3 — Diffusion Sampling (~45 min)

Generate samples from a pre-trained diffusion model on 2D "Two Moons" data. You will:

- Visualize the full DDPM denoising trajectory from noise to data
- Compare DDPM (stochastic, all steps) vs DDIM (deterministic, fewer steps)
- Use an interactive slider to explore how sample quality degrades with fewer DDIM steps

## Your Implementation Tasks

Open the three source files and complete the TODOs:

| # | File | Function | What to implement |
|---|------|----------|-------------------|
| 1 | `src/sampling_lab/importance.py` | `effective_sample_size` | ESS formula: $1 / \sum w_i^2$ (one line) |
| 2 | `src/sampling_lab/mcmc.py` | `metropolis_hastings` | MH accept/reject step (log-space comparison) |
| 3 | `src/sampling_lab/diffusion.py` | `ddim_sample` | DDIM update: predict $x_0$, then recompose $x_{t-1}$ (two lines) |

Each TODO has a test cell in the notebook that checks your implementation.

## Project Structure

```
lab_8/
├── lab.py                              # Marimo notebook (start here)
├── src/sampling_lab/
│   ├── importance.py                   # Act 1: importance sampling + ESS
│   ├── mcmc.py                         # Act 2: Metropolis-Hastings + diagnostics
│   ├── diffusion.py                    # Act 3: DDPM/DDIM + ScoreNet model
│   └── utils.py                        # Device detection, seeding
├── checkpoints/
│   ├── score_net_moons.safetensors     # Pre-trained 2D diffusion model
│   └── schedule.pt                     # Noise schedule
├── pyproject.toml
└── README.md
```

## Reflection Questions

The notebook contains 4 analysis questions (Q1–Q4) — one per act plus a synthesis question. Take a moment to think through each one after exploring the interactive visualizations.
