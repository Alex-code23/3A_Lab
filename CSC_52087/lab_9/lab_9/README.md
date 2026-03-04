# Lab 9 — World Models for Physical Systems

**Advanced Deep Learning** — 2025-2026

- **Lecture:** Prof. Ye Zhu
- **Lab:** Dr Guillaume Lachaud

---

One system. Three paradigms. One question: *what makes a trustworthy world model?*

You will compare three neural network approaches to modeling a simple pendulum — from pure data-fitting to physics-informed architectures — and discover why **conservation must be enforced structurally**.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

From the `lab_9/` directory:

```bash
uv sync                     # Install dependencies
uv run marimo edit lab.py   # Open the lab notebook
```

> **Note:** You need the pre-trained model checkpoints in `checkpoints/`.
> If they're missing, ask your instructor.

## Lab Structure

### Part 0 — Physics & Architecture Recap (~15 min)

Review Hamiltonian mechanics and the three modeling paradigms:
- Verify that the ODE solver conserves energy (ground truth)
- Understand MLP, HNN, and PINN conceptually before experiments
- Write your prediction for which model wins on long horizons

### Part 1A — MLP Baseline (~20 min)

Load a pre-trained MLP and observe:
- Short horizon (2s): reasonable predictions
- Long horizon (20s): energy drift, phase portrait spirals

### Part 1B — Hamiltonian Neural Network (~25 min)

Load a pre-trained HNN and:
- Study the forward pass (autograd-derived dynamics)
- Implement the symplectic Euler integrator (your main coding task)
- Compare energy conservation to the MLP

### Part 1C — Physics-Informed Neural Network (~25 min)

Load a pre-trained PINN and observe:
- How sparse data + physics loss reconstructs trajectories
- Short and long horizon behavior
- Optional: retrain with different physics loss weights

### Part 2 — Synthesis (~10 min)

Compare all three models side by side and answer synthesis questions.

## Your Implementation Task

| # | File | Function | What to implement |
|---|------|----------|-------------------|
| 1 | `src/physics_lab/models.py` | `symplectic_step` | Symplectic Euler: update p first, recompute, then update q (~5 lines) |

The notebook has a test cell that checks your implementation.

## Project Structure

```
lab_9/
├── lab.py                              # Marimo notebook (start here)
├── src/physics_lab/
│   ├── physics.py                      # ODE solver, Hamiltonian, data generation
│   ├── models.py                       # MLP, HNN, PINN + integration (TODO here)
│   ├── visualization.py                # Phase portraits, energy plots, comparisons
│   └── utils.py                        # Device detection, seeding
├── checkpoints/
│   ├── mlp.safetensors                 # Pre-trained MLP
│   ├── hnn.safetensors                 # Pre-trained HNN
│   ├── pinn.safetensors                # Pre-trained PINN
│   └── training_config.pt             # Shared physical parameters
├── pyproject.toml
└── README.md
```

## Reflection Questions

The notebook contains 12 analysis questions across three sections:
- **A1–A3**: MLP failure modes and structural limitations
- **B1–B2**: HNN conservation guarantees and their limits
- **C1–C3**: PINN data efficiency and approximate physics
- **S1–S4**: Synthesis — ranking models, trade-offs, design choices
