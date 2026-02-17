# Lab 7 — When Knowledge Distillation Fails

**Advanced Deep Learning** — 2025-2026

- **Lecture:** Prof. Ye Zhu
- **Lab:** Dr Guillaume Lachaud

---

In this lab you will implement the Knowledge Distillation (KD) loss function and run three experiments that reveal when distillation **helps** and when it **fails**.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) or [pixi](https://pixi.sh/) package manager

## Setup

From the `lab_7/` directory:

**With uv:**

```bash
uv sync                        # Install dependencies
uv run marimo edit notebook.py  # Open the lab notebook
```

**With pixi:**

```bash
pixi install                   # Install dependencies
pixi run lab                   # Open the lab notebook
```

## Lab Structure

### Part 1 — Understanding KD

Read the theory introduction and answer the two conceptual questions (Q1–Q2) about temperature scaling and the $T^2$ factor.

### Your Implementation Task

Open `src/distill_lab/distillation.py` and complete the **3 TODOs** inside the `distillation_loss` function:

1. **TODO 1** — Compute the soft probability distributions (softmax with temperature)
2. **TODO 2** — Compute the KL-divergence loss between soft distributions (scaled by $T^2$)
3. **TODO 3** — Combine the soft loss and the hard cross-entropy loss using $\alpha$

A test cell in the notebook will verify your implementation (expected loss $\approx 1.51$).

### Scenario A — Capacity Mismatch

A ResNet-34 teacher distills into a ResNet-18 ($\sim$11M params) and a linear classifier ($\sim$30K params). You will observe that a very weak student cannot absorb the teacher's knowledge.

### Scenario B — Imbalanced Data Bias

A ResNet-18 teacher distills into a ResNet-18 student (trained from scratch) on the DermaMNIST medical imaging dataset. You will examine how class imbalance biases propagate through distillation.

### Scenario C — Temperature Sensitivity

The same ResNet-34 $\to$ ResNet-18 pair from Scenario A is trained across temperatures $T \in \{1, 4, 8, 20\}$. You will find the optimal temperature range and understand why extremes hurt.

## What to Submit

The notebook contains **12 text-area questions** throughout the scenarios:

| Section | Questions |
|---------|-----------|
| Conceptual | Q1, Q2 |
| Scenario A | Q3, Q4 |
| Scenario B | Q5, Q6 |
| Scenario C | Q7, Q8 |
| Synthesis | S1, S2, S3, S4 |

Answer all of them directly in the notebook.

## Submission

1. Scroll to the **Submission** section at the bottom of the notebook
2. Enter your full name
3. Click **Create submission archive**
4. Upload the generated `lab_7_<your_name>.zip` file to the course platform

The archive contains your notebook, your `distillation.py` implementation, and all your answers exported as JSON.
