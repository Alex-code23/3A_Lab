import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)


def compute_metrics(preds, labels, class_names=None):
    """Compute classification metrics.

    Returns:
        dict with accuracy, per_class_f1, confusion_matrix, classification_report
    """
    acc = accuracy_score(labels, preds)
    f1_per_class = f1_score(labels, preds, average=None)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)

    return {
        "accuracy": acc,
        "per_class_f1": f1_per_class,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_training_curves(histories, title="Training Curves"):
    """Plot loss and accuracy curves for multiple models.

    Args:
        histories: dict mapping model_name → history dict (with train_loss, train_acc, test_loss, test_acc)
        title: Plot title

    Returns:
        matplotlib.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax1.plot(epochs, hist["test_loss"], label=name)
        ax2.plot(epochs, hist["test_acc"], label=name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_capacity_results(results):
    """Grouped bar chart: standalone vs distilled accuracy per student.

    Args:
        results: dict from run_scenario_a

    Returns:
        matplotlib.Figure
    """
    student_names = [k for k in results if k not in ("teacher_acc",)]
    standalone_accs = [results[n]["standalone"]["test_acc"] for n in student_names]
    distilled_accs = [results[n]["distilled"]["test_acc"] for n in student_names]

    x = np.arange(len(student_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, standalone_accs, width, label="Standalone", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, distilled_accs, width, label="Distilled", color="#DD8452")

    ax.axhline(y=results["teacher_acc"], color="red", linestyle="--", alpha=0.7, label="Teacher")

    ax.set_xlabel("Student Model")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Scenario A: Capacity Mismatch — Standalone vs Distilled")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({results[n]['params']})" for n in student_names])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return fig


def format_capacity_table(results):
    """Format Scenario A results as a markdown table.

    Returns:
        str (markdown)
    """
    lines = [
        "| Model | Parameters | Standalone Acc | Distilled Acc | Delta |",
        "|-------|-----------|---------------|--------------|-------|",
    ]
    teacher_acc = results["teacher_acc"]
    lines.append(f"| **Teacher (ResNet-34)** | — | {teacher_acc:.2%} | — | — |")

    for name in [k for k in results if k not in ("teacher_acc",)]:
        r = results[name]
        s_acc = r["standalone"]["test_acc"]
        d_acc = r["distilled"]["test_acc"]
        delta = d_acc - s_acc
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {name} | {r['params']} | {s_acc:.2%} | {d_acc:.2%} | {sign}{delta:.2%} |")

    return "\n".join(lines)


def plot_confusion_matrices(teacher_metrics, standalone_metrics, distilled_metrics,
                            class_names=None, title_prefix=""):
    """Plot 3 side-by-side row-normalized confusion matrices.

    Returns:
        matplotlib.Figure
    """
    fig, (*axes, cax) = plt.subplots(
        1, 4, figsize=(20, 5),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.05]},
    )

    for ax, metrics, title in zip(
        axes,
        [teacher_metrics, standalone_metrics, distilled_metrics],
        ["Teacher", "Standalone", "Distilled"],
    ):
        cm = metrics["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix}{title}\n(acc: {metrics['accuracy']:.2%})")

        if class_names:
            ax.set_xticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=8)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

    fig.colorbar(im, cax=cax, label="Proportion")
    fig.tight_layout()
    return fig


def plot_per_class_f1(metrics_dict, class_names):
    """Grouped bar chart of F1 score per class for multiple models.

    Args:
        metrics_dict: dict mapping model_name → metrics (from compute_metrics)
        class_names: list of class name strings

    Returns:
        matplotlib.Figure
    """
    n_classes = len(class_names)
    n_models = len(metrics_dict)
    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        f1s = metrics["per_class_f1"]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=name, color=colors[i % len(colors)])

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_class_distribution(distribution, class_names):
    """Bar chart of class distribution with percentage labels.

    Args:
        distribution: dict mapping class_index → count
        class_names: list of class name strings

    Returns:
        matplotlib.Figure
    """
    counts = [distribution.get(i, 0) for i in range(len(class_names))]
    total = sum(counts)
    percentages = [c / total * 100 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(class_names)), counts, color="#4C72B0")

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_temperature_curve(results):
    """Line chart: test accuracy vs temperature.

    Args:
        results: dict from run_scenario_c (keys are temperature values + 'teacher_acc')

    Returns:
        matplotlib.Figure
    """
    temps = sorted(k for k in results if isinstance(k, (int, float)))
    accs = [results[t]["test_acc"] for t in temps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, accs, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax.axhline(y=results["teacher_acc"], color="red", linestyle="--", alpha=0.7, label="Teacher")

    for t, a in zip(temps, accs):
        ax.annotate(f"{a:.1%}", (t, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Scenario C: Temperature Sensitivity — ResNet-34 → ResNet-18")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def visualize_temperature_effect(logits, temperatures=(1, 2, 4, 8, 16, 20)):
    """Visualize how temperature changes the softmax distribution.

    Args:
        logits: tensor of shape (num_samples, num_classes) — a few sample logits
        temperatures: tuple of temperature values to show

    Returns:
        matplotlib.Figure
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()

    n_samples = min(logits.shape[0], 4)
    n_temps = len(temperatures)

    fig, axes = plt.subplots(n_samples, n_temps, figsize=(3 * n_temps, 3 * n_samples))
    if n_samples == 1:
        axes = axes.unsqueeze(0) if isinstance(axes, torch.Tensor) else np.expand_dims(axes, 0)

    for i in range(n_samples):
        for j, T in enumerate(temperatures):
            ax = axes[i][j]
            probs = F.softmax(logits[i] / T, dim=0).numpy()
            ax.bar(range(len(probs)), probs, color="#4C72B0")
            ax.set_ylim(0, 1)
            ax.set_title(f"Sample {i + 1}, T={T}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Probability")
            if i == n_samples - 1:
                ax.set_xlabel("Class")

    fig.suptitle("Effect of Temperature on Softmax Distributions", fontsize=12)
    fig.tight_layout()
    return fig
