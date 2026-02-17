import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo


@app.cell
def imports():
    import torch
    import os

    from src.distill_lab.utils import get_device, set_seed
    from src.distill_lab.datasets import (
        get_cifar10_loaders, get_dermamnist_loaders,
        check_cifar10, check_dermamnist,
        get_class_distribution_fast,
        DERMAMNIST_CLASSES,
    )
    from src.distill_lab.distillation import distillation_loss
    from src.distill_lab.training import run_scenario_a, run_scenario_b, run_scenario_c
    from src.distill_lab.submission import create_archive
    from src.distill_lab.evaluation import (
        compute_metrics,
        plot_training_curves, plot_capacity_results, format_capacity_table,
        plot_confusion_matrices, plot_per_class_f1,
        plot_class_distribution, plot_temperature_curve, visualize_temperature_effect,
    )


    set_seed(42)
    device = get_device()

    _cifar_ok = check_cifar10()
    _derma_ok = check_dermamnist()

    mo.output.replace(mo.md(f"""
    ## Environment Setup

    | Component | Status |
    |-----------|--------|
    | **Device** | `{device}` |
    | **CIFAR-10** | {"Ready" if _cifar_ok else "Downloading..."} |
    | **DermaMNIST** | {"Ready" if _derma_ok else "Downloading..."} |
    """))
    return (
        DERMAMNIST_CLASSES,
        compute_metrics,
        create_archive,
        device,
        distillation_loss,
        format_capacity_table,
        get_cifar10_loaders,
        get_class_distribution_fast,
        get_dermamnist_loaders,
        os,
        plot_capacity_results,
        plot_class_distribution,
        plot_confusion_matrices,
        plot_per_class_f1,
        plot_temperature_curve,
        plot_training_curves,
        run_scenario_a,
        run_scenario_b,
        run_scenario_c,
        torch,
        visualize_temperature_effect,
    )


@app.cell(hide_code=True)
def title():
    mo.output.replace(mo.md("""
    # Lab 7 — When Knowledge Distillation Fails

    **Advanced Deep Learning** — 2025-2026

    - **Lecture:** Prof. Ye Zhu
    - **Lab:** Dr Guillaume Lachaud
    """))
    return


@app.cell
def part1_intro():
    mo.output.replace(mo.md("""
    # Part 1: Understanding Knowledge Distillation

    ## What is Knowledge Distillation?

    Knowledge Distillation (KD) transfers knowledge from a large **teacher** model to a smaller **student** model.
    The key insight is that the teacher's output probabilities contain richer information than hard labels alone.

    For example, if a teacher classifies a handwritten "7" as:
    - 7: 90%, 1: 5%, 2: 3%, 9: 2%

    This tells the student that "7" looks somewhat like "1" and "2" — information that the hard label [0,0,0,0,0,0,0,1,0,0] doesn't capture.

    ## The KD Loss Function

    The distillation loss combines two terms:

    $$\\mathcal{L}_{KD} = \\alpha \\cdot T^2 \\cdot \\text{KL}\\left(\\sigma\\left(\\frac{z_s}{T}\\right) \\| \\sigma\\left(\\frac{z_t}{T}\\right)\\right) + (1 - \\alpha) \\cdot \\text{CE}(z_s, y)$$

    Where:
    - $z_s, z_t$ are student and teacher logits
    - $T$ is the **temperature** (higher = softer distributions)
    - $\\alpha$ balances soft vs hard loss
    - $\\sigma$ is the softmax function

    ### Your task

    Open `distill_lab/distillation.py` and implement the three TODOs in the `distillation_loss` function.
    """))
    return


@app.cell
def conceptual_questions():
    q1 = mo.ui.text_area(
        label="Q1: Why do we divide logits by temperature before applying softmax? What happens when $T \\to \\infty$?",
        full_width=True,
    )
    q2 = mo.ui.text_area(
        label="Q2: Why do we multiply the KL divergence by $T^2$? (Hint: think about gradient magnitudes)",
        full_width=True,
    )
    mo.output.replace(mo.md("### Conceptual Questions\n\nAnswer these before implementing:"))
    return q1, q2


@app.cell
def display_questions(q1, q2):
    mo.output.replace(mo.vstack([q1, q2]))
    return


@app.cell
def temperature_viz_header():
    mo.output.replace(mo.md("### Visualizing the Temperature Effect"))
    return


@app.cell
def temperature_viz_plot(torch, visualize_temperature_effect):
    _sample_logits = torch.tensor([
        [5.0, 2.0, 1.0, 0.5, 0.1, -0.5, -1.0, -1.5, -2.0, -3.0],
        [3.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -2.5],
    ])
    visualize_temperature_effect(_sample_logits, temperatures=(1, 2, 4, 8, 16, 20))
    return


@app.cell
def kd_test_header():
    mo.output.replace(mo.md("### Test Your Implementation"))
    return


@app.cell
def kd_test_run(distillation_loss, torch):
    torch.manual_seed(0)
    _student_logits = torch.randn(4, 10)
    _teacher_logits = torch.randn(4, 10)
    _labels = torch.tensor([0, 1, 2, 3])

    _loss = distillation_loss(_student_logits, _teacher_logits, _labels, temperature=4.0, alpha=0.7)
    _loss_val = _loss.item()

    if _loss_val == 0.0:
        _result = mo.callout(
            mo.md("**Loss is 0.0** \u2014 You haven't implemented the distillation loss yet.\n\n"
                   "Open `distill_lab/distillation.py` and complete the three TODO blocks."),
            kind="warn",
        )
    elif abs(_loss_val - 1.51) < 0.15:
        _result = mo.callout(
            mo.md(f"**Loss = {_loss_val:.4f}** \u2014 Correct! Your implementation looks good."),
            kind="success",
        )
    else:
        _result = mo.callout(
            mo.md(f"**Loss = {_loss_val:.4f}** \u2014 This doesn't match the expected value (~1.51).\n\n"
                   "Check your implementation and try again."),
            kind="danger",
        )

    mo.output.replace(_result)
    return


@app.cell
def timing_tip():
    mo.output.replace(mo.callout(
        mo.md(
            "**Tip:** Once your implementation passes the test above, start **Scenario A** "
            "right away — the training takes a while on CPU. You can answer the conceptual "
            "questions (Q1–Q2) while it runs."
        ),
        kind="info",
    ))
    return


@app.cell
def load_datasets(get_cifar10_loaders, get_dermamnist_loaders, torch):
    mo.output.replace(mo.md("## Loading Datasets"))

    cifar_train_loader, cifar_test_loader = get_cifar10_loaders(batch_size=128)
    derma_train_loader, derma_val_loader, derma_test_loader = get_dermamnist_loaders(batch_size=64)

    # Use a random 25K subset of CIFAR-10 training data to keep training
    # times manageable on CPU. The full 50K set is not needed to observe
    # the distillation effects studied in this lab.
    _full_train = cifar_train_loader.dataset
    _indices = torch.randperm(len(_full_train))[:25000]
    cifar_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(_full_train, _indices),
        batch_size=128, shuffle=True,
    )

    mo.output.replace(mo.md(f"""
    ### Datasets Loaded

    | Dataset | Train Size | Test Size | Note |
    |---------|-----------|----------|------|
    | CIFAR-10 | {len(cifar_train_loader.dataset):,} | {len(cifar_test_loader.dataset):,} | 25K subset (time constraints) |
    | DermaMNIST | {len(derma_train_loader.dataset):,} | {len(derma_test_loader.dataset):,} | Full dataset |
    """))
    return (
        cifar_test_loader,
        cifar_train_loader,
        derma_test_loader,
        derma_train_loader,
    )


@app.cell
def scenario_a_intro():
    mo.output.replace(mo.md("""
    ---
    # Scenario A: Capacity Mismatch

    **Research question**: Does knowledge distillation always help, regardless of the student's capacity?

    We will use a **ResNet-34** teacher to distill into two students at opposite ends of the capacity spectrum:
    1. **ResNet-18** ($\\sim$11M params) — large student, similar architecture to the teacher
    2. **Linear classifier** ($\\sim$30K params) — minimal student, no hidden layers

    Each student is trained both **standalone** (with cross-entropy) and **distilled** (with the KD loss).

    **Hypothesis**: Distillation should help when the capacity gap is moderate. A very weak student
    may not have enough capacity to absorb the teacher's knowledge.

    > **Note**: You need a trained teacher checkpoint at `checkpoints/resnet34_cifar10.safetensors`.
    > If you don't have one, ask your instructor.
    """))
    return


@app.cell
def scenario_a_button():
    run_a_btn = mo.ui.run_button(label="Run Scenario A (~5-10 min on CPU)")
    mo.output.replace(run_a_btn)
    return (run_a_btn,)


@app.cell
def scenario_a_run(
    cifar_test_loader,
    cifar_train_loader,
    device,
    os,
    run_a_btn,
    run_scenario_a,
):
    mo.stop(not run_a_btn.value)

    _teacher_path = "checkpoints/resnet34_cifar10.safetensors"
    if not os.path.exists(_teacher_path):
        mo.output.replace(mo.callout(
            mo.md(f"**Teacher checkpoint not found** at `{_teacher_path}`.\n\n"
                   "Ask your instructor for the checkpoint file."),
            kind="danger",
        ))
        mo.stop(True)

    with mo.status.progress_bar(
        title="Scenario A: Capacity Mismatch",
        subtitle="Training students...",
        total=4 * 10 + 2,  # 2 students * 2 modes * 10 epochs + 2 init callbacks
    ) as _pbar:
        def _progress_cb(_label, _epoch, _tl, _ta, _vl, _va):
            _pbar.update(increment=1, subtitle=f"{_label} \u2014 epoch {_epoch + 1}")

        results_a = run_scenario_a(
            _teacher_path, cifar_train_loader, cifar_test_loader, device,
            epochs=10, lr=0.1, progress_callback=_progress_cb,
        )

    mo.output.replace(mo.md("### Scenario A Complete!"))
    return (results_a,)


@app.cell
def scenario_a_results(
    format_capacity_table,
    os,
    plot_capacity_results,
    results_a,
):
    os.makedirs("plots", exist_ok=True)
    _table = format_capacity_table(results_a)
    _fig = plot_capacity_results(results_a)
    _fig.savefig("plots/scenario_a_capacity.pdf", bbox_inches="tight")
    mo.output.replace(mo.vstack([mo.md("### Results"), mo.md(_table), _fig]))
    return


@app.cell
def scenario_a_curves(os, plot_training_curves, results_a):
    os.makedirs("plots", exist_ok=True)
    _student_names = [k for k in results_a if k not in ("teacher_acc",)]
    _histories = {}
    for _name in _student_names:
        _histories[f"{_name} (standalone)"] = results_a[_name]["standalone"]["history"]
        _histories[f"{_name} (distilled)"] = results_a[_name]["distilled"]["history"]

    _fig = plot_training_curves(_histories, title="Scenario A")
    _fig.savefig("plots/scenario_a_curves.pdf", bbox_inches="tight")
    mo.output.replace(mo.vstack([mo.md("### Training Curves"), _fig]))
    return


@app.cell
def scenario_a_analysis():
    qa1 = mo.ui.text_area(
        label="Q3: Which student benefited most from distillation? Which benefited least? Why?",
        full_width=True,
    )
    qa2 = mo.ui.text_area(
        label="Q4: What does this tell us about the relationship between student capacity and distillation effectiveness?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([mo.md("### Analysis Questions"), qa1, qa2]))
    return qa1, qa2


@app.cell
def scenario_b_intro():
    mo.output.replace(mo.md("""
    ---
    # Scenario B: Imbalanced Data Bias

    **Research question**: Can distillation amplify class imbalance biases from the teacher?

    We use the **DermaMNIST** dataset \u2014 a medical imaging dataset with 7 skin lesion classes
    that has significant class imbalance. A **ResNet-18** teacher distills into a **ResNet-18** student (trained from scratch).

    **Hypothesis**: The teacher may perform poorly on rare classes. Distillation could propagate
    (or even amplify) this bias to the student, because the student learns to mimic the teacher's
    distribution \u2014 including its mistakes.

    > **Note**: You need a trained teacher checkpoint at `checkpoints/resnet18_dermamnist.safetensors`.
    """))
    return


@app.cell
def scenario_b_distribution(
    DERMAMNIST_CLASSES,
    derma_train_loader,
    get_class_distribution_fast,
    plot_class_distribution,
):
    _dist = get_class_distribution_fast(derma_train_loader.dataset)
    _fig = plot_class_distribution(_dist, DERMAMNIST_CLASSES)
    mo.output.replace(mo.vstack([mo.md("### DermaMNIST Class Distribution"), _fig]))
    return


@app.cell
def scenario_b_button():
    run_b_btn = mo.ui.run_button(label="Run Scenario B (~5 min on CPU)")
    mo.output.replace(run_b_btn)
    return (run_b_btn,)


@app.cell
def scenario_b_run(
    derma_test_loader,
    derma_train_loader,
    device,
    os,
    run_b_btn,
    run_scenario_b,
):
    mo.stop(not run_b_btn.value)

    _teacher_path = "checkpoints/resnet18_dermamnist.safetensors"
    if not os.path.exists(_teacher_path):
        mo.output.replace(mo.callout(
            mo.md(f"**Teacher checkpoint not found** at `{_teacher_path}`.\n\n"
                   "Ask your instructor for the checkpoint file."),
            kind="danger",
        ))
        mo.stop(True)

    with mo.status.progress_bar(
        title="Scenario B: Imbalanced Data",
        subtitle="Training students...",
        total=2 * 10,
    ) as _pbar:
        def _progress_cb(_label, _epoch, _tl, _ta, _vl, _va):
            _pbar.update(increment=1, subtitle=f"{_label} \u2014 epoch {_epoch + 1}")

        results_b = run_scenario_b(
            _teacher_path, derma_train_loader, derma_test_loader, device,
            epochs=10, lr=0.01, progress_callback=_progress_cb,
        )

    mo.output.replace(mo.md("### Scenario B Complete!"))
    return (results_b,)


@app.cell
def scenario_b_metrics(
    DERMAMNIST_CLASSES,
    compute_metrics,
    os,
    plot_confusion_matrices,
    plot_per_class_f1,
    results_b,
):
    os.makedirs("plots", exist_ok=True)
    _teacher_m = compute_metrics(results_b["teacher"]["preds"], results_b["teacher"]["labels"], DERMAMNIST_CLASSES)
    _standalone_m = compute_metrics(results_b["standalone"]["preds"], results_b["standalone"]["labels"], DERMAMNIST_CLASSES)
    _distilled_m = compute_metrics(results_b["distilled"]["preds"], results_b["distilled"]["labels"], DERMAMNIST_CLASSES)

    _fig_cm = plot_confusion_matrices(_teacher_m, _standalone_m, _distilled_m,
                                      class_names=DERMAMNIST_CLASSES, title_prefix="DermaMNIST: ")
    _fig_cm.savefig("plots/scenario_b_confusion.pdf", bbox_inches="tight")

    _fig_f1 = plot_per_class_f1(
        {"Teacher": _teacher_m, "Standalone": _standalone_m, "Distilled": _distilled_m},
        DERMAMNIST_CLASSES,
    )
    _fig_f1.savefig("plots/scenario_b_f1.pdf", bbox_inches="tight")

    mo.output.replace(mo.vstack([
        mo.md("### Confusion Matrices (row-normalized)"),
        _fig_cm,
        mo.md("### Per-Class F1 Scores"),
        _fig_f1,
    ]))
    return


@app.cell
def scenario_b_analysis():
    qb1 = mo.ui.text_area(
        label="Q5: Compare per-class F1 scores. Which classes suffer most? Are these the minority classes?",
        full_width=True,
    )
    qb2 = mo.ui.text_area(
        label="Q6: Did distillation help or hurt on minority classes compared to standalone training? What mechanism explains this?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([mo.md("### Analysis Questions"), qb1, qb2]))
    return qb1, qb2


@app.cell(hide_code=True)
def scenario_c_intro():
    mo.output.replace(mo.md("""
    ---
    # Scenario C: Temperature Sensitivity

    **Research question**: How sensitive is distillation to the temperature hyperparameter?

    We use the same **ResNet-34** teacher and **ResNet-18** student from Scenario A, but sweep
    across temperatures $T \in \{1, 4, 8, 20\}$.

    **Hypothesis**: There should be an optimal temperature range. Too low ($T=1$) gives hard distributions
    with little extra information. Too high makes everything uniform and loses discriminative signal.

    > **Note**: Uses the same teacher checkpoint as Scenario A.
    """))
    return


@app.cell
def scenario_c_button():
    run_c_btn = mo.ui.run_button(label="Run Scenario C (~10-15 min on CPU)")
    mo.output.replace(run_c_btn)
    return (run_c_btn,)


@app.cell
def scenario_c_run(
    cifar_test_loader,
    cifar_train_loader,
    device,
    os,
    run_c_btn,
    run_scenario_c,
):
    mo.stop(not run_c_btn.value)

    _teacher_path = "checkpoints/resnet34_cifar10.safetensors"
    if not os.path.exists(_teacher_path):
        mo.output.replace(mo.callout(
            mo.md(f"**Teacher checkpoint not found** at `{_teacher_path}`."),
            kind="danger",
        ))
        mo.stop(True)

    _temps = (1, 4, 8, 20)
    with mo.status.progress_bar(
        title="Scenario C: Temperature Sweep",
        subtitle="Training...",
        total=len(_temps) * 10,
    ) as _pbar:
        def _progress_cb(_label, _epoch, _tl, _ta, _vl, _va):
            _pbar.update(increment=1, subtitle=f"{_label} \u2014 epoch {_epoch + 1}")

        results_c = run_scenario_c(
            _teacher_path, cifar_train_loader, cifar_test_loader, device,
            epochs=10, lr=0.1, temperatures=_temps, progress_callback=_progress_cb,
        )

    mo.output.replace(mo.md("### Scenario C Complete!"))
    return (results_c,)


@app.cell
def scenario_c_results(
    os,
    plot_temperature_curve,
    results_c,
    visualize_temperature_effect,
):
    os.makedirs("plots", exist_ok=True)
    _fig_temp = plot_temperature_curve(results_c)
    _fig_temp.savefig("plots/scenario_c_temperature.pdf", bbox_inches="tight")
    _fig_softmax = visualize_temperature_effect(
        results_c["teacher_logits"],
        temperatures=(1, 2, 4, 8, 16, 20),
    )
    _fig_softmax.savefig("plots/scenario_c_softmax.pdf", bbox_inches="tight")

    mo.output.replace(mo.vstack([
        mo.md("### Accuracy vs Temperature"),
        _fig_temp,
        mo.md("### Teacher Softmax at Different Temperatures"),
        _fig_softmax,
    ]))
    return


@app.cell
def scenario_c_analysis():
    qc1 = mo.ui.text_area(
        label="Q7: What is the optimal temperature range? Why does performance degrade at the extremes?",
        full_width=True,
    )
    qc2 = mo.ui.text_area(
        label="Q8: Look at the softmax visualizations. At $T=20$, what does the distribution look like? Why is this problematic for learning?",
        full_width=True,
    )
    mo.output.replace(mo.vstack([mo.md("### Analysis Questions"), qc1, qc2]))
    return qc1, qc2


@app.cell
def synthesis_intro():
    mo.output.replace(mo.md("""
    ---
    # Synthesis: When Does Knowledge Distillation Fail?

    Reflect on all three scenarios and answer the following questions.
    """))
    return


@app.cell
def synthesis_questions():
    s1 = mo.ui.text_area(
        label="S1: Under what conditions did distillation clearly help? Summarize the common factors.",
        full_width=True,
    )
    s2 = mo.ui.text_area(
        label="S2: Under what conditions did distillation fail or hurt? What went wrong?",
        full_width=True,
    )
    s3 = mo.ui.text_area(
        label="S3: If you were deploying a model in production with limited compute, what practical guidelines would you follow for knowledge distillation?",
        full_width=True,
    )
    s4 = mo.ui.text_area(
        label="S4: Propose one modification to the standard KD approach that might address one of the failure cases you observed.",
        full_width=True,
    )
    mo.output.replace(mo.vstack([s1, s2, s3, s4]))
    return s1, s2, s3, s4


@app.cell(hide_code=True)
def submission_form():
    student_name = mo.ui.text(label="Your full name")
    include_ckpts = mo.ui.checkbox(label="Include checkpoints (~180 MB)")
    submit_btn = mo.ui.run_button(label="Create submission archive")

    mo.output.replace(mo.vstack([
        mo.md("""
    ---
    # Submission

    Fill in your name and click the button to generate a `.zip` archive.
    The archive will contain your notebook, your `distillation.py` implementation,
    all your answers (as JSON and as a Markdown report), and your plots.
        """),
        student_name,
        include_ckpts,
        submit_btn,
    ]))
    return include_ckpts, student_name, submit_btn


@app.cell
def submission_handler(
    create_archive,
    include_ckpts,
    q1,
    q2,
    qa1,
    qa2,
    qb1,
    qb2,
    qc1,
    qc2,
    s1,
    s2,
    s3,
    s4,
    student_name,
    submit_btn,
):
    mo.stop(not submit_btn.value)

    _name = (student_name.value or "").strip()
    if not _name:
        mo.output.replace(mo.callout(
            mo.md("**Please enter your name** before creating the archive."),
            kind="warn",
        ))
        mo.stop(True)

    _answers = {
        "student_name": _name,
        "Q1_temperature_softmax": q1.value,
        "Q2_T_squared": q2.value,
        "Q3_capacity_benefit": qa1.value,
        "Q4_capacity_relationship": qa2.value,
        "Q5_per_class_f1": qb1.value,
        "Q6_distillation_minority": qb2.value,
        "Q7_optimal_temperature": qc1.value,
        "Q8_high_temperature": qc2.value,
        "S1_when_helps": s1.value,
        "S2_when_fails": s2.value,
        "S3_practical_guidelines": s3.value,
        "S4_proposed_modification": s4.value,
    }

    _zip_path = create_archive(_name, _answers, include_checkpoints=include_ckpts.value)

    mo.output.replace(mo.callout(
        mo.md(f"**Submission archive created!**\n\n"
              f"📦 `{_zip_path}`\n\n"
              f"Upload this file to the course platform."),
        kind="success",
    ))
    return


if __name__ == "__main__":
    app.run()
