"""Submission utilities — build report and archive for grading."""

import json
import re
import zipfile
from pathlib import Path

# Question labels shown in the report, grouped by section.
QUESTIONS = [
    ("Part 1: Conceptual Questions", [
        ("Q1_temperature_softmax",
         "Q1: Why do we divide logits by temperature before applying softmax? "
         "What happens when $T \\to \\infty$?"),
        ("Q2_T_squared",
         "Q2: Why do we multiply the KL divergence by $T^2$?"),
    ]),
    ("Scenario A: Capacity Mismatch", [
        ("Q3_capacity_benefit",
         "Q3: Which student benefited most from distillation? Which benefited least? Why?"),
        ("Q4_capacity_relationship",
         "Q4: What does this tell us about the relationship between student capacity "
         "and distillation effectiveness?"),
    ]),
    ("Scenario B: Imbalanced Data Bias", [
        ("Q5_per_class_f1",
         "Q5: Compare per-class F1 scores. Which classes suffer most? "
         "Are these the minority classes?"),
        ("Q6_distillation_minority",
         "Q6: Did distillation help or hurt on minority classes compared to "
         "standalone training? What mechanism explains this?"),
    ]),
    ("Scenario C: Temperature Sensitivity", [
        ("Q7_optimal_temperature",
         "Q7: What is the optimal temperature range? Why does performance "
         "degrade at the extremes?"),
        ("Q8_high_temperature",
         "Q8: Look at the softmax visualizations. At $T=20$, what does the "
         "distribution look like? Why is this problematic for learning?"),
    ]),
    ("Synthesis", [
        ("S1_when_helps",
         "S1: Under what conditions did distillation clearly help?"),
        ("S2_when_fails",
         "S2: Under what conditions did distillation fail or hurt?"),
        ("S3_practical_guidelines",
         "S3: Practical guidelines for knowledge distillation in production."),
        ("S4_proposed_modification",
         "S4: Propose one modification to address a failure case you observed."),
    ]),
]

# Which plot files belong to each section.
SECTION_PLOTS = {
    "Scenario A: Capacity Mismatch": [
        "plots/scenario_a_capacity.pdf",
        "plots/scenario_a_curves.pdf",
    ],
    "Scenario B: Imbalanced Data Bias": [
        "plots/scenario_b_confusion.pdf",
        "plots/scenario_b_f1.pdf",
    ],
    "Scenario C: Temperature Sensitivity": [
        "plots/scenario_c_temperature.pdf",
        "plots/scenario_c_softmax.pdf",
    ],
}


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s-]+", "_", safe)


def generate_report(answers: dict) -> str:
    """Return a Markdown report string from the answers dict."""
    student_name = answers.get("student_name", "Unknown")
    lines = [
        "# Lab 7 — Knowledge Distillation Report",
        "",
        f"**Student:** {student_name}",
        "",
    ]

    for section_title, questions in QUESTIONS:
        lines.append("---")
        lines.append(f"## {section_title}")
        lines.append("")

        # Insert plots before the questions for this section
        for plot_path in SECTION_PLOTS.get(section_title, []):
            lines.append(f"![{Path(plot_path).stem}]({plot_path})")
            lines.append("")

        for key, label in questions:
            answer = answers.get(key, "").strip()
            lines.append(f"**{label}**")
            lines.append("")
            if answer:
                for paragraph in answer.split("\n"):
                    lines.append(f"> {paragraph}")
            else:
                lines.append("> *(no answer)*")
            lines.append("")

    return "\n".join(lines)


def create_archive(student_name: str, answers: dict, include_checkpoints: bool = False) -> Path:
    """Build a submission zip and return its path.

    The archive contains:
      - notebook.py
      - src/distill_lab/distillation.py
      - answers.json
      - report.md
      - plots/*.pdf   (if any)
      - checkpoints/*  (if requested)
    """
    safe_name = _sanitize_name(student_name)
    zip_path = Path(f"lab_7_{safe_name}.zip")

    # Write temporary files
    answers_path = Path("answers.json")
    report_path = Path("report.md")
    answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False))
    report_path.write_text(generate_report(answers))

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write("notebook.py")
            zf.write("answers.json")
            zf.write("report.md")

            distill_py = Path("src/distill_lab/distillation.py")
            if distill_py.exists():
                zf.write(str(distill_py))

            plots_dir = Path("plots")
            if plots_dir.is_dir():
                for f in sorted(plots_dir.glob("*.pdf")):
                    zf.write(str(f))

            if include_checkpoints:
                ckpt_dir = Path("checkpoints")
                if ckpt_dir.is_dir():
                    for f in sorted(ckpt_dir.glob("*.safetensors")):
                        zf.write(str(f))
    finally:
        answers_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)

    return zip_path
