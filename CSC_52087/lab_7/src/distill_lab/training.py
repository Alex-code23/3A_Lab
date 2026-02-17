import torch
import torch.nn as nn
import torch.optim as optim

from src.distill_lab.distillation import distillation_loss
from src.distill_lab.models import (
    resnet18, resnet34, LinearClassifier, save_model,
)
from src.distill_lab.utils import count_parameters, format_params


def train_one_epoch_standard(model, loader, optimizer, device):
    """Train one epoch with standard cross-entropy loss.

    Returns:
        (avg_loss, accuracy) as floats
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def train_one_epoch_distilled(student, teacher, loader, optimizer, device, T=4.0, alpha=0.7):
    """Train one epoch with knowledge distillation loss.

    Returns:
        (avg_loss, accuracy) as floats
    """
    student.train()
    teacher.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        student_logits = student(inputs)
        with torch.no_grad():
            teacher_logits = teacher(inputs)

        loss = distillation_loss(student_logits, teacher_logits, targets, T, alpha)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += student_logits.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate a model on a data loader.

    Returns:
        (avg_loss, accuracy, all_preds, all_labels) — preds and labels are lists
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(targets.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def train_standard(model, train_loader, test_loader, device, epochs=20, lr=0.1, progress_callback=None):
    """Full standard training loop with SGD + cosine annealing.

    Args:
        progress_callback: Optional callable(epoch, train_loss, train_acc, test_loss, test_acc)

    Returns:
        dict with keys: train_loss, train_acc, test_loss, test_acc (lists per epoch)
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_standard(model, train_loader, optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if progress_callback:
            progress_callback(epoch, train_loss, train_acc, test_loss, test_acc)

    return history


def train_distilled(student, teacher, train_loader, test_loader, device,
                    T=4.0, alpha=0.7, epochs=20, lr=0.1, progress_callback=None):
    """Full distillation training loop with SGD + cosine annealing.

    Args:
        progress_callback: Optional callable(epoch, train_loss, train_acc, test_loss, test_acc)

    Returns:
        dict with keys: train_loss, train_acc, test_loss, test_acc (lists per epoch)
    """
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    student.to(device)
    teacher.to(device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_distilled(
            student, teacher, train_loader, optimizer, device, T, alpha
        )
        test_loss, test_acc, _, _ = evaluate(student, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if progress_callback:
            progress_callback(epoch, train_loss, train_acc, test_loss, test_acc)

    return history


def train_teacher(model, train_loader, test_loader, device, save_path, epochs=50, lr=0.1):
    """Train and save a teacher model checkpoint.

    Returns:
        history dict
    """
    history = train_standard(model, train_loader, test_loader, device, epochs=epochs, lr=lr)
    save_model(model, save_path)
    final_acc = history["test_acc"][-1]
    print(f"Teacher saved to {save_path} — test accuracy: {final_acc:.2%}")
    return history


# --- Scenario orchestrators ---

def run_scenario_a(teacher_path, train_loader, test_loader, device, epochs=20, lr=0.1,
                   progress_callback=None):
    """Scenario A: Capacity mismatch — ResNet-34 teacher → 2 students.

    Students: ResNet-18, LinearClassifier
    Each trained standalone and with distillation.

    Returns:
        dict mapping student_name → {
            'standalone': {'history': ..., 'test_acc': ..., 'preds': ..., 'labels': ...},
            'distilled': {'history': ..., 'test_acc': ..., 'preds': ..., 'labels': ...},
            'params': str,
        }
    """
    teacher = resnet34(num_classes=10, pretrained_path=teacher_path).to(device)
    teacher.eval()
    teacher_loss, teacher_acc, _, _ = evaluate(teacher, test_loader, device)

    students_config = [
        ("ResNet-18", lambda: resnet18(num_classes=10)),
        ("Linear", lambda: LinearClassifier(input_size=3 * 32 * 32, num_classes=10)),
    ]

    results = {
        "teacher_acc": teacher_acc,
    }

    for name, make_student in students_config:
        # Standalone training
        student_standalone = make_student().to(device)
        params = format_params(count_parameters(student_standalone))

        if progress_callback:
            progress_callback(f"Training {name} (standalone)", 0, 0, 0, 0, 0)

        def _make_cb(label):
            def cb(epoch, tl, ta, vl, va):
                if progress_callback:
                    progress_callback(label, epoch, tl, ta, vl, va)
            return cb

        hist_s = train_standard(
            student_standalone, train_loader, test_loader, device,
            epochs=epochs, lr=lr, progress_callback=_make_cb(f"{name} standalone"),
        )
        _, acc_s, preds_s, labels_s = evaluate(student_standalone, test_loader, device)

        # Distilled training
        student_distilled = make_student().to(device)
        hist_d = train_distilled(
            student_distilled, teacher, train_loader, test_loader, device,
            epochs=epochs, lr=lr, progress_callback=_make_cb(f"{name} distilled"),
        )
        _, acc_d, preds_d, labels_d = evaluate(student_distilled, test_loader, device)

        results[name] = {
            "standalone": {"history": hist_s, "test_acc": acc_s, "preds": preds_s, "labels": labels_s},
            "distilled": {"history": hist_d, "test_acc": acc_d, "preds": preds_d, "labels": labels_d},
            "params": params,
        }

    return results


def run_scenario_b(teacher_path, train_loader, test_loader, device, epochs=20, lr=0.01,
                   progress_callback=None):
    """Scenario B: Imbalanced data bias — ResNet-18 teacher → ResNet-18 student on DermaMNIST.

    Returns:
        dict with teacher and student standalone/distilled results + metrics
    """
    teacher = resnet18(num_classes=7, pretrained_path=teacher_path).to(device)
    teacher.eval()
    _, teacher_acc, teacher_preds, teacher_labels = evaluate(teacher, test_loader, device)

    def _make_cb(label):
        def cb(epoch, tl, ta, vl, va):
            if progress_callback:
                progress_callback(label, epoch, tl, ta, vl, va)
        return cb

    # Standalone ResNet-18
    student_standalone = resnet18(num_classes=7).to(device)
    hist_s = train_standard(
        student_standalone, train_loader, test_loader, device,
        epochs=epochs, lr=lr, progress_callback=_make_cb("ResNet-18 standalone"),
    )
    _, acc_s, preds_s, labels_s = evaluate(student_standalone, test_loader, device)

    # Distilled ResNet-18
    student_distilled = resnet18(num_classes=7).to(device)
    hist_d = train_distilled(
        student_distilled, teacher, train_loader, test_loader, device,
        epochs=epochs, lr=lr, progress_callback=_make_cb("ResNet-18 distilled"),
    )
    _, acc_d, preds_d, labels_d = evaluate(student_distilled, test_loader, device)

    return {
        "teacher": {"test_acc": teacher_acc, "preds": teacher_preds, "labels": teacher_labels},
        "standalone": {"history": hist_s, "test_acc": acc_s, "preds": preds_s, "labels": labels_s},
        "distilled": {"history": hist_d, "test_acc": acc_d, "preds": preds_d, "labels": labels_d},
    }


def run_scenario_c(teacher_path, train_loader, test_loader, device, epochs=20, lr=0.1,
                   temperatures=(1, 2, 4, 8, 16, 20), progress_callback=None):
    """Scenario C: Temperature sensitivity — ResNet-34 → ResNet-18 at various T.

    Returns:
        dict mapping T → {'history': ..., 'test_acc': ..., 'preds': ..., 'labels': ...}
        Plus 'teacher_acc' and 'teacher_logits' keys.
    """
    teacher = resnet34(num_classes=10, pretrained_path=teacher_path).to(device)
    teacher.eval()
    _, teacher_acc, _, _ = evaluate(teacher, test_loader, device)

    # Collect teacher logits for visualization
    teacher_logits_sample = None
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            teacher_logits_sample = teacher(inputs)[:8]  # first 8 samples
            break

    def _make_cb(label):
        def cb(epoch, tl, ta, vl, va):
            if progress_callback:
                progress_callback(label, epoch, tl, ta, vl, va)
        return cb

    results = {
        "teacher_acc": teacher_acc,
        "teacher_logits": teacher_logits_sample.cpu(),
    }

    for T in temperatures:
        student = resnet18(num_classes=10).to(device)
        hist = train_distilled(
            student, teacher, train_loader, test_loader, device,
            T=T, alpha=0.7, epochs=epochs, lr=lr,
            progress_callback=_make_cb(f"T={T}"),
        )
        _, acc, preds, labels = evaluate(student, test_loader, device)
        results[T] = {"history": hist, "test_acc": acc, "preds": preds, "labels": labels}

    return results
