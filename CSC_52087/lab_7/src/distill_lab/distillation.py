import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.7):
    """Compute the knowledge distillation loss.

    The KD loss combines two terms:
    1. **Soft loss**: KL divergence between softened student and teacher distributions
    2. **Hard loss**: Standard cross-entropy between student predictions and true labels

    Args:
        student_logits: Raw logits from the student model (batch_size, num_classes)
        teacher_logits: Raw logits from the teacher model (batch_size, num_classes)
        labels: Ground truth labels (batch_size,)
        temperature: Temperature for softening distributions (higher = softer)
        alpha: Weight for the soft loss (1-alpha weights the hard loss)

    Returns:
        Combined distillation loss (scalar tensor)
    """

    # =========================================================================
    # TODO 1: Compute the soft loss
    # -------------------------------------------------------------------------
    # Steps:
    #   a) Divide student_logits and teacher_logits by the temperature
    #   b) Apply log_softmax to the scaled student logits
    #   c) Apply softmax to the scaled teacher logits
    #   d) Compute the KL divergence: F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')
    #   e) Multiply by temperature^2 (this corrects the gradient magnitude)
    #
    # Replace the line below with your implementation:
    # soft_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    # =========================================================================
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    kl_div = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')
    soft_loss = kl_div * (temperature ** 2)

    # =========================================================================
    # TODO 2: Compute the hard loss
    # -------------------------------------------------------------------------
    # This is simply the standard cross-entropy loss between student logits
    # and the true labels. Use F.cross_entropy(student_logits, labels).
    #
    # Replace the line below with your implementation:
    # hard_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    # =========================================================================
    hard_loss = F.cross_entropy(student_logits, labels)

    # =========================================================================
    # TODO 3: Combine the two losses
    # -------------------------------------------------------------------------
    # The final loss is: alpha * soft_loss + (1 - alpha) * hard_loss
    #
    # Replace the line below with your implementation:
    # loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
    # =========================================================================
    loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

    return loss
