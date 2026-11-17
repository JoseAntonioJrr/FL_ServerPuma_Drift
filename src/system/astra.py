"""ASTRA curriculum and local objective (ASTRA-FL, Eq. 2 and Eq. 3).

The server computes the curriculum; clients compute the objective on private data.
Rounds here are zero-based, as in Algorithm 1 of the paper.
"""


def astra_round_config(round_index, total_rounds, boot_rounds=10, interval=2,
                       mu=0.01, alpha_init=0.2, temperature=3.0, local_epochs=5):
    """Return the per-round instructions to accompany the global model."""
    if total_rounds < 1 or not 0 <= round_index < total_rounds:
        raise ValueError("ASTRA requires 0 <= round_index < total_rounds")
    if boot_rounds < 0 or interval < 1 or local_epochs < 1:
        raise ValueError("ASTRA boot_rounds must be >= 0 and interval/epochs >= 1")
    if mu < 0 or alpha_init < 0 or temperature <= 0:
        raise ValueError("ASTRA requires mu/alpha >= 0 and temperature > 0")

    active = round_index <= boot_rounds or round_index % interval == 0
    alpha = alpha_init * (1 - round_index / total_rounds) if active else 0.0
    return {"enabled": True, "mu": mu, "alpha": alpha,
            "temperature": temperature, "local_epochs": local_epochs,
            "round_index": round_index}


def astra_loss(student_logits, labels, student, reference_params, mu,
               alpha=0.0, temperature=3.0, teacher_logits=None):
    """Cross entropy + mu/2 * squared L2 distance + alpha*T²*KL(teacher||student)."""
    import torch
    import torch.nn.functional as F

    loss = F.cross_entropy(student_logits, labels)
    if mu:
        proximal = sum(
            (param - reference_params[name]).square().sum()
            for name, param in student.named_parameters() if param.requires_grad
        )
        loss = loss + mu / 2 * proximal
    if alpha:
        if teacher_logits is None:
            raise ValueError("ASTRA teacher logits required in an active round")
        kd = F.kl_div(
            F.log_softmax(student_logits.float() / temperature, dim=1),
            F.softmax(teacher_logits.float() / temperature, dim=1),
            reduction="batchmean",
        )
        loss = loss + alpha * temperature ** 2 * kd
    return loss
