import torch
import torch.nn.functional as F

def cross_entropy(
    pred,                      # [B,1] or [B,C]
    target,                    # [B], ints; {0,1} if binary
    *,
    from_logits: bool = True,
    class_weights: torch.Tensor | None = None,   # CE only, shape [C]
    pos_weight: torch.Tensor | None = None,      # BCE only, shape [] or [1]
    label_smoothing: float = 0.0,                # CE only
    bce_label_smoothing: float = 0.0,            # BCE only, smooth toward 0.5
    reduction: str = "mean",
):
    """
    Unified classification loss:
      - Binary (pred.shape[-1] == 1): BCEWithLogits (or BCE if from_logits=False)
      - Multiclass/2-logit binary (pred.shape[-1] > 1): CrossEntropy (or NLL if probs)
    Notes:
      * DO NOT apply sigmoid/softmax before calling with from_logits=True.
      * pos_weight should be computed on TRAIN: N_neg / N_pos.
    """
    C = pred.shape[-1]

    if class_weights is not None:
        class_weights = class_weights.to(pred.device, dtype=pred.dtype)
    if pos_weight is not None:
        pos_weight = pos_weight.to(pred.device, dtype=pred.dtype)

    if C == 1:
        # ----- Binary path (preferred: logits) -----
        tgt = target.float().view(-1)
        if bce_label_smoothing > 0.0:
            eps = bce_label_smoothing
            tgt = tgt * (1.0 - eps) + 0.5 * eps

        x = pred.view(-1)
        if from_logits:
            return F.binary_cross_entropy_with_logits(
                x, tgt, pos_weight=pos_weight, reduction=reduction
            )
        else:
            probs = x.clamp(1e-8, 1 - 1e-8)
            return F.binary_cross_entropy(probs, tgt, reduction=reduction)

    # ----- CE path (covers multiclass and 2-logit binary) -----
    if from_logits:
        return F.cross_entropy(
            pred, target.long(),
            weight=class_weights,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    else:
        log_probs = pred.clamp_min(1e-8).log()
        return F.nll_loss(
            log_probs, target.long(),
            weight=class_weights,
            reduction=reduction,
        )