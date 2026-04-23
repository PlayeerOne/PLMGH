from torchmetrics import MetricCollection
import torchmetrics.classification as C

def _make_cls_metrics(num_classes: int, average: str | None = None, *, 
use_probs: bool = False, threshold : float | None = None):
    
    if num_classes == 2:
        if use_probs:
            # expects probabilities in [0,1]; will apply threshold (default 0.5)
            return MetricCollection({
                "f1":         C.BinaryF1Score(threshold=threshold),
                "precision":  C.BinaryPrecision(threshold=threshold),
                "recall":     C.BinaryRecall(threshold=threshold),
                "accuracy":   C.BinaryAccuracy(threshold=threshold),
                "auprc":      C.BinaryAveragePrecision(),   # <-- AUPRC for Devign
            })
        else:
            # expects 0/1 integer predictions (already thresholded)
            return MetricCollection({
                "f1":         C.BinaryF1Score(),
                "precision":  C.BinaryPrecision(),
                "recall":     C.BinaryRecall(),
                "accuracy":   C.BinaryAccuracy(),
                "auprc":      C.BinaryAveragePrecision(),   # optional but safe (if you log probs separately)
            })
    else:
        if average not in {"micro", "macro", "weighted", "none", None}:
            raise ValueError(f"Invalid average={average}")
        return MetricCollection({
            "f1":         C.MulticlassF1Score(num_classes=num_classes, average=average),
            "precision":  C.MulticlassPrecision(num_classes=num_classes, average=average),
            "recall":     C.MulticlassRecall(num_classes=num_classes, average=average),
            "accuracy":   C.MulticlassAccuracy(num_classes=num_classes, average=average),
        })


