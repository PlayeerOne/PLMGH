import torch
import lightning as L
import torch.nn as nn
from utils.metrics.losses import cross_entropy
from utils.metrics.metrics import _make_cls_metrics
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchmetrics.classification as C

class MLP_Classifier(L.LightningModule):
    def __init__(self,
        input_dim: int = 768,
        hidden_dim: int = 1024,
        depth : int = 4,
        num_classes: int = 250,
        dropout: float = 0.25,
        weight_decay: float = 4e-5,
        use_class_weight=True,
        label_smoothing: float = 0.0,
        metrics_average: str = "macro",
        prob_threshold: float = 0.5,
        lr: float = 1e-3,
        div_factor: float = 100,
        pct_start: float = 0.15,
        anneal_strategy: str = "cos",
        final_div_factor: float = 1e4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_class_weight = use_class_weight
        self.class_weights = None
        self.pos_weight = None
        self.is_binary = (num_classes == 2)

        modules = []
        for i in range(depth): 
          if i == 0 :
            in_dim = input_dim
            out_dim = hidden_dim 
          else :
            in_dim = hidden_dim 
            out_dim = hidden_dim 
          
          block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
          )
          modules.append(block)
        #Output Layer
        modules.append(nn.Linear(hidden_dim, 1 if self.is_binary else num_classes ))
          
        self.mlp = nn.Sequential(*modules)

        self._build_and_clone_metrics()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        features = features.to(dtype)
        return self.mlp(features)

    def criterion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            pred=pred, target=target, from_logits=True,
            class_weights=self.class_weights,
            pos_weight=(self.pos_weight if self.is_binary else None),
            label_smoothing=self.hparams.label_smoothing,
        )

    def _build_and_clone_metrics(self):
        if self.is_binary:
            base = _make_cls_metrics(num_classes=2, average=None, use_probs=True, 
                                     threshold=self.hparams.prob_threshold)
        else:
            base = _make_cls_metrics(num_classes=self.hparams.num_classes,
                                     average=self.hparams.metrics_average)
        self.train_metrics = base.clone(prefix="train_")
        self.val_metrics = base.clone(prefix="val_")
        self.test_metrics = base.clone(prefix="test_")

    def _preds_from_logits(self, logits):
        if self.is_binary:
            return (logits.view(-1).sigmoid() >= self.hparams.prob_threshold).long()
        else:
            return logits.argmax(dim=-1)

    def _log_metrics(self, preds_or_probs, target, stage: str, bs: int):
        coll = {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]
        out = coll(preds_or_probs, target)
        self.log_dict(out, on_step=False, on_epoch=True, prog_bar=(stage != "train"), batch_size=bs)

    def _common_step(self, batch, batch_idx, stage: str):
        x,y = batch
        logits = self.forward(x)

        target = y.long().view(-1)
        loss = self.criterion(pred=logits, target=target)
        bs = target.shape[0]

        if self.is_binary:
            probs = logits.view(-1).sigmoid()
            preds = self._preds_from_logits(logits)
            self._log_metrics(probs, target, stage, bs)
        else:
            preds = self._preds_from_logits(logits)
            self._log_metrics(preds, target, stage, bs)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches or 1000
        sch = OneCycleLR(
            optimizer=opt,
            max_lr=self.hparams.lr,
            total_steps=int(total_steps),
            pct_start=self.hparams.pct_start,
            anneal_strategy=self.hparams.anneal_strategy,
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1}}

    def set_threshold(self, new_threshold: float):
        """
        Safely update the probability threshold used for binary classification.
        This updates:
          1) self.hparams.prob_threshold   (Lightning hyperparams)
          2) all metrics (via rebuild)
          3) prediction logic
        """
        if not self.is_binary:
            raise ValueError("Threshold is only used for binary classification.")

        # Clamp for safety
        new_threshold = float(max(0.0, min(1.0, new_threshold)))

        # Update hyperparameters
        self.hparams.prob_threshold = new_threshold

        # Rebuild metrics so they actually use the new threshold
        self._build_and_clone_metrics()


    @torch.no_grad()
    def calibrate_threshold_on_val(self, val_dataloader, steps: int = 200):
        """
        Calibrate optimal threshold on validation set to maximize F1 score.
        Assumes binary classification and access to val_dataloader.
        """
        assert self.is_binary, "Threshold calibration only makes sense for binary cls."

        self.eval()
        all_probs, all_targets = [], []

        device = self.device

        for batch in val_dataloader:
            x, y  = batch
            x = x.to(device)
            y = y.view(-1).to(device)

            logits = self.forward(x)
            probs = logits.view(-1).sigmoid()

            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

        probs = torch.cat(all_probs)    # shape [N]
        targets = torch.cat(all_targets)

        thresholds = torch.linspace(0.0, 1.0, steps)
        best_thresh, best_f1 = 0.5, 0.0

        for t in thresholds:
            preds = (probs >= t).long()
            # new metric each time → no state carry-over
            f1_metric = C.BinaryF1Score().to(probs.device)
            f1 = f1_metric(preds, targets)
            f1 = f1.item()
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        print(f"Optimal F1={best_f1:.4f} at threshold={best_thresh:.3f}")

        self.set_threshold(best_thresh)
