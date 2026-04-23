import torch
import lightning as L
import torchmetrics.classification as C

from utils.metrics.losses import cross_entropy
from utils.metrics.metrics import _make_cls_metrics

# Optim / Scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden [B,T,H], attention_mask [B,T] (1 for valid, 0 for pad)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B,T,1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B,H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B,1]
    return summed / denom


class PLMClassifier(L.LightningModule):

    def __init__(
        self,
        encoder,
        freeze_encoder=True,
        hidden_size=768,
        num_classes=250,

        # --- Regularization ---
        dropout: float = 0.25,
        weight_decay: float = 4e-5,

        # --- Loss / metrics ---
        use_class_weight=True,
        label_smoothing: float = 0.0,
        metrics_average: str = "macro",
        prob_threshold: float = 0.5,

        # --- Optim and OneCycle Scheduler  ---
        lr: float = 1e-3,
        div_factor: float = 100,
        pct_start: float = 0.15,
        anneal_strategy: str = "cos",
        final_div_factor: float = 1e4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        # Non-hparam runtime fields
        self.use_class_weight = self.hparams.get("use_class_weight", True)
        self.class_weights = None
        self.pos_weight = None
        self.is_binary = (num_classes == 2)

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)

        self.hidden_size = hidden_size

        # head
        self.out_dim = 1 if self.is_binary else num_classes

        self.cls_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
            torch.nn.LayerNorm(self.hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_size * 2, self.out_dim),
        )

        self._build_and_clone_metrics()

    def extract_features(self, batch_dict):
        input_ids = batch_dict["input_ids"]
        attention_mask = batch_dict["attention_mask"]
        if getattr(self.encoder.config, "is_encoder_decoder", False):
            hs = self.encoder.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
        else:
            hs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
        return mean_pool(hs, attention_mask)

    # ------------------------- forward & loss -------------------------
    def forward(self, batch_dict) -> torch.Tensor:
        features = batch_dict.get("features", None)

        if features is None:
            features = self.extract_features(batch_dict)
            if self.freeze_encoder:
                features = features.detach()

        logits = self.cls_decoder(features)  # [B, C] or [B, 1]
        return logits

    def criterion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            pred=pred,
            target=target,
            from_logits=True,
            class_weights=self.class_weights,
            pos_weight=(self.pos_weight if self.is_binary else None),
            label_smoothing=self.hparams.label_smoothing,
        )

    # ------------------------- metrics helpers -------------------------
    def _build_and_clone_metrics(self):
        if self.is_binary:
            base = _make_cls_metrics(
                num_classes=2,
                average=None,
                use_probs=True,
                threshold=self.hparams.prob_threshold,
            )
        else:
            base = _make_cls_metrics(
                num_classes=self.hparams.num_classes,
                average=self.hparams.metrics_average,
            )
        self.train_metrics = base.clone(prefix="train_")
        self.val_metrics = base.clone(prefix="val_")
        self.test_metrics = base.clone(prefix="test_")

    # ------------------------- imbalance utilities -------------------------
    def _datamodule(self):
        return getattr(self.trainer, "datamodule", None)

    # ------------------------- lightning hooks -------------------------
    def on_fit_start(self):
        if not self.use_class_weight:
            return

        dm = self._datamodule()
        if dm is None:
            raise RuntimeError("Attach TextDataModuleCls or implement a fallback scanner.")

        self.is_binary = dm.get_is_binary()

        if self.is_binary:
            if self.pos_weight is None:
                self.pos_weight = dm.get_pos_weight(self.device)
            self.class_weights = None
        else:
            if self.class_weights is None:
                self.class_weights = dm.get_class_weights(self.device)
            self.pos_weight = None

    def on_load_checkpoint(self, checkpoint):
        self._build_and_clone_metrics()

    # ------------------------- prediction methods -------------------------
    def _preds_from_logits(self, logits):
        if self.is_binary:
            return (logits.view(-1).sigmoid() >= self.hparams.prob_threshold).long()
        return logits.argmax(dim=-1)

    # ------------------------- metrics (update per batch, compute+reset per epoch) -------------------------
    def _update_metrics(self, preds_or_probs, target, stage: str):
        coll = {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]
        coll.update(preds_or_probs, target)

    def _compute_log_reset_metrics(self, stage: str, prog_bar: bool):
        coll = {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]
        out = coll.compute()
        self.log_dict(out, on_step=False, on_epoch=True, prog_bar=prog_bar)
        coll.reset()

    def on_train_epoch_end(self):
        self._compute_log_reset_metrics(stage="train", prog_bar=False)

    def on_validation_epoch_end(self):
        self._compute_log_reset_metrics(stage="val", prog_bar=True)

    def on_test_epoch_end(self):
        self._compute_log_reset_metrics(stage="test", prog_bar=True)

    # ------------------------- train/val/test steps -------------------------
    def _common_step(self, batch_dict, batch_idx, stage: str):
        logits = self.forward(batch_dict)

        target = batch_dict["labels"].long().view(-1)
        loss = self.criterion(pred=logits, target=target)
        bs = target.shape[0]

        if self.is_binary:
            probs = logits.view(-1).sigmoid()
            self._update_metrics(probs, target, stage)
        else:
            preds = self._preds_from_logits(logits)
            self._update_metrics(preds, target, stage)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=bs,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, stage="test")

    # ------------------------- optimizer and scheduler -------------------------
    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        if not total_steps or total_steps <= 0:
            train_loader = (
                self.trainer.datamodule.train_dataloader()
                if self.trainer.datamodule
                else self.trainer.train_dataloader
            )
            steps_per_epoch = len(train_loader)
            acc = max(1, self.trainer.accumulate_grad_batches)
            total_steps = (steps_per_epoch // acc) * self.trainer.max_epochs

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

    # ========================= THRESHOLD CALIBRATION (same as GNN_Classifier) =========================
    def set_threshold(self, new_threshold: float):
        """
        Safely update the probability threshold used for binary classification.
        This updates:
          1) self.hparams.prob_threshold
          2) all metrics (via rebuild)
          3) prediction logic
        """
        if not self.is_binary:
            raise ValueError("Threshold is only used for binary classification.")

        new_threshold = float(max(0.0, min(1.0, new_threshold)))
        self.hparams.prob_threshold = new_threshold
        self._build_and_clone_metrics()

    @torch.no_grad()
    def calibrate_threshold_on_val(self, val_dataloader, steps: int = 200):
        if not self.is_binary:
            raise ValueError("Threshold calibration only makes sense for binary classification.")

        self.eval()
        device = self.device

        all_probs, all_targets = [], []

        # Use autocast so bf16 encoder outputs + fp32 head work (like during training)
        use_cuda_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16  # matches your trainer precision="bf16-mixed"

        for batch in val_dataloader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            y = batch["labels"].long().view(-1)

            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = self.forward(batch)
            else:
                logits = self.forward(batch)

            probs = logits.view(-1).sigmoid()

            all_probs.append(probs.detach().float().cpu())   # store as float32 on CPU
            all_targets.append(y.detach().cpu())

        probs = torch.cat(all_probs)      # [N] float32 CPU
        targets = torch.cat(all_targets)  # [N] int64 CPU

        thresholds = torch.linspace(0.0, 1.0, steps)
        best_thresh, best_f1 = 0.5, -1.0

        f1_metric = C.BinaryF1Score()  # CPU metric

        for t in thresholds:
            preds = (probs >= t).long()
            f1 = f1_metric(preds, targets).item()
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        print(f"Optimal F1={best_f1:.4f} at threshold={best_thresh:.3f}")
        self.set_threshold(best_thresh)
        return best_thresh, best_f1
