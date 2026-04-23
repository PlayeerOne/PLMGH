import torch
import lightning as L
import torch.nn as nn
from torch_geometric.data import Data
import torchmetrics.classification as C

from utils.metrics.losses import cross_entropy
from utils.metrics.metrics import _make_cls_metrics
from model.gnn_encoder import *
from model.fusion_layer import FeatureFusionLayer
from model.gnn_decoder import Cls_Decoder

# Optim / Scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class GNN_Classifier(L.LightningModule):   # <-- LightningModule
    """
    Lightning GNN classifier with:
    - feature fusion (node types / positional / semantic)
    - pluggable encoders (GCN/GAT/GraphTransformer)
    - unified CE/BCE loss with optional class/pos weights
    - threshold-aware binary metrics
    - OneCycleLR (step-wise) with robust total_steps fallback
    """
    def __init__(self,
        # --- GNN Encoder --- 
        gnn_type: str = "GCN",
        gnn_depth: int = 4,
        gnn_heads: int = 8,                    
        gnn_dim: int = 1024,
        gnn_norm_type: str = "graph_norm",
        gnn_activation : str = "ReLU",
        
        # --- Feature keys & toggles ---
        node_type_key: str = "node_type_idx",
        use_node_type: bool = True,
        pos_features_key: str = "positional_feats",
        use_pos_features: bool = True,
        semantic_features_key: str = "semantic_feats",
        use_semantic_features: bool = True,
        
        # --- Feature dims ---
        semantic_features_dim: int = 768,
        pos_features_dim: int = 32,
        n_node_types: int = 250,
        
        # --- Fuser ---
        node_type_dim: int = 256,
        fuser_dim: int = 1024,
        fusion_method: str = "concat",
        
        # --- Task ---
        num_classes: int = 250,
        cls_pooling_method: str = "attentional",
        
        # --- Regularization ---
        dropout: float = 0.25,
        weight_decay: float = 4e-5,
        
        # --- Loss / metrics ---
        use_class_weight = True,                      
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

        # 1) Save scalar hparams (keep tensors/flags separate if needed)
        self.save_hyperparameters()
        
        # 2) Non-hparam runtime fields
        self.use_class_weight = self.hparams.get("use_class_weight", True)
        self.class_weights = None
        self.pos_weight = None
        
        # 3) Classification type flag
        self.is_binary = (num_classes == 2)
        
        # 4) Feature fusion
        self.node_embedding = FeatureFusionLayer(
            node_type_key=self.hparams.node_type_key, use_node_type=self.hparams.use_node_type,
            pos_features_key=self.hparams.pos_features_key, use_pos_features=self.hparams.use_pos_features,
            semantic_features_key=self.hparams.semantic_features_key, use_semantic_features=self.hparams.use_semantic_features,
            semantic_features_dim=self.hparams.semantic_features_dim,
            pos_features_dim=self.hparams.pos_features_dim,
            node_type_dim=self.hparams.node_type_dim, n_node_types=self.hparams.n_node_types,
            h_dim=self.hparams.fuser_dim, out_dim=self.hparams.fuser_dim,
            fusion_method=self.hparams.fusion_method, dropout=self.hparams.dropout
        )
        
        
        # 5) Encoder
        gnn_args = dict(
          in_dim=fuser_dim,
          out_dim=gnn_dim,
          hidden_dim=gnn_dim,
          depth=gnn_depth,
          dropout=dropout,
          norm_type=gnn_norm_type,
          activation=gnn_activation
        )
        if gnn_type in {"GAT", "GraphTransformer"}:
          gnn_args["heads"] = gnn_heads
        gnn_args["use_norm"] = True
        
        gnn_types = {
          "GCN": GCN_Encoder,
          "GAT": GAT_Encoder,
          "GraphTransformer": GraphTransformer_Encoder,
        }
        self.gnn_encoder = gnn_types[gnn_type](**gnn_args)
        
        # 6) Decoder
        self.cls_decoder = Cls_Decoder(
          in_dim=gnn_dim,
          h_dim=gnn_dim * 2,
          out_dim=(1 if self.is_binary else num_classes),
          dropout=dropout,
          pooling_method=cls_pooling_method,
        )
        
        
        # 7) Metrics
        self._build_and_clone_metrics()


  # ------------------------- forward & loss -------------------------
    def forward(self, input_graph: Data) -> torch.Tensor:
        """Run fusion → GNN encoder → pooled decoder → logits."""
          
        graph = self.node_embedding(input_graph)
        graph.x = self.gnn_encoder(graph)
        logits = self.cls_decoder(graph)
        return logits

    def criterion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unified BCE/CE with optional class/pos weights and label smoothing."""
        return cross_entropy(
            pred=pred, target=target, from_logits=True,
            class_weights=self.class_weights,
            pos_weight=(self.pos_weight if self.is_binary else None),
            label_smoothing=self.hparams.label_smoothing,
        )

  # ------------------------- metrics helpers -------------------------
    def _build_and_clone_metrics(self):
        """
        Build metric collection and clone for train/val/test.
        Binary: threshold-aware (prob_threshold).
        Multiclass: average per hparam (macro/micro/weighted/none/None).
        """
        if self.is_binary:
            base = _make_cls_metrics(num_classes=2, average=None, use_probs=True, 
                                     threshold = self.hparams.prob_threshold )
        else:
            base = _make_cls_metrics(num_classes=self.hparams.num_classes,
                                     average=self.hparams.metrics_average)
        self.train_metrics = base.clone(prefix="train_")
        self.val_metrics   = base.clone(prefix="val_")
        self.test_metrics  = base.clone(prefix="test_")

  # ------------------------- imbalance utilities -------------------------
    def _train_loader_for_scan(self):
        """Return the train dataloader (works for datamodule or bare trainer)."""
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            return dm.train_dataloader()
        dl = getattr(self.trainer, "train_dataloader", None)
        return dl() if callable(dl) else dl

    def _datamodule(self):
        return getattr(self.trainer, "datamodule", None)
  
    @torch.no_grad()
    def _compute_binary_pos_weight(self):
        dm = self._datamodule()
        if dm is None:
            raise RuntimeError("No DataModule attached; cannot fetch pos_weight.")
        return dm.get_pos_weight(self.device)  # already computed in dm.setup()

    @torch.no_grad()
    def _compute_multiclass_weights(self):
        dm = self._datamodule()
        if dm is None:
            raise RuntimeError("No DataModule attached; cannot fetch class_weights.")
        return dm.get_class_weights(self.device)  # already computed in dm.setup()

  # ------------------------- lightning hooks -------------------------
    def on_fit_start(self):
        if not self.use_class_weight:
          return
    
        dm = self._datamodule()
        if dm is None:
            # (Optional) fallback to old scanning path if you want backwards-compat
            # weights = self._compute_multiclass_weights_by_scanning(...)
            # return
            raise RuntimeError("Attach GraphDataModuleCls or implement a fallback scanner.")

        # authoritative binary/multiclass from the DM
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
        """Rebuild metrics so CLI-updated hparams (e.g., prob_threshold) take effect on resume."""
        self._build_and_clone_metrics()

      # ------------------------- prediction methods -------------------------
    def _binary_probs_from_logits(self, logits):
        """Sigmoid + flatten to shape [B] for binary metrics."""
        return logits.view(-1).sigmoid()

    def _preds_from_logits(self, logits): 
        """Return integer predictions: thresholded for binary, argmax for multiclass."""
        if self.is_binary : 
            return (self._binary_probs_from_logits(logits) >= self.hparams.prob_threshold).long()
        else : 
            return logits.argmax(dim=-1)

  # ------------------------- logging -------------------------
    def _log_metrics(self, preds_or_probs, target, stage: str, bs: int):
        """Route metrics to the appropriate collection and log (epoch-level; train also logs per-step)."""
        coll = {"train": self.train_metrics, "val": self.val_metrics, "test": self.test_metrics}[stage]
        out = coll(preds_or_probs, target)
        self.log_dict(out, on_step=False, on_epoch=True, prog_bar=(stage!="train"), batch_size=bs)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)

  # ------------------------- train/val/test steps -------------------------      
    def _common_step(self, batch, batch_idx, stage: str):
        """Shared forward+loss+metrics path for train/val/test."""
        logits = self.forward(batch)
        target = batch.y.long().view(-1)
        loss = self.criterion(pred=logits, target=target)
        bs = getattr(batch, "num_graphs", target.shape[0])
        
        if self.is_binary : 
            probs = self._binary_probs_from_logits(logits)
            preds = self._preds_from_logits(logits)
            self._log_metrics(probs, target, stage, bs)
            with torch.no_grad(): 
                pos_rate_y = target.float().mean()
                pos_rate_hat = preds.float().mean()
                mean_prob = probs.mean()
            self.log(f"Binary Prediction Metrics/{stage}_y_pos_rate", pos_rate_y, on_epoch=True, prog_bar=False, batch_size=bs)
            self.log(f"Binary Prediction Metrics/{stage}_hat_pos_rate", pos_rate_hat, on_epoch=True, prog_bar=False, batch_size=bs)
            self.log(f"Binary Prediction Metrics/{stage}_mean_prob", mean_prob, on_epoch=True, prog_bar=False, batch_size=bs)
        else:
            preds = self._preds_from_logits(logits) 
            self._log_metrics(preds, target, stage, bs)
        
        self.log(f"{stage}_loss", loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True, batch_size=bs)
        
        return loss

    def training_step(self, batch, batch_idx):
        """One training iteration."""
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """One validation iteration."""
        self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """One test iteration."""
        self._common_step(batch, batch_idx, stage="test")

    # ------------------------- optimizer and scheduler  -------------------------
    def configure_optimizers(self):
        # Select Optimizer
        opt = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # Setup Scheduler
        total_steps = self.trainer.estimated_stepping_batches
        if not total_steps or total_steps <= 0: 
            train_loader = self.trainer.datamodule.train_dataloader() if self.trainer.datamodule else self.trainer.train_dataloader
            steps_per_epoch = len(train_loader)
            acc = max(1, self.trainer.accumulate_grad_batches)
            total_steps = (steps_per_epoch // acc) * self.trainer.max_epochs
    
        div_factor = self.hparams.div_factor
    
        sch = OneCycleLR(
          optimizer=opt,
          max_lr=self.hparams.lr,
          total_steps=int(total_steps),
          pct_start=self.hparams.pct_start,
          anneal_strategy=self.hparams.anneal_strategy,
          div_factor=div_factor,
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
            batch = batch.to(device)
            y = batch.y.long().view(-1)

            logits = self.forward(batch)
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

