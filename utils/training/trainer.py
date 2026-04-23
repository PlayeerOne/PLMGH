# ============================================================
# training_utils.py
# ------------------------------------------------------------
# Utilities to assemble a Lightning Trainer with optional W&B
# logging, checkpointing, early stopping, and a Rich progress bar.
# Split into:
#   - create_callbacks(...)
#   - build_trainer(...)
#   - make_trainer(...)  <-- orchestrates both
# ============================================================

import os
from typing import Iterable, Optional, Tuple, List, Dict

from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    RichProgressBar,
)
from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBarTheme,
)
from lightning.pytorch.loggers import WandbLogger


# ---------------------------------------------------------------------
# 1) Callback factory
# ---------------------------------------------------------------------
def create_callbacks(
    *,
    # ---- Checkpointing ----
    ckpt_monitor: str = "val_f1",
    ckpt_mode: str = "max",
    ckpt_filename: str = "{epoch}-{val_f1:.4f}",
    ckpt_save_top_k: int = 1,
    ckpt_dir: Optional[str] = None,
    save_last: bool = True,                  
    save_weights_only: bool = False,      

    # ---- Early stopping ----
    early_monitor: str = "val_loss",
    early_mode: str = "min",
    early_patience: int = 5,

    # ---- Progress bar ----
    enable_progress_bar: bool = True,
    progress_bar_color: str = "green",
):
    checkpoint = ModelCheckpoint(
        monitor=ckpt_monitor,
        mode=ckpt_mode,
        save_top_k=ckpt_save_top_k,
        filename=ckpt_filename,
        dirpath=ckpt_dir,
        save_last=save_last,
        save_weights_only=save_weights_only,
    )

    early_stop = EarlyStopping(
        monitor=early_monitor,
        mode=early_mode,
        patience=early_patience,
    )

    callbacks: List = [checkpoint, early_stop]
    rich_bar = None
    if enable_progress_bar:
        theme = RichProgressBarTheme(
            description="white",
            progress_bar=progress_bar_color,
            progress_bar_finished=progress_bar_color,
            progress_bar_pulse=progress_bar_color,
            batch_progress="white",
            time="white",
            processing_speed="white",
            metrics="white",
        )
        rich_bar = RichProgressBar(refresh_rate=1, theme=theme)
        callbacks.append(rich_bar)

    return {"checkpoint": checkpoint, "early_stop": early_stop, "rich_bar": rich_bar}

# ---------------------------------------------------------------------
# 2) Trainer builder (pure: no side effects besides constructing Trainer)
# ---------------------------------------------------------------------
def build_trainer(
    *,
    # ---- Core ----
    accelerator: str = "gpu",
    devices: int | str = 1,
    max_epochs: int = 20,
    precision: str = "16-mixed",
    log_every_n_steps: int = 10,
    enable_progress_bar: bool = True,

    # ---- Grad clipping ----
    gradient_clip_val: float = 1.0,
    gradient_clip_algorithm: str = "norm",

    # ---- I/O ----
    logger: Optional[WandbLogger] = None,
    callbacks: Optional[List] = None,
) -> Trainer:
    """
    Build and return a configured Lightning Trainer from the provided parameters.
    """
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        logger=logger,
        callbacks=callbacks or [],
    )
    return trainer


# ---------------------------------------------------------------------
# 3) Orchestrator: optional W&B + callbacks + trainer
# ---------------------------------------------------------------------
def make_trainer(
    *,
    # ---- W&B Logging (optional) ----
    wandb_api_key: Optional[str] = None,        # if None, assumes env is already set
    log_wandb: bool = True,
    project: str = "devign_debugging",
    entity: Optional[str] = "taoufik-k-idrissi-polytechnique-montr-al",
    run_name: str = "jav250_GCN_run1",
    tags: Optional[Iterable[str]] = ("jav250", "GCN", "baseline"),
    log_model: bool = False,

    # ---- Trainer Core ----
    accelerator: str = "gpu",
    devices: int | str = 1,
    max_epochs: int = 20,
    precision: str = "16-mixed",
    log_every_n_steps: int = 10,
    enable_progress_bar: bool = True,

    # ---- Gradient Clipping ----
    gradient_clip_val: float = 1.0,
    gradient_clip_algorithm: str = "norm",

    # ---- Callbacks ----
    ckpt_monitor: str = "val_f1",
    ckpt_mode: str = "max",
    ckpt_filename: str = "{epoch}-{val_f1:.4f}",
    ckpt_save_top_k: int = 1,
    early_monitor: str = "val_loss",
    early_mode: str = "min",
    early_patience: int = 5,
    progress_bar_color: str = "green",
) -> Tuple[Trainer, Optional[WandbLogger], Dict[str, object]]:
    """
    High-level helper that (optionally) creates a W&B logger, builds callbacks, and returns a ready-to-use Trainer.

    Returns:
        trainer (Trainer): Configured Lightning Trainer.
    """
    # (a) Optional W&B logger
    wb = None
    if log_wandb:
        if wandb_api_key:
          os.environ["WANDB_API_KEY"] = wandb_api_key
          wb = WandbLogger(
              project=project,
              entity=entity,
              name=run_name,
              tags=list(tags) if tags else None,
              log_model=log_model,
          )

    # (b) Callbacks
    callbacks_dict = create_callbacks(
        ckpt_monitor=ckpt_monitor,
        ckpt_mode=ckpt_mode,
        ckpt_filename=ckpt_filename,
        ckpt_save_top_k=ckpt_save_top_k,
        early_monitor=early_monitor,
        early_mode=early_mode,
        early_patience=early_patience,
        enable_progress_bar=enable_progress_bar,
        progress_bar_color=progress_bar_color,
    )
    callbacks_list = [cb for cb in callbacks_dict.values() if cb is not None]

    # (c) Trainer
    trainer = build_trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        logger=wb if wb is not None else False,    
        callbacks= callbacks_list,
    )

    return trainer