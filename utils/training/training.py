import os
import torch
from model.gnn_classifier import GNN_Classifier
from utils.training.trainer import build_trainer, create_callbacks
from lightning.pytorch.loggers import WandbLogger

def build_model(hparams, model_type="GNN_Classifier", num_classes=250):
    if model_type == "GNN_Classifier":
        hparams = dict(hparams)  # avoid mutating caller
        hparams["num_classes"] = num_classes
        return GNN_Classifier(**hparams)
    raise ValueError(f"Unknown model_type: {model_type}")

def train_model(
    model_params,
    data_module,
    model_type="GNN_Classifier",
    model_name="Model_Name", #Model Name
    num_classes=250,

    # Trainer params
    accelerator="gpu",
    devices=1,
    epochs=20,
    precision="16-mixed",
    log_every_n_steps=10,
    gradient_clip_val=3.0,
    gradient_clip_algorithm="norm",

    # Monitors / callbacks
    pruner_monitor="val_f1",
    ckpt_monitor="val_f1",
    ckpt_mode="max",
    early_monitor="val_loss",
    early_mode="min",
    early_patience=5,
    ckpt_save_top_k=1,
    enable_progress_bar=True,
    progress_bar_color="green",

    # W&B
    log_wandb=True,
    wandb_api_key=None,
    project="devign_debugging",
    entity="taoufik-k-idrissi-polytechnique-montr-al",
    tags=("java250", "GCN", "baseline"),
    log_model=False,
):
    #1 Build Model
    model = build_model(hparams=model_params, model_type=model_type, num_classes=num_classes)

    #2 Decide a checkpoint directory per run
    ckpt_dir = os.path.join("checkpoints", project, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    #3 CallBacks
    callbacks_dict = create_callbacks(
        ckpt_monitor=ckpt_monitor,
        ckpt_mode=ckpt_mode,
        ckpt_filename=model_name,
        ckpt_save_top_k=ckpt_save_top_k,
        ckpt_dir=ckpt_dir,                  
        save_last=True,
        save_weights_only=False, 
        early_monitor=early_monitor,
        early_mode=early_mode,
        early_patience=early_patience,
        enable_progress_bar=enable_progress_bar,
        progress_bar_color=progress_bar_color,
    )
    ckpt_cb = callbacks_dict["checkpoint"]
    callbacks_list = [cb for cb in callbacks_dict.values() if cb is not None]

    #4 WanB logging
    wb = None
    if log_wandb:
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        wb = WandbLogger(
            project=project,
            entity=entity,
            name=model_name,
            tags=list(tags) if tags else None,
            log_model=log_model,
        )
    #5 Build trainer
    trainer = build_trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        logger=wb if wb is not None else False,
        callbacks=callbacks_list,
    )
    #6 Train
    trainer.fit(model, datamodule=data_module)
    #7 Test the model
    model.eval()
    test_metrics = trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    #8 Get checkpoint
    best_ckpt = ckpt_cb.best_model_path  # full .ckpt path
    weights_pt = None
    try:
        if best_ckpt and os.path.isfile(best_ckpt):
            # save a pure state_dict for lightweight reuse
            sd = torch.load(best_ckpt, map_location="cpu").get("state_dict", None)
            if sd is None:
                # fallback: current model (in case no state_dict key)
                sd = model.state_dict()
            weights_pt = os.path.join(ckpt_dir, f"{model_name}_best.weights.pt")
            torch.save(sd, weights_pt)
        else:
            # fallback if no best (e.g., monitor never logged)
            weights_pt = os.path.join(ckpt_dir, f"{model_name}_last.weights.pt")
            torch.save(model.state_dict(), weights_pt)
    except Exception as e:
        print(f"[warn] failed to export pure weights: {e}")
    # Kill WandB Logger
    if wb is not None:
        try:
            wb.experiment.finish()
        except Exception:
            pass
                
    return {
        "run_name": model_name,
        "test_metrics": test_metrics,
        "best_ckpt": best_ckpt,
        "weights_path": weights_pt,
        "ckpt_dir": ckpt_dir,
    }
