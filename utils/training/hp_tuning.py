import os
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from model.gnn_classifier import GNN_Classifier
from utils.training.trainer import build_trainer, create_callbacks
from lightning.pytorch.loggers import WandbLogger


def sample_hparams(trial, search_space):
    hparams = {}
    for name, spec in search_space.items():
        kind = spec["kind"]
        if kind == "choice" : 
            hparams[name] = trial.suggest_categorical(name, spec["values"])
        elif kind in ("float", "log_float"):
            low = float(spec["low"])
            high = float(spec["high"])
            hparams[name] = trial.suggest_float(name, low, high, log=(kind=="log_float"))
        else : 
            raise ValueError(f"Bad range for {name}: low={low}, high={high}")
    return hparams


def build_model(hparams, model_type="GNN_Classifier", num_classes=250):
    if model_type == "GNN_Classifier":
        hparams = dict(hparams)  # avoid mutating caller
        hparams["num_classes"] = num_classes
        return GNN_Classifier(**hparams)
    raise ValueError(f"Unknown model_type: {model_type}")


def create_hparams_callbacks(
    trial,
    ckpt_monitor="val_f1",
    ckpt_mode="max",
    ckpt_filename="{epoch}-{val_f1:.4f}",
    ckpt_save_top_k=1,
    early_monitor="val_loss",
    early_mode="min",
    early_patience=5,
    enable_progress_bar=True,
    progress_bar_color="green",
    pruner_monitor="val_f1",
):
    """
    Wraps base create_callbacks and appends an Optuna pruner.
    Returns a dict of callbacks; NOTE: values can include None (rich_bar).
    """
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
    from optuna.integration import PyTorchLightningPruningCallback
    pruner = PyTorchLightningPruningCallback(trial, monitor=pruner_monitor)
    callbacks_dict["pruner"] = pruner
    return callbacks_dict


def objective(
    trial,
    *,
    search_space,
    data_module,
    model_type="GNN_Classifier",
    run_name_prefix="hptuning", #Model Name
    num_classes=250,

    # Trainer params
    accelerator="gpu",
    devices=1,
    max_epochs=20,
    precision="16-mixed",
    log_every_n_steps=10,
    gradient_clip_val=1.0,
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
    tags=("jav250", "GCN", "baseline"),
    log_model=False,
):
    # 1) sample hparams & build model
    hparams = sample_hparams(trial, search_space)
    model = build_model(hparams=hparams, model_type=model_type, num_classes=num_classes)

    # 2) callbacks (+ pruner)
    trial_run_name = f"{run_name_prefix}-t{trial.number}"
    suffix = "{epoch}-"
    ckpt_filename_suffix = f"{suffix}-{ckpt_monitor}"
    ckpt_filename = f"{trial_run_name}_{ckpt_filename_suffix}"

    callbacks_dict = create_hparams_callbacks(
        trial=trial,
        ckpt_monitor=ckpt_monitor,
        ckpt_mode=ckpt_mode,
        ckpt_filename=ckpt_filename,
        ckpt_save_top_k=ckpt_save_top_k,
        early_monitor=early_monitor,
        early_mode=early_mode,
        early_patience=early_patience,
        enable_progress_bar=enable_progress_bar,
        progress_bar_color=progress_bar_color,
        pruner_monitor=pruner_monitor,
    )
    callbacks_list = [cb for cb in callbacks_dict.values() if cb is not None]

    # 3) optional W&B
    wb = None
    if log_wandb:
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        wb = WandbLogger(
            project=project,
            entity=entity,
            name=trial_run_name,
            tags=list(tags) if tags else None,
            log_model=log_model,
        )

    try:
        # 4) trainer
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
            callbacks=callbacks_list,
        )

        # 5) fit
        trainer.fit(model, datamodule=data_module)

        # 6) return target metric
        metric = trainer.callback_metrics.get(pruner_monitor)
        if metric is None:
            raise RuntimeError(
                f"Metric '{pruner_monitor}' not found in callback_metrics. "
                f"Available: {list(trainer.callback_metrics.keys())}"
            )
        return float(metric.item())
    finally:
        if wb is not None:
            try:
                wb.experiment.finish()
            except Exception:
                pass


def create_sampler(seed=42, multivariate=True, group=True, n_startup_trials=20, constant_liar=False, n_ei_candidates  = 64):
    return TPESampler(
        seed=seed,
        multivariate=multivariate,
        group=group,
        n_startup_trials=n_startup_trials,
        constant_liar=constant_liar,
        n_ei_candidates = n_ei_candidates
    )


def create_pruner(pruner_kwargs):
  return optuna.pruners.HyperbandPruner(**pruner_kwargs)


def create_study(sampler, pruner, study_name="jav250_gcn", direction="maximize", storage=None, load_if_exists=True):
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    return study


def run_hptuning(
    data_module,
    search_space,
    study_name="jav250_gcn",
    direction="maximize",
    n_trials=50,
    load_if_exists=True,
    sampler_kwargs=None,
    pruner_kwargs=None,
    pruner_monitor="val_f1",
    model_type="GNN_Classifier",
    num_classes=250,
    accelerator="gpu",
    devices=1,
    max_epochs=20,
    precision="16-mixed",
    log_every_n_steps=10,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    ckpt_monitor="val_f1",
    ckpt_mode="max",
    ckpt_save_top_k=1,
    early_monitor="val_loss",
    early_mode="min",
    early_patience=5,
    enable_progress_bar=False,
    progress_bar_color="green",
    log_wandb=False,
    wandb_api_key=None,
    project="devign_debugging",
    entity="taoufik-k-idrissi-polytechnique-montr-al",
    tags=("jav250", "GCN", "baseline"),
    log_model=False,
    ):
    # 1) sampler / pruner
    sampler_kwargs = sampler_kwargs or {}
    sampler = create_sampler(**sampler_kwargs)
    pruner_kwargs = pruner_kwargs or {}
    pruner = create_pruner(pruner_kwargs)

    # 2) study
    study = create_study(
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction=direction,
        storage=f"sqlite:///./{study_name}.db",
        load_if_exists=load_if_exists,
    )
    #Figure out how many more studies to do
    done = len(study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)))
    n_trials = max(0, n_trials - done)
    # 3) objective wrapper
    def _objective(trial):
        return objective(
            trial,
            search_space=search_space,
            data_module=data_module,
            model_type=model_type,
            run_name_prefix=study_name,
            num_classes=num_classes,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            precision=precision,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            pruner_monitor=pruner_monitor,
            ckpt_monitor=ckpt_monitor,
            ckpt_mode=ckpt_mode,
            early_monitor=early_monitor,
            early_mode=early_mode,
            early_patience=early_patience,
            ckpt_save_top_k=ckpt_save_top_k,
            enable_progress_bar=enable_progress_bar,
            progress_bar_color=progress_bar_color,
            log_wandb=log_wandb,
            wandb_api_key=wandb_api_key,
            project=project,
            entity=entity,
            tags=tags,
            log_model=log_model,
        )

    # 4) optimize
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    # 5) summary
    print(f"[Optuna] Best trial: #{study.best_trial.number}")
    print(f"[Optuna] Best value: {study.best_value}")
    print(f"[Optuna] Best params: {study.best_trial.params}")
    return study

