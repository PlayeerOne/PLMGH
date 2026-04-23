from lightning.pytorch import LightningDataModule

from torch_geometric.loader import DataLoader as GeoLoader
from torch.utils.data import DataLoader as TorchLoader

import torch
from tqdm import tqdm

class GraphDataModuleCls(LightningDataModule):
    def __init__(
        self,
        train_ds,
        val_ds=None,
        test_ds=None,
        label_key: str = "y",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Filled during setup()
        self.has_been_setup = False
        self.num_classes: int | None = None
        self.is_binary: bool | None = None
        self.class_counts: torch.Tensor | None = None
        self.class_weights: torch.Tensor | None = None
        self.pos_weight: torch.Tensor | None = None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _iter_labels(self, ds):
        str_labels = [y for _, y in ds.id_to_str_label.items()]
        for str_y in str_labels:
            y = ds.str_to_index(str_y)
            y = torch.as_tensor(y).view(-1)
            yield y.long()

    def _infer_classes_from_all_splits(self):
        uniques = set()
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if ds is None:
                continue
            for y in self._iter_labels(ds):
                uniques.update(y.unique().tolist())
        if not uniques:
            raise ValueError("No labels found across splits.")
        uniq_sorted = sorted(int(u) for u in uniques)

        # Assume labels are already 0-based and dense:
        self.num_classes = int(max(uniq_sorted) + 1)
        self.is_binary = (len(uniq_sorted) == 2 and set(uniq_sorted) == {0, 1})

    def _compute_train_stats(self):
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for y in self._iter_labels(self.train_ds):
            counts += torch.bincount(y, minlength=self.num_classes)
        counts = torch.clamp(counts, min=1)
        self.class_counts = counts

        if self.is_binary:
            pos = int(counts[1].item()) if self.num_classes > 1 else 0
            neg = int(counts[0].item())
            w = 1.0 if pos == 0 else (neg / max(1, pos))  # for BCEWithLogits
            self.pos_weight = torch.tensor(float(w), dtype=torch.float32)
            self.class_weights = None
        else:
            total = float(counts.sum().item())
            inv = total / (counts.float() * self.num_classes)  # inverse freq
            inv = inv * (self.num_classes / inv.sum())         # mean=1
            self.class_weights = inv.to(dtype=torch.float32)
            self.pos_weight = None

    # ----------------------------
    # Lightning hooks
    # ----------------------------
    def setup(self, stage=None):
        if self.has_been_setup == False:
            self._infer_classes_from_all_splits()
            self._compute_train_stats()
            self.has_been_setup = True

    # ----------------------------
    # Accessors for the model
    # ----------------------------
    def get_num_classes(self) -> int:
        if self.num_classes is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.num_classes

    def get_is_binary(self) -> bool:
        if self.is_binary is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.is_binary

    def get_class_weights(self, device=None):
        if self.class_weights is None:
            return None
        return self.class_weights.to(device) if device is not None else self.class_weights

    def get_pos_weight(self, device=None):
        if self.pos_weight is None:
            return None
        return self.pos_weight.to(device) if device is not None else self.pos_weight

    # ----------------------------
    # DataLoaders
    # ----------------------------
    def train_dataloader(self):
        return GeoLoader(
            self.train_ds,
            drop_last=False, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return GeoLoader(
            self.val_ds,
            drop_last=False, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return GeoLoader(
            self.test_ds,
            drop_last=False, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )


class MLPDataModuleCls(LightningDataModule):
    def __init__(
        self,
        train_ds,
        val_ds=None,
        test_ds=None,
        label_key: str = "y",   # kept for API symmetry, not actually used
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.label_key = label_key
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Filled during setup()
        self.has_been_setup = False
        self.num_classes = None
        self.is_binary = None
        self.class_counts = None
        self.class_weights = None
        self.pos_weight = None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _iter_labels(self, ds):
        """
        Iterate over all labels in a dataset as tensors of label indices.
        Uses the id_to_str_label + str_to_index mapping, same as your graph module.
        """
        str_labels = [y for _, y in ds.id_to_str_label.items()]
        for str_y in str_labels:
            y = ds.str_to_index(str_y)
            y = torch.as_tensor(y).view(-1)
            yield y.long()

    def _infer_classes_from_all_splits(self):
        uniques = set()
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if ds is None:
                continue
            for y in self._iter_labels(ds):
                uniques.update(y.unique().tolist())
        if not uniques:
            raise ValueError("No labels found across splits.")
        uniq_sorted = sorted(int(u) for u in uniques)

        # Assume labels are already 0-based and dense:
        self.num_classes = int(max(uniq_sorted) + 1)
        self.is_binary = (len(uniq_sorted) == 2 and set(uniq_sorted) == {0, 1})

    def _compute_train_stats(self):
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for y in self._iter_labels(self.train_ds):
            counts += torch.bincount(y, minlength=self.num_classes)
        counts = torch.clamp(counts, min=1)
        self.class_counts = counts

        if self.is_binary:
            pos = int(counts[1].item()) if self.num_classes > 1 else 0
            neg = int(counts[0].item())
            w = 1.0 if pos == 0 else (neg / max(1, pos))  # for BCEWithLogits
            self.pos_weight = torch.tensor(float(w), dtype=torch.float32)
            self.class_weights = None
        else:
            total = float(counts.sum().item())
            inv = total / (counts.float() * self.num_classes)  # inverse freq
            inv = inv * (self.num_classes / inv.sum())         # mean=1
            self.class_weights = inv.to(dtype=torch.float32)
            self.pos_weight = None

    # ----------------------------
    # Lightning hooks
    # ----------------------------
    def setup(self, stage=None):
        if not self.has_been_setup:
            self._infer_classes_from_all_splits()
            self._compute_train_stats()
            self.has_been_setup = True

    # ----------------------------
    # Accessors for the model
    # ----------------------------
    def get_num_classes(self) -> int:
        if self.num_classes is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.num_classes

    def get_is_binary(self) -> bool:
        if self.is_binary is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.is_binary

    def get_class_weights(self, device=None):
        if self.class_weights is None:
            return None
        return self.class_weights.to(device) if device is not None else self.class_weights

    def get_pos_weight(self, device=None):
        if self.pos_weight is None:
            return None
        return self.pos_weight.to(device) if device is not None else self.pos_weight

    # ----------------------------
    # DataLoaders
    # ----------------------------
    def train_dataloader(self):
        return TorchLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return TorchLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return TorchLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )






# keep your plm_collate_fn
def plm_collate_fn(batch, tokenizer):
  import warnings
  
  input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
  attention_mask = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
  labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt",
        )
  padded["labels"] = labels
  padded["sample_ids"] = [x["sample_id"] for x in batch]
  if "path" in batch[0]:
      padded["paths"] = [x.get("path") for x in batch]
  if "features" in batch[0]:
      padded["features"] = torch.stack([x["features"] for x in batch])
  return padded
  
class TextDataModuleCls(LightningDataModule):
    def __init__(
        self,
        train_ds,
        val_ds=None,
        test_ds=None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.tokenizer = train_ds.tokenizer     # <-- store tokenizer

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.has_been_setup = False
        self.num_classes  = None
        self.is_binary  = None
        self.class_counts  = None
        self.class_weights = None
        self.pos_weight = None

    # --- helpers (same as yours) ---
    def _iter_labels(self, ds):
        for _, str_y in ds.id_to_str_label.items():
            y = ds.str_to_index(str_y)
            yield torch.as_tensor(y).view(-1).long()

    def _infer_classes_from_all_splits(self):
        uniques = set()
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            if ds is None:
                continue
            for y in self._iter_labels(ds):
                uniques.update(y.unique().tolist())
        if not uniques:
            raise ValueError("No labels found across splits.")
        uniq_sorted = sorted(int(u) for u in uniques)
        self.num_classes = int(max(uniq_sorted) + 1)
        self.is_binary = (len(uniq_sorted) == 2 and set(uniq_sorted) == {0, 1})

    def _compute_train_stats(self):
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for y in self._iter_labels(self.train_ds):
            counts += torch.bincount(y, minlength=self.num_classes)
        counts = torch.clamp(counts, min=1)
        self.class_counts = counts
        if self.is_binary:
            pos = int(counts[1].item()) if self.num_classes > 1 else 0
            neg = int(counts[0].item())
            w = 1.0 if pos == 0 else (neg / max(1, pos))
            self.pos_weight = torch.tensor(float(w), dtype=torch.float32)
            self.class_weights = None
        else:
            total = float(counts.sum().item())
            inv = total / (counts.float() * self.num_classes)
            inv = inv * (self.num_classes / inv.sum())
            self.class_weights = inv.to(torch.float32)
            self.pos_weight = None

    def setup(self, stage=None):
        if not self.has_been_setup:
            self._infer_classes_from_all_splits()
            self._compute_train_stats()
            self.has_been_setup = True

    def get_num_classes(self) -> int:
        if self.num_classes is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.num_classes

    def get_is_binary(self) -> bool:
        if self.is_binary is None:
            raise RuntimeError("DataModule not set up yet.")
        return self.is_binary

    def get_class_weights(self, device=None):
        return None if self.class_weights is None else (self.class_weights.to(device) if device else self.class_weights)

    def get_pos_weight(self, device=None):
        return None if self.pos_weight is None else (self.pos_weight.to(device) if device else self.pos_weight)

    # --- DataLoaders with collate_fn ---
    def train_dataloader(self):
        return TorchLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=lambda b: plm_collate_fn(b, self.tokenizer),
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return TorchLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=lambda b: plm_collate_fn(b, self.tokenizer),
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None
        return TorchLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=lambda b: plm_collate_fn(b, self.tokenizer),
        )



