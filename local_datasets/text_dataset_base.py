# plm_dataset_base.py  (patch)
import os, pickle, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.preprocessing.utils import get_tokenizer



def _pad_batch(ids_seqs: List[List[int]], pad_id: int):
    L = max(len(x) for x in ids_seqs)
    B = len(ids_seqs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.bool)
    for i, seq in enumerate(ids_seqs):
        L_i = len(seq)
        ids[i, :L_i] = torch.tensor(seq, dtype=torch.long)
        attn[i, :L_i] = True
    return ids, attn

class TextDatasetBase(Dataset):

    def __init__(
        self,
        model_name,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        
        self.model_name = model_name

        tok = get_tokenizer(model_name, max_length)

        self.tokenizer = tok["tokenizer"]
        self.max_length = tok["max_length"]

        # --- subclass-provided mappings ---
        self.id_to_text: Dict[str, str] = self.resolve_id_to_text()
        self.id_to_str_label: Dict[str, str] = self.resolve_id_to_str_label()
        self.str_label_to_idx: Dict[str, int] = self.resolve_str_label_to_idx()

        # Optional path mapping
        self.id_to_path: Dict[str, Optional[str]] = {}
        if hasattr(self, "resolve_id_to_path"):
            try:
                self.id_to_path = dict(self.resolve_id_to_path())  # type: ignore
            except NotImplementedError:
                self.id_to_path = {}

        ids = set(self.id_to_text) & set(self.id_to_str_label)
        if self.id_to_path:
            ids &= set(self.id_to_path)
        self.ids: List[str] = sorted(ids)

        # Label map sanity (dense 0..C-1)
        max_idx = max(self.str_label_to_idx.values()) if self.str_label_to_idx else -1
        self.idx_to_str: List[Optional[str]] = [None] * (max_idx + 1)
        for s, i in self.str_label_to_idx.items():
            if i >= len(self.idx_to_str):
                self.idx_to_str.extend([None] * (i - len(self.idx_to_str) + 1))
            self.idx_to_str[i] = s
        if any(v is None for v in self.idx_to_str):
            raise ValueError("Label indices must be dense 0..C-1 with no gaps.")

    # ---------------- abstract resolvers ----------------
    def resolve_id_to_text(self) -> Dict[str, str]:
        raise NotImplementedError

    def resolve_id_to_str_label(self) -> Dict[str, str]:
        raise NotImplementedError

    def resolve_str_label_to_idx(self) -> Dict[str, int]:
        raise NotImplementedError

    def resolve_id_to_path(self) -> Dict[str, str]:
        raise NotImplementedError

    # ---------------- text access ----------------
    def get_text(self, sid: str) -> str:
        txt = self.id_to_text.get(sid, "")
        if txt:
            return txt
        p = self.id_to_path.get(sid)
        if p:
            path = Path(p)
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return path.read_text(errors="ignore")
        raise KeyError(f"No text or path available for sample_id={sid}")

    # ---------------- tokenization ----------------
    def _tokenize(self, sid: str) -> Dict[str, List[int]]:

        text = self.get_text(sid)
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"][0] if isinstance(enc["input_ids"][0], list) else enc["input_ids"]
        attention_mask = enc["attention_mask"][0] if isinstance(enc["attention_mask"][0], list) else enc["attention_mask"]

        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        return out


    # ---------------- dunder ----------------
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.ids[idx]
        label_idx = self.str_to_index(self.id_to_str_label[sid])

        toks = self._tokenize(sid)
        item = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "labels": label_idx,
            "sample_id": sid,
        }
        p = self.id_to_path.get(sid)
        if p is not None:
            item["path"] = p

        return item

    # label helpers
    def str_to_index(self, label: str) -> int:
        return self.str_label_to_idx[label]

    def index_to_str(self, idx: int) -> str:
        return self.idx_to_str[idx]  # type: ignore

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_str)

    def sklearn_class_weight(self, mode: str = "balanced") -> Dict[int, float]:
      """
      Return a scikit-learn compatible class_weight mapping {class_idx: weight}.
      mode="balanced" matches sklearn's heuristic using label frequencies.
      """
      import numpy as np
      from collections import Counter

      labels = [self.str_to_index(self.id_to_str_label[sid]) for sid in self.ids]
      cnt = Counter(labels)
      classes = sorted(cnt.keys())
      freqs = np.array([cnt[c] for c in classes], dtype=np.float64)

      if mode == "balanced":
          # sklearn's 'balanced': n_samples / (n_classes * count_c)
          n_samples = float(len(labels))
          n_classes = float(len(classes))
          weights = n_samples / (n_classes * freqs)
      else:
          raise ValueError(f"Unsupported mode={mode}")

      return {int(c): float(w) for c, w in zip(classes, weights)}

    def sklearn_label_names(self) -> List[str]:
      """
      Returns a list mapping class index -> original string label.
      """
      return [s if s is not None else f"<UNK_{i}>" for i, s in enumerate(self.idx_to_str)]

    def sklearn_prevalence(self) -> Dict[int, float]:
      """
      Returns class prevalence {class_idx: fraction in [0,1]} over the whole dataset.
      Useful to report the random baseline for AUPRC on imbalanced tasks.
      """
      from collections import Counter
      import numpy as np

      labels = [self.str_to_index(self.id_to_str_label[sid]) for sid in self.ids]
      cnt = Counter(labels)
      total = float(len(labels))
      return {k: float(v / total) for k, v in cnt.items()}
      return path
