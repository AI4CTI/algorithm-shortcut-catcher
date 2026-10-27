"""Frozen ELECTRA encoder with a trainable pooling head and MLP classifier (experimental).

Ported from the legacy text implementation: the encoder is frozen, a gated
(or plain) attention pooling layer turns token states into one vector, and a
one-hidden-layer MLP classifies it. Each fit trains a fresh head with AdamW,
a cosine schedule with warmup, and keeps the epoch with the best validation
score. Encoder outputs are cached by text, since a frozen encoder always
returns the same states for the same text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .data import TextData

#: Default encoder. The legacy experiments used ``google/electra-base-discriminator``.
DEFAULT_CHECKPOINT = "google/electra-small-discriminator"
HEADS = ("gate_weight_pooling", "normal_weight_pooling")
CHECKPOINT_METRICS = ("validation_f1_macro", "validation_loss")


def content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask padding, the leading CLS token and the final SEP token."""
    mask = attention_mask.to(dtype=torch.bool).clone()
    mask[:, 0] = False
    last = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
    mask[torch.arange(mask.shape[0]), last] = False
    return mask


def weighted_pool(
    hidden: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax token scores over content tokens and pool the hidden states."""
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=1) * mask.to(scores.dtype)
    total = weights.sum(dim=1, keepdim=True)
    weights = torch.where(
        total > 0,
        weights / total.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )
    return torch.sum(hidden * weights.unsqueeze(-1), dim=1), weights


class GateWeightPooling(torch.nn.Module):
    """Gated attention pooling (Ilse et al., ICML 2018, eq. 9)."""

    def __init__(self, hidden_size: int, attention_size: int = 128) -> None:
        super().__init__()
        self.V = torch.nn.Linear(hidden_size, attention_size, bias=False)
        self.U = torch.nn.Linear(hidden_size, attention_size, bias=False)
        self.w = torch.nn.Linear(attention_size, 1, bias=False)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.V(hidden)) * torch.sigmoid(self.U(hidden))
        return weighted_pool(hidden, self.w(gated).squeeze(-1), mask)


class NormalWeightPooling(torch.nn.Module):
    """One learned scalar score per token."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score_projection = torch.nn.Linear(hidden_size, 1)

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return weighted_pool(hidden, self.score_projection(hidden).squeeze(-1), mask)


class PoolingHead(torch.nn.Module):
    """Pooling layer followed by Linear -> GELU -> Dropout -> Linear."""

    def __init__(
        self, hidden_size: int, num_classes: int, head: str, *, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}.")
        self.pooler: torch.nn.Module = (
            GateWeightPooling(hidden_size)
            if head == "gate_weight_pooling"
            else NormalWeightPooling(hidden_size)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = content_mask(attention_mask)
        pooled, weights = self.pooler(hidden, mask)
        # A text with no content tokens falls back to the CLS state.
        pooled = torch.where(mask.any(dim=1, keepdim=True), pooled, hidden[:, 0])
        return self.classifier(pooled), weights


@dataclass(frozen=True)
class Encoded:
    """Encoder output for one text, truncated to the model window."""

    hidden: np.ndarray  # (tokens, hidden_size), float32
    offsets: tuple[tuple[int, int], ...]


def _collate(rows: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(r) for r in rows)
    hidden = torch.zeros(len(rows), length, rows[0].shape[1])
    mask = torch.zeros(len(rows), length, dtype=torch.long)
    for i, row in enumerate(rows):
        hidden[i, : len(row)] = torch.from_numpy(row)
        mask[i, : len(row)] = 1
    return hidden, mask


class ElectraModel:
    """Train a pooling head on a frozen ELECTRA encoder.

    Args:
        checkpoint: Hugging Face model id or local directory.
        head: ``"gate_weight_pooling"`` (legacy default) or ``"normal_weight_pooling"``.
        epochs, batch_size, learning_rate: Training settings. The legacy
            experiments used 30 epochs, batch 128, learning rate 0.005.
        checkpoint_metric: Picks the best epoch: ``"validation_f1_macro"`` or ``"validation_loss"``.
        early_stopping_patience: Stop after this many epochs without improvement; None disables it.
        max_length: Token window of the encoder.
        device: Torch device; defaults to CUDA when available, else CPU.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        head: str = "gate_weight_pooling",
        epochs: int = 30,
        batch_size: int = 128,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.1,
        gradient_clip: float = 1.0,
        dropout: float = 0.1,
        checkpoint_metric: str = "validation_f1_macro",
        early_stopping_patience: int | None = None,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}.")
        if checkpoint_metric not in CHECKPOINT_METRICS:
            raise ValueError(f"checkpoint_metric must be one of {CHECKPOINT_METRICS}.")
        for name, value in (
            ("epochs", epochs),
            ("batch_size", batch_size),
            ("max_length", max_length),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.checkpoint = checkpoint
        self.head = head
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_clip = gradient_clip
        self.dropout = dropout
        self.checkpoint_metric = checkpoint_metric
        self.early_stopping_patience = early_stopping_patience
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._backbone: Any = None
        self._tokenizer: Any = None
        self._cache: dict[str, Encoded] = {}

    @classmethod
    def from_components(cls, backbone: Any, tokenizer: Any, **kwargs: Any) -> ElectraModel:
        """Use an already loaded encoder and fast tokenizer (e.g. for offline tests)."""
        model = cls(checkpoint="<components>", **kwargs)
        model._set_encoder(backbone, tokenizer)
        return model

    def _set_encoder(self, backbone: Any, tokenizer: Any) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for word offsets.")
        backbone.to(self.device)
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self._backbone, self._tokenizer = backbone, tokenizer

    def _load(self) -> None:
        if self._backbone is None:
            from transformers import AutoModel, AutoTokenizer

            self._set_encoder(
                AutoModel.from_pretrained(self.checkpoint),
                AutoTokenizer.from_pretrained(self.checkpoint, use_fast=True),
            )

    @property
    def hidden_size(self) -> int:
        self._load()
        return int(self._backbone.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> list[Encoded]:
        """Return encoder states and token offsets per text, computing only unseen texts."""
        self._load()
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").tolist()
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                hidden = self._backbone(**inputs).last_hidden_state.float().cpu().numpy()
            lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for i, text in enumerate(batch):
                n = int(lengths[i])
                self._cache[text] = Encoded(
                    hidden=np.ascontiguousarray(hidden[i, :n]),
                    offsets=tuple((int(a), int(b)) for a, b in offsets[i][:n]),
                )
        return [self._cache[t] for t in texts]

    def clear_cache(self) -> None:
        self._cache.clear()

    def fit(self, train: TextData, validation: TextData | None, *, seed: int) -> FittedElectra:
        """Train a fresh head; the best epoch is chosen on ``validation``."""
        if validation is None:
            raise ValueError("The text backend needs a validation split to pick the best epoch.")
        classes = sorted(set(train.labels), key=str)
        if len(classes) < 2:
            raise ValueError("Training data must contain at least two classes.")
        index = {label: i for i, label in enumerate(classes)}
        unknown = set(validation.labels) - set(index)
        if unknown:
            raise ValueError(f"Validation labels absent from train: {sorted(unknown, key=str)}.")

        train_rows = [e.hidden for e in self.encode(train.texts)]
        val_rows = [e.hidden for e in self.encode(validation.texts)]
        y_train = torch.tensor([index[label] for label in train.labels])
        y_val = np.array([index[label] for label in validation.labels])

        # Seeded locally so the global torch RNG is left as it was.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            head = PoolingHead(self.hidden_size, len(classes), self.head, dropout=self.dropout)
        head.to(self.device)
        generator = torch.Generator().manual_seed(seed)

        from transformers import get_cosine_schedule_with_warmup

        optimizer = torch.optim.AdamW(
            head.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        steps_per_epoch = max(1, math.ceil(len(train_rows) / self.batch_size))
        total = self.epochs * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.warmup_ratio * total), num_training_steps=total
        )

        minimize = self.checkpoint_metric == "validation_loss"
        best_value = math.inf if minimize else -math.inf
        best_state: dict[str, torch.Tensor] | None = None
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)  # dropout masks
            for epoch in range(1, self.epochs + 1):
                head.train()
                order = torch.randperm(len(train_rows), generator=generator).tolist()
                for start in range(0, len(order), self.batch_size):
                    rows = order[start : start + self.batch_size]
                    hidden, mask = _collate([train_rows[i] for i in rows])
                    logits, _ = head(hidden.to(self.device), mask.to(self.device))
                    loss = torch.nn.functional.cross_entropy(logits, y_train[rows].to(self.device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), self.gradient_clip)
                    optimizer.step()
                    scheduler.step()
                proba = _predict_proba(head, val_rows, self.batch_size, self.device)
                metrics = {
                    "epoch": float(epoch),
                    "validation_loss": _nll(y_val, proba),
                    "validation_f1_macro": float(
                        f1_score(y_val, proba.argmax(axis=1), average="macro", zero_division=0)
                    ),
                }
                history.append(metrics)
                value = metrics[self.checkpoint_metric]
                if (value < best_value) if minimize else (value > best_value):
                    best_value, best_epoch, stale = value, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                else:
                    stale += 1
                    if (
                        self.early_stopping_patience is not None
                        and stale >= self.early_stopping_patience
                    ):
                        break
        assert best_state is not None
        head.load_state_dict(best_state)
        head.eval()
        return FittedElectra(
            encoder=self,
            head=head,
            classes=tuple(classes),
            best_epoch=best_epoch,
            history=tuple(history),
        )


def _predict_proba(
    head: PoolingHead, rows: Sequence[np.ndarray], batch_size: int, device: torch.device
) -> np.ndarray:
    head.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            hidden, mask = _collate(rows[start : start + batch_size])
            logits, _ = head(hidden.to(device), mask.to(device))
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0), dtype=np.float32)


def _nll(y: np.ndarray, proba: np.ndarray) -> float:
    picked = proba[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))


@dataclass(frozen=True)
class FittedElectra:
    """A trained pooling head on the shared frozen encoder."""

    encoder: ElectraModel
    head: PoolingHead
    classes: tuple[Any, ...]
    best_epoch: int
    history: tuple[dict[str, float], ...]

    def predict_proba(self, data: TextData) -> np.ndarray:
        rows = [e.hidden for e in self.encoder.encode(data.texts)]
        return _predict_proba(self.head, rows, self.encoder.batch_size, self.encoder.device)

    def predict(self, data: TextData) -> list[Any]:
        return [self.classes[i] for i in self.predict_proba(data).argmax(axis=1)]

    def evaluate(self, data: TextData) -> dict[str, float]:
        """Return accuracy, macro F1 and NLL (NLL only over labels seen in training)."""
        proba = self.predict_proba(data)
        pred = [self.classes[i] for i in proba.argmax(axis=1)]
        y = list(data.labels)
        index = {label: i for i, label in enumerate(self.classes)}
        known = [i for i, label in enumerate(y) if label in index]
        nll = _nll(np.array([index[y[i]] for i in known]), proba[known]) if known else float("nan")
        labels = sorted(set(y) | set(pred), key=str)
        y_str = [str(v) for v in y]
        p_str = [str(v) for v in pred]
        return {
            "accuracy": float(accuracy_score(y_str, p_str)),
            "f1_macro": float(
                f1_score(
                    y_str, p_str, average="macro", labels=[str(v) for v in labels], zero_division=0
                )
            ),
            "nll": nll,
        }

    def token_weights(
        self, texts: Sequence[str]
    ) -> list[tuple[tuple[tuple[int, int], ...], np.ndarray]]:
        """Return (token offsets, pooling weights) per text."""
        encoded = self.encoder.encode(texts)
        result = []
        self.head.eval()
        with torch.inference_mode():
            for start in range(0, len(encoded), self.encoder.batch_size):
                chunk = encoded[start : start + self.encoder.batch_size]
                hidden, mask = _collate([e.hidden for e in chunk])
                _, weights = self.head(hidden.to(self.encoder.device), mask.to(self.encoder.device))
                weights_np = weights.float().cpu().numpy()
                for i, e in enumerate(chunk):
                    result.append((e.offsets, weights_np[i, : len(e.offsets)]))
        return result
