"""Training utilities shared by the Food-101 notebooks."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call `fn` with only the keyword arguments its signature accepts."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)

    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
    }

    missing_required = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect._empty
        and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in supported_kwargs
    ]
    if missing_required:
        missing = ", ".join(missing_required)
        raise TypeError(
            f"{getattr(fn, '__name__', repr(fn))} is missing required arguments: {missing}"
        )

    return fn(**supported_kwargs)


def _normalize_scheduler_config(
    scheduler_config: Any,
    monitor: str,
) -> dict[str, Any]:
    """Normalize Lightning scheduler configs and inject `monitor` when required."""
    if isinstance(scheduler_config, Mapping):
        normalized_config = dict(scheduler_config)
        scheduler = normalized_config.get("scheduler")
        if isinstance(scheduler, ReduceLROnPlateau) and "monitor" not in normalized_config:
            normalized_config["monitor"] = monitor
        return normalized_config

    normalized_config = {"scheduler": scheduler_config}
    if isinstance(scheduler_config, ReduceLROnPlateau):
        normalized_config["monitor"] = monitor
    return normalized_config


def _normalize_optimizer_config(
    optimizer_config: Any,
    monitor: str,
) -> Any:
    """Convert several optimizer/scheduler return shapes into Lightning format."""
    if isinstance(optimizer_config, Optimizer):
        return optimizer_config

    if isinstance(optimizer_config, tuple):
        if len(optimizer_config) == 1:
            return optimizer_config[0]

        if len(optimizer_config) == 2:
            optimizer, scheduler = optimizer_config
            if scheduler is None:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": _normalize_scheduler_config(scheduler, monitor),
            }

        if len(optimizer_config) == 3:
            optimizer, scheduler, custom_monitor = optimizer_config
            if scheduler is None:
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": _normalize_scheduler_config(
                    scheduler,
                    custom_monitor or monitor,
                ),
            }

        raise ValueError(
            "optimizer_scheduler_fn returned an unsupported tuple shape. "
            "Use (optimizer,), (optimizer, scheduler), or "
            "(optimizer, scheduler, monitor)."
        )

    return optimizer_config


class FlexibleFood101Classifier(pl.LightningModule):
    """Reusable LightningModule that wraps a backbone factory plus optimizer factory."""

    def __init__(
        self,
        num_classes: int = 101,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        model: nn.Module | None = None,
        load_model_fn: Callable[..., nn.Module] | None = None,
        optimizer_scheduler_fn: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()

        # Ignore callable factories because Lightning cannot serialize them directly.
        self.save_hyperparameters(
            ignore=[
                "model",
                "load_model_fn",
                "optimizer_scheduler_fn",
            ]
        )

        self.load_model_fn = load_model_fn
        self.optimizer_scheduler_fn = optimizer_scheduler_fn

        if model is not None:
            self.model = model
        elif load_model_fn is not None:
            self.model = _call_with_supported_kwargs(
                load_model_fn,
                num_classes=num_classes,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError("You must provide either `model` or `load_model_fn`.")

        if optimizer_scheduler_fn is None:
            raise ValueError("You must provide `optimizer_scheduler_fn`.")

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Run a forward pass through the wrapped backbone."""
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """Compute and log training loss for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        """Compute and log validation loss plus accuracy for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx=None):
        """Compute and log test loss plus accuracy for one batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Build the optimizer configuration expected by Lightning."""
        optimizer_config = _call_with_supported_kwargs(
            self.optimizer_scheduler_fn,
            model=self.model,
            module=self,
            lightning_module=self,
            num_classes=self.hparams.num_classes,
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return _normalize_optimizer_config(optimizer_config, monitor="val_loss")


def Create_flexible_Food101Classifier(**kwargs: Any) -> FlexibleFood101Classifier:
    """Preserve the notebook's original constructor name for backward compatibility."""
    return FlexibleFood101Classifier(**kwargs)


__all__ = [
    "Create_flexible_Food101Classifier",
    "FlexibleFood101Classifier",
]
