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
    # Keep the notebooks usable in environments that still depend on the older
    # `pytorch_lightning` package name.
    import pytorch_lightning as pl


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call `fn` with only the keyword arguments its signature accepts."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins or C-extension callables may not expose a Python signature.
        # In that case we fall back to a direct call and let Python raise naturally.
        return fn(**kwargs)

    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_kwargs:
        # When the callable already accepts arbitrary keyword arguments there is
        # nothing to filter, so we can pass the full context through unchanged.
        return fn(**kwargs)

    # Keep only the keyword arguments that the target callable explicitly declares.
    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
    }

    # Surface missing required keyword arguments early with an error message that
    # points to the actual factory function instead of failing deeper inside it.
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
        # Copy to a plain dict so downstream code can mutate safely without
        # affecting the caller's original configuration object.
        normalized_config = dict(scheduler_config)
        scheduler = normalized_config.get("scheduler")
        # ReduceLROnPlateau is special in Lightning: it must know which logged
        # metric to watch, otherwise training setup fails at runtime.
        if isinstance(scheduler, ReduceLROnPlateau) and "monitor" not in normalized_config:
            normalized_config["monitor"] = monitor
        return normalized_config

    # Lightning also accepts a bare scheduler instance, so wrap it into the
    # dict shape it expects when paired with an optimizer config dict.
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
        # The simplest case: the factory already returned a ready-to-use optimizer.
        return optimizer_config

    if isinstance(optimizer_config, tuple):
        if len(optimizer_config) == 1:
            # Preserve backward compatibility with helpers that return `(optimizer,)`.
            return optimizer_config[0]

        if len(optimizer_config) == 2:
            # Interpret two-tuples as `(optimizer, scheduler)` because that is the
            # most common compact return shape used in notebooks.
            optimizer, scheduler = optimizer_config
            if scheduler is None:
                # Treat `(optimizer, None)` the same as returning the optimizer alone.
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": _normalize_scheduler_config(scheduler, monitor),
            }

        if len(optimizer_config) == 3:
            # The third element lets a helper override which metric a scheduler
            # should monitor without rebuilding the full Lightning config dict.
            optimizer, scheduler, custom_monitor = optimizer_config
            if scheduler is None:
                return optimizer

            return {
                "optimizer": optimizer,
                # Prefer an explicit monitor returned by the factory, but fall back
                # to the module default so ReduceLROnPlateau still works.
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

    # If the caller already returned a Lightning-native config dict, keep it as-is.
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
            # Allow callers to pass a fully constructed backbone directly.
            self.model = model
        elif load_model_fn is not None:
            # Factories across notebooks do not all share the same signature, so
            # pass a rich context and let `_call_with_supported_kwargs` trim it.
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

        # Keep loss and metrics on the module so all steps share the same
        # training/validation/test bookkeeping setup.
        self.loss_fn = nn.CrossEntropyLoss()
        # Reuse one metric object so Lightning can aggregate validation/test
        # accuracy across batches and epochs consistently.
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Run a forward pass through the wrapped backbone."""
        # Delegate to the wrapped model so notebook code can still treat this
        # LightningModule like the original backbone during inference.
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """Compute and log training loss for one batch."""
        # Lightning dataloaders conventionally return `(inputs, labels)`.
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        # Notebook progress bars are cleaner when training loss is shown once per
        # epoch instead of being updated on every optimization step.
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        """Compute and log validation loss plus accuracy for one batch."""
        # Validation mirrors training, but also records accuracy because model
        # selection in the notebooks usually depends on validation performance.
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        # `val_loss` is also the default monitor consumed by optimizer/scheduler
        # factories through `configure_optimizers`.
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx=None):
        """Compute and log test loss plus accuracy for one batch."""
        # Test logging matches validation naming conventions so downstream plots
        # and summaries can read them predictably.
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Build the optimizer configuration expected by Lightning."""
        # Pass both the wrapped backbone and the LightningModule itself so helper
        # factories can choose the abstraction level they want to work with.
        optimizer_config = _call_with_supported_kwargs(
            self.optimizer_scheduler_fn,
            model=self.model,
            module=self,
            lightning_module=self,
            num_classes=self.hparams.num_classes,
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # Normalize the return shape because notebook helpers may return either a
        # bare optimizer, a tuple, or a full Lightning config dict.
        return _normalize_optimizer_config(optimizer_config, monitor="val_loss")


def Create_flexible_Food101Classifier(**kwargs: Any) -> FlexibleFood101Classifier:
    """Preserve the notebook's original constructor name for backward compatibility."""
    # Keep the old factory-like notebook API working after the refactor to a
    # class-based implementation.
    return FlexibleFood101Classifier(**kwargs)


__all__ = [
    "Create_flexible_Food101Classifier",
    "FlexibleFood101Classifier",
]
