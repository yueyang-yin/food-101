# Training utilities shared by the Food-101 notebook.

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy
from torchvision import datasets, models as tv_models, transforms

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class TransformBundle:
    # Keep train/val/test transforms grouped together so notebook cells can swap
    # between presets without threading three separate variables around.
    train: Any
    val: Any
    test: Any


# Build the default ImageNet-style augmentation/evaluation transforms used by
# the notebook's transfer-learning experiments.
def create_food101_transforms(
    train_crop_size: int = 224,
    eval_resize_size: int = 256,
    eval_crop_size: int | None = None,
    mean: list[float] | tuple[float, ...] = IMAGENET_MEAN,
    std: list[float] | tuple[float, ...] = IMAGENET_STD,
) -> TransformBundle:
    eval_crop_size = eval_crop_size or train_crop_size

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize_size),
            transforms.CenterCrop(eval_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return TransformBundle(
        train=train_transform,
        val=eval_transform,
        test=eval_transform,
    )


def create_datasets(
    data_dir: str,
    train_transform: Any,
    val_transform: Any,
    test_transform: Any,
    seed: int = 42,
):
    # Food-101 ships only train/test splits, so we create our own validation
    # subset from the training split while keeping the official test split intact.
    train_source = datasets.Food101(
        root=data_dir,
        split="train",
        transform=train_transform,
        download=False,
    )

    val_source = datasets.Food101(
        root=data_dir,
        split="train",
        transform=val_transform,
        download=False,
    )

    test_dataset = datasets.Food101(
        root=data_dir,
        split="test",
        transform=test_transform,
        download=False,
    )

    if hasattr(train_source, "_labels"):
        labels = train_source._labels
    elif hasattr(train_source, "targets"):
        labels = train_source.targets
    else:
        labels = [train_source[i][1] for i in range(len(train_source))]

    # Group indices by class first so the sampled validation split stays roughly
    # class-balanced instead of depending on the dataset's original ordering.
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in label_to_indices.values():
        indices_tensor = torch.tensor(indices)
        permutation = torch.randperm(len(indices_tensor), generator=generator)
        shuffled_indices = indices_tensor[permutation].tolist()

        # Reserve 20% per class for validation, but keep at least one sample so
        # very small classes still appear in both splits.
        split_point = max(1, int(len(shuffled_indices) * 0.2))
        val_indices.extend(shuffled_indices[:split_point])
        train_indices.extend(shuffled_indices[split_point:])

    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)
    return train_dataset, val_dataset, test_dataset


# Select the requested split and build a DataLoader with split-appropriate
# shuffling behavior.
def load_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
    split: str,
    num_workers: int,
):
    if split == "train":
        dataset = train_dataset
        shuffle = True
    elif split == "val":
        dataset = val_dataset
        shuffle = False
    elif split == "test":
        dataset = test_dataset
        shuffle = False
    else:
        raise ValueError(f"Unsupported split: {split}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class Food101DataModule(pl.LightningDataModule):
    # LightningDataModule wrapper that centralizes transforms, split creation,
    # and loader configuration for the Food-101 notebook.
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        seed: int = 42,
        transforms_bundle: TransformBundle | None = None,
        train_transform: Any | None = None,
        val_transform: Any | None = None,
        test_transform: Any | None = None,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        default_transforms = transforms_bundle or create_food101_transforms()
        self.train_transform = train_transform or default_transforms.train
        self.val_transform = val_transform or default_transforms.val
        self.test_transform = test_transform or default_transforms.test

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        # Build datasets lazily so repeated `setup()` calls from Lightning do not
        # recreate the same split objects.
        if (
            self.train_dataset is None
            and self.val_dataset is None
            and self.test_dataset is None
        ):
            self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(
                data_dir=self.data_dir,
                train_transform=self.train_transform,
                val_transform=self.val_transform,
                test_transform=self.test_transform,
                seed=self.seed,
            )

    def train_dataloader(self):
        # Training keeps shuffling enabled inside `load_dataloader`.
        return load_dataloader(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=self.batch_size,
            split="train",
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Validation is deterministic so metrics are comparable across epochs.
        return load_dataloader(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=self.batch_size,
            split="val",
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # Test follows the same deterministic loader settings as validation.
        return load_dataloader(
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
            batch_size=self.batch_size,
            split="test",
            num_workers=self.num_workers,
        )

# Load a pretrained MobileNetV3-Large and swap the classifier head.
def load_mobilenetV3_large(num_classes: int) -> nn.Module:
    weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = tv_models.mobilenet_v3_large(weights=weights)

    # Replace the ImageNet head with a Food-101-sized classifier.
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)
    return model

# Default optimizer/scheduler pair used by the notebook.
def define_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # Reduce the LR when validation loss stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )
    return optimizer, scheduler

# Call a factory/helper with only the kwargs it actually supports.
def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
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
        if name in parameters  # keep only kwargs the function accepts
    }

    # Surface missing required args early with a cleaner error message.
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

# Convert scheduler outputs into Lightning's expected config format.
def _normalize_scheduler_config(
    scheduler_config: Any,
    monitor: str,
) -> dict[str, Any]:
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

# Normalize different optimizer builder return styles into one format.
def _normalize_optimizer_config(
    optimizer_config: Any,
    monitor: str,
) -> Any:
    if isinstance(optimizer_config, Optimizer):
        return optimizer_config

    if isinstance(optimizer_config, tuple):
        # Support (optimizer,), (optimizer, scheduler), and
        # (optimizer, scheduler, monitor) return shapes.
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

# Reusable LightningModule for Food-101 experiments.
class FlexibleFood101Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 101,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        load_model_fn: Callable[..., nn.Module] | None = load_mobilenetV3_large,
        model: nn.Module | None = None,
        model_kwargs: dict[str, Any] | None = None,
        optimizer_scheduler_fn: Callable[..., Any] = define_optimizer_and_scheduler,
        optimizer_scheduler_kwargs: dict[str, Any] | None = None,
        loss_fn: nn.Module | None = None,
        metric: nn.Module | None = None,
        metric_factory: Callable[..., nn.Module] | None = None,
        optimizer_monitor: str = "val_loss",
    ) -> None:
        super().__init__()

        # Save only serializable hyperparameters into checkpoints.
        self.save_hyperparameters(
            ignore=[
                "load_model_fn",
                "model",
                "optimizer_scheduler_fn",
                "loss_fn",
                "metric",
                "metric_factory",
            ]
        )

        self.load_model_fn = load_model_fn
        self.optimizer_scheduler_fn = optimizer_scheduler_fn
        self.model_kwargs = dict(model_kwargs or {})
        self.optimizer_scheduler_kwargs = dict(optimizer_scheduler_kwargs or {})
        self.optimizer_monitor = optimizer_monitor

        # Prefer an already-built model when Stage 2 starts from Stage 1 weights.
        if model is not None:
            self.model = model  # e.g. reuse Stage 1 weights for Stage 2
        elif load_model_fn is not None:
            # Build a fresh model through the injected factory.
            self.model = _call_with_supported_kwargs(
                load_model_fn,
                num_classes=num_classes,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                **self.model_kwargs,
            )
        else:
            raise ValueError("You must provide either `model` or `load_model_fn`.")

        # Use standard classification defaults unless the caller overrides them.
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        if metric is not None:
            self.accuracy = metric
        elif metric_factory is not None:
            # Allow a custom metric factory that depends on num_classes.
            self.accuracy = _call_with_supported_kwargs(
                metric_factory, num_classes=num_classes
            )
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    # Forward directly into the wrapped torchvision model.
    def forward(self, x):
        return self.model(x)

    # Compute and log the epoch-level training loss.
    def training_step(self, batch, batch_idx=None):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Log only at epoch level to keep CSV output compact and easy to plot.
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # Compute and log validation loss and accuracy.
    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        # `val_loss` drives LR scheduling by default; `val_acc` drives reporting/checkpoints.
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

    # Delegate optimizer creation to the injected builder function.
    def configure_optimizers(self):
        # Pass multiple aliases (`model`, `module`, `lightning_module`) so custom
        # optimizer builders in notebooks can choose whichever name they expect.
        optimizer_config = _call_with_supported_kwargs(
            self.optimizer_scheduler_fn,
            model=self.model,
            module=self,
            lightning_module=self,
            num_classes=self.hparams.num_classes,
            learning_rate=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            **self.optimizer_scheduler_kwargs,
        )
        return _normalize_optimizer_config(optimizer_config, self.optimizer_monitor)

# Keep the notebook-facing constructor name stable.
def Create_flexible_Food101Classifier(**kwargs: Any) -> FlexibleFood101Classifier:
    # Preserve the original mixed-case helper name used throughout older notebook cells.
    return FlexibleFood101Classifier(**kwargs)


__all__ = [
    "Food101DataModule",
    "TransformBundle",
    "Create_flexible_Food101Classifier",
    "FlexibleFood101Classifier",
    "create_datasets",
    "create_food101_transforms",
    "define_optimizer_and_scheduler",
    "load_dataloader",
    "load_mobilenetV3_large",
]
