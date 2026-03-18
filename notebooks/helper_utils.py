# Helper utilities used by the Food-101 notebook.

from collections import Counter
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Accepted image extensions when scanning folder-based datasets.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Supported split names across different dataset layouts.
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")

# Remember the most recently rendered figure so notebook save cells can still
# work even when `plt.show()` has already run in a previous cell.
_LAST_RENDERED_FIGURE = None

# Resolve the actual dataset root from either the dataset folder itself
# or a parent directory that contains the dataset as a child folder.
def _resolve_dataset_root(data_dir):
    data_dir = Path(data_dir)

    if (data_dir / "meta" / "train.txt").exists() or (data_dir / "train").is_dir():
        return data_dir

    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta" / "train.txt").exists() or (child / "train").is_dir():
            return child

    raise FileNotFoundError(f"No supported dataset structure found under: {data_dir}")

# Accept flexible notebook inputs like `../data`, `../data/train`,
# or even string concatenations such as `data_dir + "train"`.
def _resolve_dataset_root_and_split(data_dir, default_split="train"):
    raw_path = str(data_dir).rstrip("/\\")
    path_obj = Path(raw_path)

    if path_obj.exists():
        if path_obj.is_dir() and path_obj.name in SPLIT_NAMES:
            return path_obj.parent, path_obj.name
        return _resolve_dataset_root(path_obj), default_split

    for split_name in SPLIT_NAMES:
        if raw_path.endswith(split_name):
            base_path = raw_path[: -len(split_name)].rstrip("/\\")
            if base_path and Path(base_path).exists():
                return _resolve_dataset_root(base_path), split_name

    raise FileNotFoundError(f"Cannot resolve dataset path or split from: {data_dir}")

# Count class frequency from metadata files such as Food-101 `meta/train.txt`.
def _count_from_meta_split(dataset_root, split_name):
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    counts = Counter(line.split("/")[0] for line in lines)
    return dict(sorted(counts.items()))

# Count class frequency from a standard split/class/image directory tree.
def _count_from_folder_split(dataset_root, split_name):
    split_dir = dataset_root / split_name
    if not split_dir.is_dir():
        return None

    counts = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        counts[class_dir.name] = sum(
            1
            for file_path in class_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts

# Turn a metadata stem like `apple_pie/1005649` into an actual image path.
def _resolve_meta_image_path(images_root, relative_stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = images_root / f"{relative_stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image file not found for metadata entry: {relative_stem}")

# Build `{class_name: [image_paths...]}` from metadata-defined splits.
def _image_map_from_meta_split(dataset_root, split_name):
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    image_map = {}
    images_root = dataset_root / "images"
    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    for relative_stem in lines:
        class_name = relative_stem.split("/")[0]
        image_map.setdefault(class_name, []).append(
            _resolve_meta_image_path(images_root, relative_stem)
        )

    return dict(sorted(image_map.items()))

# Build `{class_name: [image_paths...]}` from folder-defined splits.
def _image_map_from_folder_split(dataset_root, split_name):
    split_dir = dataset_root / split_name
    if not split_dir.is_dir():
        return None

    image_map = {}
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        image_paths = sorted(
            file_path
            for file_path in class_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_map[class_dir.name] = image_paths

    return image_map

# Print image counts for each available split so the notebook can quickly
# verify class balance and total dataset size.
def display_dataset_count(data_dir):
    dataset_root = _resolve_dataset_root(data_dir)
    # Check several common split names because different datasets organize
    # validation data as `val`, `valid`, or `validation`.
    split_candidates = [
        ("train", "Train Set"),
        ("val", "Val Set"),
        ("valid", "Valid Set"),
        ("validation", "Validation Set"),
        ("test", "Test Set"),
    ]

    found_any_split = False

    for split_key, split_title in split_candidates:
        counts = _count_from_meta_split(dataset_root, split_key)
        if counts is None:
            counts = _count_from_folder_split(dataset_root, split_key)
        if counts is None:
            continue

        found_any_split = True
        total = sum(counts.values())
        max_name_len = max(len(name) for name in counts)

        print(f"--- {split_title} ---")
        for class_name, num_images in counts.items():
            print(f"- {class_name:<{max_name_len}} : {num_images} images")
        print("-" * (max_name_len + 20))
        print(f"Total: {total} images\n")

    if not found_any_split:
        raise ValueError(f"No train/val/test style splits found in: {dataset_root}")

# Show a small random gallery from the training split for visual sanity checks.
def display_random_images(data_dir, num_classes=3, images_per_class=2, random_seed=None):
    dataset_root, split_name = _resolve_dataset_root_and_split(data_dir, default_split="train")
    image_map = _image_map_from_meta_split(dataset_root, split_name)
    if image_map is None:
        image_map = _image_map_from_folder_split(dataset_root, split_name)
    if image_map is None:
        raise ValueError(f"No readable image split found for '{split_name}' in: {dataset_root}")

    eligible_classes = [
        class_name
        for class_name, image_paths in image_map.items()
        if len(image_paths) >= images_per_class
    ]
    if len(eligible_classes) < num_classes:
        raise ValueError(
            f"Need at least {num_classes} classes with {images_per_class} images each, "
            f"but only found {len(eligible_classes)} eligible classes."
        )

    rng = random.Random(random_seed)
    selected_classes = rng.sample(eligible_classes, num_classes)

    # Keep the layout deterministic and notebook-friendly.
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=images_per_class,
        figsize=(images_per_class * 5, num_classes * 5),
        layout="constrained",
        squeeze=False,
    )

    for row_index, class_name in enumerate(selected_classes):
        selected_images = rng.sample(image_map[class_name], images_per_class)
        for col_index, image_path in enumerate(selected_images):
            axis = axes[row_index, col_index]
            axis.imshow(plt.imread(image_path))
            axis.set_title(class_name, fontsize=16)
            axis.axis("off")

    plt.show()

# Peel off wrappers like `Subset` until the underlying dataset is reached.
def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset

# Read class names from the underlying torchvision-style dataset.
def _resolve_class_names(dataset):
    base_dataset = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    raise ValueError("Cannot resolve class names from the provided dataset.")

# Undo normalization before plotting tensors as RGB images.
def _denormalize_image(image_tensor, mean, std):
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std_tensor + mean_tensor).clamp(0, 1)

# Cache the latest figure produced by a plotting helper.
def _remember_figure(fig):
    global _LAST_RENDERED_FIGURE
    _LAST_RENDERED_FIGURE = fig
    return fig

# Resolve which figure should actually be saved.
def _resolve_figure_for_saving(fig=None):
    if fig is not None:
        return fig

    global _LAST_RENDERED_FIGURE
    if _LAST_RENDERED_FIGURE is not None:
        try:
            if plt.fignum_exists(_LAST_RENDERED_FIGURE.number):
                return _LAST_RENDERED_FIGURE
        except Exception:
            pass

    current_fig = plt.gcf()
    if current_fig.axes:
        return current_fig

    raise ValueError(
        "No plotted Matplotlib figure is available to save. "
        "Pass `fig=...` explicitly or rerun the plotting cell first."
    )

# Sample validation images, run inference, and show predicted vs true labels.
def show_random_validation_predictions(
    model,
    data_module,
    num_images=8,
    random_seed=None,
    class_names=None,
    denormalize_mean=(0.485, 0.456, 0.406),
    denormalize_std=(0.229, 0.224, 0.225),
    max_cols=4,
    figsize=None,
):
    if getattr(data_module, "val_dataset", None) is None:
        # Lazily initialize the validation split if the datamodule has not been set up yet.
        data_module.setup(stage="fit")

    val_dataset = data_module.val_dataset
    if val_dataset is None:
        raise ValueError("`data_module.val_dataset` is not available.")

    if class_names is None:
        class_names = _resolve_class_names(val_dataset)

    dataset_size = len(val_dataset)
    if dataset_size == 0:
        raise ValueError("Validation dataset is empty.")

    num_images = min(num_images, dataset_size)
    rng = random.Random(random_seed)
    selected_indices = rng.sample(range(dataset_size), num_images)

    was_training = model.training  # restore the original mode afterwards
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    rows = math.ceil(num_images / max_cols)
    cols = min(num_images, max_cols)
    if figsize is None:
        figsize = (cols * 4.5, rows * 4.5)

    # Always create a 2D axes array so the loop logic stays simple.
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    try:
        with torch.inference_mode():
            for axis, sample_idx in zip(flat_axes, selected_indices):
                # Pull one sample, run a forward pass, and keep only the top class.
                image_tensor, true_label = val_dataset[sample_idx]
                logits = model(image_tensor.unsqueeze(0).to(device))
                pred_label = int(logits.argmax(dim=1).item())

                display_image = _denormalize_image(
                    image_tensor.detach().cpu(),
                    denormalize_mean,
                    denormalize_std,  # undo ImageNet normalization for plotting
                ).permute(1, 2, 0)

                axis.imshow(display_image)
                axis.axis("off")

                is_correct = pred_label == int(true_label)
                title_color = "green" if is_correct else "red"
                axis.set_title(
                    f"Pred: {class_names[pred_label]}\nTrue: {class_names[int(true_label)]}",
                    fontsize=11,
                    color=title_color,
                )

            for axis in flat_axes[num_images:]:
                axis.axis("off")
    finally:
        if was_training:
            model.train()

    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return fig, axes

# Save a Matplotlib figure into `artifacts/figures` by default.
def save_figure_to_artifacts(
    filename,
    fig=None,
    artifact_subdir="figures",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
):
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "artifacts" / artifact_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow callers to pass either `figure_name` or `figure_name.png`.
    output_path = Path(filename)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")

    fig = _resolve_figure_for_saving(fig)

    save_path = output_dir / output_path.name
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
    )
    return save_path

# Collapse Lightning's per-step/per-epoch CSV rows into one clean row per epoch.
def _build_epoch_history(df, one_based_epoch=True):
    history = (
        df.groupby("epoch", as_index=False)
        .agg({
            "train_loss": "max",
            "val_loss": "max",
            "val_acc": "max",
        })
        .sort_values("epoch")
    )
    # Keep only rows that correspond to completed training epochs.
    history = history[history["train_loss"].notna()].copy()

    if history.empty:
        raise ValueError("No training epochs with `train_loss` were found in the provided metrics.")

    epoch_col = "epoch_num" if one_based_epoch else "epoch"
    history[epoch_col] = history["epoch"] + 1 if one_based_epoch else history["epoch"]
    return history, epoch_col

# Plot training loss, validation loss, and validation accuracy from Lightning logs.
def plot_training_curves(
    df=None,
    metrics_csv=None,
    title="Food-101 Baseline Training Curves",
    figsize=(14, 5),
    one_based_epoch=True,
    mark_best=True,
):
    if df is None:
        if metrics_csv is None:
            raise ValueError("Provide either `df` or `metrics_csv`.")
        df = pd.read_csv(metrics_csv)
    else:
        df = df.copy()

    history, epoch_col = _build_epoch_history(df, one_based_epoch=one_based_epoch)

    # Use best validation accuracy as the notebook's main model-selection signal.
    best_point = history.loc[history["val_acc"].idxmax()]
    best_label = f"Best Val Acc Checkpoint (ckpt epoch={int(best_point['epoch']):02d})"
    epoch_label = "Epoch (1-based)" if one_based_epoch else "Epoch (0-based)"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left panel: train/val loss share the same scale and are easy to compare together.
    axes[0].plot(
        history[epoch_col],
        history["train_loss"],
        marker="o",
        linewidth=2,
        label="Train Loss",
    )
    axes[0].plot(
        history[epoch_col],
        history["val_loss"],
        marker="s",
        linewidth=2,
        label="Val Loss",
    )
    if mark_best:
        axes[0].scatter(
            best_point[epoch_col],
            best_point["val_loss"],
            marker="*",
            s=160,
            zorder=5,
            label=best_label,
        )
    axes[0].set_title("Training and Validation Loss", fontsize=13)
    axes[0].set_xlabel(epoch_label)
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(history[epoch_col])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Right panel: validation accuracy gets its own axis so small gains stay visible.
    axes[1].plot(
        history[epoch_col],
        history["val_acc"],
        marker="o",
        linewidth=2,
        label="Val Accuracy",
    )
    if mark_best:
        axes[1].scatter(
            best_point[epoch_col],
            best_point["val_acc"],
            marker="*",
            s=160,
            zorder=5,
            label=best_label,
        )
    axes[1].set_title("Validation Accuracy", fontsize=13)
    axes[1].set_xlabel(epoch_label)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(history[epoch_col])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return fig, axes

# Compare Stage 1 and Stage 2 logs and return both plots and summary tables.
def compare_stage_training_runs(
    stage1_df,
    stage2_df,
    stage1_name="Stage 1",
    stage2_name="Stage 2",
    one_based_epoch=True,
    print_summary=False,
):
    stage1_history, epoch_col = _build_epoch_history(stage1_df.copy(), one_based_epoch=one_based_epoch)
    stage2_history, _ = _build_epoch_history(stage2_df.copy(), one_based_epoch=one_based_epoch)
    epoch_label = "Epoch (1-based)" if one_based_epoch else "Epoch (0-based)"

    # Track both the best validation checkpoint and the final epoch for each stage.
    stage1_best = stage1_history.loc[stage1_history["val_acc"].idxmax()]
    stage2_best = stage2_history.loc[stage2_history["val_acc"].idxmax()]
    stage1_final = stage1_history.iloc[-1]
    stage2_final = stage2_history.iloc[-1]

    # Keep the summary compact so the notebook can render it directly.
    summary_df = pd.DataFrame(
        [
            {
                "stage": stage1_name,
                "best_epoch": int(stage1_best[epoch_col]),
                "best_val_acc": float(stage1_best["val_acc"]),
                "best_val_loss": float(stage1_best["val_loss"]),
                "train_loss_at_best": float(stage1_best["train_loss"]),
                "final_epoch": int(stage1_final[epoch_col]),
                "final_train_loss": float(stage1_final["train_loss"]),
                "final_val_loss": float(stage1_final["val_loss"]),
                "final_val_acc": float(stage1_final["val_acc"]),
            },
            {
                "stage": stage2_name,
                "best_epoch": int(stage2_best[epoch_col]),
                "best_val_acc": float(stage2_best["val_acc"]),
                "best_val_loss": float(stage2_best["val_loss"]),
                "train_loss_at_best": float(stage2_best["train_loss"]),
                "final_epoch": int(stage2_final[epoch_col]),
                "final_train_loss": float(stage2_final["train_loss"]),
                "final_val_loss": float(stage2_final["val_loss"]),
                "final_val_acc": float(stage2_final["val_acc"]),
            },
            {
                "stage": f"{stage2_name} - {stage1_name}",
                "best_epoch": int(stage2_best[epoch_col] - stage1_best[epoch_col]),
                "best_val_acc": float(stage2_best["val_acc"] - stage1_best["val_acc"]),
                "best_val_loss": float(stage2_best["val_loss"] - stage1_best["val_loss"]),
                "train_loss_at_best": float(stage2_best["train_loss"] - stage1_best["train_loss"]),
                "final_epoch": int(stage2_final[epoch_col] - stage1_final[epoch_col]),
                "final_train_loss": float(stage2_final["train_loss"] - stage1_final["train_loss"]),
                "final_val_loss": float(stage2_final["val_loss"] - stage1_final["val_loss"]),
                "final_val_acc": float(stage2_final["val_acc"] - stage1_final["val_acc"]),
            },
        ]
    )

    max_epoch = int(max(stage1_history[epoch_col].max(), stage2_history[epoch_col].max()))
    xticks = list(range(1, max_epoch + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # Panel 1: train loss progression.
    axes[0].plot(
        stage1_history[epoch_col],
        stage1_history["train_loss"],
        marker="o",
        linewidth=2,
        label=f"{stage1_name} Train Loss ({int(stage1_final[epoch_col])} epochs)",
    )
    axes[0].plot(
        stage2_history[epoch_col],
        stage2_history["train_loss"],
        marker="s",
        linewidth=2,
        label=f"{stage2_name} Train Loss ({int(stage2_final[epoch_col])} epochs)",
    )
    axes[0].scatter(
        stage1_final[epoch_col],
        stage1_final["train_loss"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage1_name} End",
    )
    axes[0].scatter(
        stage2_final[epoch_col],
        stage2_final["train_loss"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage2_name} End",
    )
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel(epoch_label)
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(xticks)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Panel 2: validation loss with best-epoch and final-epoch markers.
    axes[1].plot(
        stage1_history[epoch_col],
        stage1_history["val_loss"],
        marker="o",
        linewidth=2,
        label=f"{stage1_name} Val Loss",
    )
    axes[1].plot(
        stage2_history[epoch_col],
        stage2_history["val_loss"],
        marker="s",
        linewidth=2,
        label=f"{stage2_name} Val Loss",
    )
    axes[1].scatter(
        stage1_best[epoch_col],
        stage1_best["val_loss"],
        marker="*",
        s=160,
        zorder=5,
        label=f"{stage1_name} Best Val Acc",
    )
    axes[1].scatter(
        stage2_best[epoch_col],
        stage2_best["val_loss"],
        marker="*",
        s=160,
        zorder=5,
        label=f"{stage2_name} Best Val Acc",
    )
    axes[1].scatter(
        stage1_final[epoch_col],
        stage1_final["val_loss"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage1_name} End",
    )
    axes[1].scatter(
        stage2_final[epoch_col],
        stage2_final["val_loss"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage2_name} End",
    )
    axes[1].set_title("Val Loss")
    axes[1].set_xlabel(epoch_label)
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(xticks)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    # Panel 3: validation accuracy with best-epoch and final-epoch markers.
    axes[2].plot(
        stage1_history[epoch_col],
        stage1_history["val_acc"],
        marker="o",
        linewidth=2,
        label=f"{stage1_name} Val Acc",
    )
    axes[2].plot(
        stage2_history[epoch_col],
        stage2_history["val_acc"],
        marker="s",
        linewidth=2,
        label=f"{stage2_name} Val Acc",
    )
    axes[2].scatter(
        stage1_best[epoch_col],
        stage1_best["val_acc"],
        marker="*",
        s=160,
        zorder=5,
        label=f"{stage1_name} Best",
    )
    axes[2].scatter(
        stage2_best[epoch_col],
        stage2_best["val_acc"],
        marker="*",
        s=160,
        zorder=5,
        label=f"{stage2_name} Best",
    )
    axes[2].scatter(
        stage1_final[epoch_col],
        stage1_final["val_acc"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage1_name} End",
    )
    axes[2].scatter(
        stage2_final[epoch_col],
        stage2_final["val_acc"],
        marker="D",
        s=90,
        zorder=5,
        label=f"{stage2_name} End",
    )
    axes[2].set_title("Val Accuracy")
    axes[2].set_xlabel(epoch_label)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_xticks(xticks)
    axes[2].grid(True, linestyle="--", alpha=0.4)
    axes[2].legend()

    # Panel 4: compact bar chart of the metrics most likely to be cited later.
    comparison_plot_df = pd.DataFrame(
        {
            "metric": ["Best Val Acc", "Final Val Acc", "Best Val Loss", "Final Val Loss"],
            stage1_name: [
                stage1_best["val_acc"],
                stage1_final["val_acc"],
                stage1_best["val_loss"],
                stage1_final["val_loss"],
            ],
            stage2_name: [
                stage2_best["val_acc"],
                stage2_final["val_acc"],
                stage2_best["val_loss"],
                stage2_final["val_loss"],
            ],
        }
    )
    comparison_plot_df = comparison_plot_df.set_index("metric")
    comparison_plot_df.plot(kind="bar", ax=axes[3], width=0.75)
    axes[3].set_title("Best / Final Metric Summary")
    axes[3].set_xlabel("")
    axes[3].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[3].legend()

    plt.suptitle(f"{stage1_name} vs {stage2_name}", fontsize=15)
    plt.tight_layout()
    _remember_figure(fig)
    plt.show()

    if print_summary:
        print("=== Stage Training Summary ===")
        print(summary_df.to_string(index=False))

    return {
        "summary": summary_df,
        "stage1_history": stage1_history,
        "stage2_history": stage2_history,
        "fig": fig,
        "axes": axes,
    }

# Build a `name -> parameter` mapping for comparison helpers.
def _collect_named_parameters(model):
    if not hasattr(model, "named_parameters"):
        raise TypeError("`model` must be a PyTorch/Lightning module with `named_parameters()`.")

    return {name: parameter for name, parameter in model.named_parameters()}

# Summarize parameter counts for one model.
def _build_model_summary(model, model_name):
    parameter_map = _collect_named_parameters(model)
    # Count both raw parameter volume and how much of it is trainable.
    total_params = sum(parameter.numel() for parameter in parameter_map.values())
    trainable_params = sum(
        parameter.numel() for parameter in parameter_map.values() if parameter.requires_grad
    )
    frozen_params = total_params - trainable_params

    return {
        "model_name": model_name,
        "module_class": model.__class__.__name__,
        "parameter_tensors": len(parameter_map),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_ratio": trainable_params / total_params if total_params else 0.0,
    }

# Compare trainable flags, weights, and shared storage between two models.
def compare_stage_models(
    stage1_model,
    stage2_model,
    stage1_name="Stage 1",
    stage2_name="Stage 2",
    only_differences=True,
    compare_weights=True,
    compare_requires_grad=True,
    max_changed_parameters=20,
    atol=0.0,
    rtol=0.0,
    verbose=True,
):
    stage1_params = _collect_named_parameters(stage1_model)
    stage2_params = _collect_named_parameters(stage2_model)

    # Split parameter names into shared vs stage-specific groups first.
    common_names = sorted(set(stage1_params) & set(stage2_params))
    only_stage1 = sorted(set(stage1_params) - set(stage2_params))
    only_stage2 = sorted(set(stage2_params) - set(stage1_params))

    # Start with two model summaries, then append one comparison row.
    summary_rows = [
        _build_model_summary(stage1_model, stage1_name),
        _build_model_summary(stage2_model, stage2_name),
    ]
    summary_rows.append(
        {
            "model_name": "Comparison",
            "module_class": f"{len(common_names)} shared parameter names",
            "parameter_tensors": len(common_names),
            "total_params": len(only_stage1),
            "trainable_params": len(only_stage2),
            "frozen_params": 0,
            "trainable_ratio": None,
        }
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.rename(
        columns={
            "total_params": "only_in_stage1",
            "trainable_params": "only_in_stage2",
            "frozen_params": "unused",
        }
    )
    summary_df.loc[summary_df["model_name"] != "Comparison", "only_in_stage1"] = [
        _build_model_summary(stage1_model, stage1_name)["total_params"],
        _build_model_summary(stage2_model, stage2_name)["total_params"],
    ]
    summary_df.loc[summary_df["model_name"] != "Comparison", "only_in_stage2"] = [
        _build_model_summary(stage1_model, stage1_name)["trainable_params"],
        _build_model_summary(stage2_model, stage2_name)["trainable_params"],
    ]
    summary_df.loc[summary_df["model_name"] != "Comparison", "unused"] = [
        _build_model_summary(stage1_model, stage1_name)["frozen_params"],
        _build_model_summary(stage2_model, stage2_name)["frozen_params"],
    ]
    summary_df = summary_df.rename(
        columns={
            "only_in_stage1": "total_params",
            "only_in_stage2": "trainable_params",
            "unused": "frozen_params",
        }
    )
    summary_df["stage1_only_parameter_names"] = [None, None, only_stage1]
    summary_df["stage2_only_parameter_names"] = [None, None, only_stage2]

    requires_grad_rows = []
    parameter_change_rows = []
    shared_parameter_rows = []

    for name in common_names:
        stage1_parameter = stage1_params[name]
        stage2_parameter = stage2_params[name]

        # `same_storage` catches accidental sharing between Stage 1 and Stage 2 models.
        same_shape = tuple(stage1_parameter.shape) == tuple(stage2_parameter.shape)
        same_storage = (
            same_shape
            and stage1_parameter.detach().data_ptr() == stage2_parameter.detach().data_ptr()
        )

        if compare_requires_grad:
            requires_grad_rows.append(
                {
                    "parameter_name": name,
                    "shape": tuple(stage1_parameter.shape),
                    "numel": stage1_parameter.numel(),
                    f"{stage1_name}_requires_grad": stage1_parameter.requires_grad,
                    f"{stage2_name}_requires_grad": stage2_parameter.requires_grad,
                    "requires_grad_changed": (
                        stage1_parameter.requires_grad != stage2_parameter.requires_grad
                    ),
                    "same_storage": same_storage,
                }
            )

        if same_storage:
            shared_parameter_rows.append(
                {
                    "parameter_name": name,
                    "shape": tuple(stage1_parameter.shape),
                    "numel": stage1_parameter.numel(),
                    "same_storage": True,
                }
            )

        if compare_weights:
            if not same_shape:
                parameter_change_rows.append(
                    {
                        "parameter_name": name,
                        "shape": f"{tuple(stage1_parameter.shape)} vs {tuple(stage2_parameter.shape)}",
                        "numel": None,
                        "same_storage": same_storage,
                        "allclose": False,
                        "max_abs_diff": None,
                        "mean_abs_diff": None,
                        "l2_diff": None,
                        f"{stage1_name}_requires_grad": stage1_parameter.requires_grad,
                        f"{stage2_name}_requires_grad": stage2_parameter.requires_grad,
                    }
                )
                continue

            # Compare on CPU float tensors so this works regardless of training device.
            stage1_tensor = stage1_parameter.detach().float().cpu()
            stage2_tensor = stage2_parameter.detach().float().cpu()
            diff = (stage1_tensor - stage2_tensor).abs()

            parameter_change_rows.append(
                {
                    "parameter_name": name,
                    "shape": tuple(stage1_parameter.shape),
                    "numel": stage1_parameter.numel(),
                    "same_storage": same_storage,
                    "allclose": torch.allclose(
                        stage1_tensor,
                        stage2_tensor,
                        atol=atol,
                        rtol=rtol,
                    ),
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                    "l2_diff": float(torch.linalg.vector_norm((stage1_tensor - stage2_tensor).reshape(-1)).item()),
                    f"{stage1_name}_requires_grad": stage1_parameter.requires_grad,
                    f"{stage2_name}_requires_grad": stage2_parameter.requires_grad,
                }
            )

    # Sort the comparison tables so the most interesting differences appear first.
    requires_grad_df = pd.DataFrame(requires_grad_rows)
    if not requires_grad_df.empty:
        requires_grad_df = requires_grad_df.sort_values(
            by=["requires_grad_changed", "numel", "parameter_name"],
            ascending=[False, False, True],
        )
        if only_differences:
            requires_grad_df = requires_grad_df[requires_grad_df["requires_grad_changed"]].copy()

    parameter_changes_df = pd.DataFrame(parameter_change_rows)
    if not parameter_changes_df.empty:
        parameter_changes_df = parameter_changes_df.sort_values(
            by=["same_storage", "l2_diff", "max_abs_diff", "parameter_name"],
            ascending=[True, False, False, True],
            na_position="last",
        )
        if only_differences:
            parameter_changes_df = parameter_changes_df[
                (~parameter_changes_df["allclose"]) | (parameter_changes_df["same_storage"])
            ].copy()
        if max_changed_parameters is not None:
            parameter_changes_df = parameter_changes_df.head(max_changed_parameters).copy()

    shared_parameters_df = pd.DataFrame(shared_parameter_rows)
    if not shared_parameters_df.empty:
        shared_parameters_df = shared_parameters_df.sort_values(
            by=["numel", "parameter_name"],
            ascending=[False, True],
        )

    # Return DataFrames so notebook cells can inspect only the table they need.
    results = {
        "summary": summary_df,
        "requires_grad": requires_grad_df.reset_index(drop=True),
        "parameter_changes": parameter_changes_df.reset_index(drop=True),
        "shared_parameters": shared_parameters_df.reset_index(drop=True),
    }

    if verbose:
        print("=== Model Comparison Summary ===")
        print(summary_df[["model_name", "module_class", "parameter_tensors", "total_params", "trainable_params", "frozen_params", "trainable_ratio"]].to_string(index=False))
        print(f"\nShared parameter names: {len(common_names)}")
        print(f"Stage 1-only parameter names: {len(only_stage1)}")
        print(f"Stage 2-only parameter names: {len(only_stage2)}")
        print(f"Parameters with changed requires_grad: {len(requires_grad_df)}")
        print(f"Reported parameter tensor changes: {len(parameter_changes_df)}")
        print(f"Parameters sharing the same storage: {len(shared_parameters_df)}")

    return results
