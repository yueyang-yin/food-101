"""Notebook helper functions for dataset inspection and visualization."""

from collections import Counter
import math
from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd
import torch


# Supported image file extensions used when scanning folder-based datasets.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# Common split names supported by the helper functions.
SPLIT_NAMES = ("train", "val", "valid", "validation", "test")


# Resolve the actual dataset root. This supports both:
# 1. A dataset root that directly contains `meta/` or split folders.
# 2. A parent directory such as `../data` that contains a child dataset folder.
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


# Infer both dataset root and requested split from user input.
# This intentionally supports notebook calls like `data_dir + "train"` in addition
# to standard inputs such as `../data`, `../data/train`, or `../data/food-101`.
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


# Count images per class from metadata files like Food-101 `meta/train.txt`.
def _count_from_meta_split(dataset_root, split_name):
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    counts = Counter(line.split("/")[0] for line in lines)
    return dict(sorted(counts.items()))


# Count images per class from folder-based splits like `train/class_name/*.jpg`.
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


# Resolve a metadata entry like `apple_pie/1005649` into a real image path.
def _resolve_meta_image_path(images_root, relative_stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = images_root / f"{relative_stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image file not found for metadata entry: {relative_stem}")


# Build a class-to-image-paths mapping from metadata-based datasets.
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


# Build a class-to-image-paths mapping from folder-based datasets.
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


# Print per-class image counts for every detected dataset split.
def display_dataset_count(data_dir):
    dataset_root = _resolve_dataset_root(data_dir)
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


# Display a 3x2-style random image grid: three random classes and two images
# per class by default, matching the layout used in the notebook screenshot.
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

    # Use Matplotlib's subplot grid to create a clean image gallery layout.
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


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _resolve_class_names(dataset):
    base_dataset = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    raise ValueError("Cannot resolve class names from the provided dataset.")


def _denormalize_image(image_tensor, mean, std):
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std_tensor + mean_tensor).clamp(0, 1)


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

    was_training = model.training
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    rows = math.ceil(num_images / max_cols)
    cols = min(num_images, max_cols)
    if figsize is None:
        figsize = (cols * 4.5, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    try:
        with torch.inference_mode():
            for axis, sample_idx in zip(flat_axes, selected_indices):
                image_tensor, true_label = val_dataset[sample_idx]
                logits = model(image_tensor.unsqueeze(0).to(device))
                pred_label = int(logits.argmax(dim=1).item())

                display_image = _denormalize_image(
                    image_tensor.detach().cpu(),
                    denormalize_mean,
                    denormalize_std,
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
    plt.show()
    return fig, axes


# Save a Matplotlib figure into the project's artifacts folder.
# If no figure is provided, the current active figure is used.
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

    output_path = Path(filename)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")

    if fig is None:
        fig = plt.gcf()

    save_path = output_dir / output_path.name
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
    )
    return save_path


# Plot training curves from a Lightning CSVLogger metrics dataframe or csv file.
# Validation-only rows appended after training are excluded from the epoch curves,
# while the best training epoch is highlighted explicitly.
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

    history = (
        df.groupby("epoch", as_index=False)
        .agg({
            "train_loss": "max",
            "val_loss": "max",
            "val_acc": "max",
        })
        .sort_values("epoch")
    )
    history = history[history["train_loss"].notna()].copy()

    if history.empty:
        raise ValueError("No training epochs with `train_loss` were found in the provided metrics.")

    epoch_col = "epoch_num" if one_based_epoch else "epoch"
    history[epoch_col] = history["epoch"] + 1 if one_based_epoch else history["epoch"]

    best_point = history.loc[history["val_loss"].idxmin()]
    best_label = f"Best Checkpoint (ckpt epoch={int(best_point['epoch']):02d})"
    epoch_label = "Epoch (1-based)" if one_based_epoch else "Epoch (0-based)"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

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
    plt.show()
    return fig, axes
