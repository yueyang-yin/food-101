"""Helper utilities used by the Food-101 notebooks.

The file is organized to match notebook usage order:
1. Shared helpers used by both notebooks.
2. Public helpers consumed by `01_baseline_and_transfer_learning.ipynb`.
3. Public helpers consumed by `02_analysis_and_optimization.ipynb`.
4. Lower-level private implementations and MLflow internals.
"""

from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from IPython.display import HTML, display
except Exception:
    # Notebook helpers can still run in plain Python scripts, so keep a text-only
    # fallback instead of failing on the optional rich-display dependency.
    HTML = None
    display = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
# Cache the most recently shown figure so notebook users can call the save helper
# in a later cell without manually threading `fig` around.
_LAST_RENDERED_FIGURE = None


# ---------------------------------------------------------------------------
# Shared path, serialization, and runtime helpers
# ---------------------------------------------------------------------------

def _notebook_dir():
    """Return the directory that contains the project notebooks and helpers."""
    return Path(__file__).resolve().parent


def _project_root():
    """Return the repository root used by the notebook helpers."""
    return _notebook_dir().parent


def _experiment_checkpoint_dir():
    """Return the folder where notebook experiments cache their results."""
    return _project_root() / "artifacts" / "checkpoint_experiments"


def _display_relative_path(path, start=None):
    """Return a stable relative path string for notebook-facing status messages."""
    # Relative paths keep notebook logs shorter and reproducible across machines
    # whose absolute workspace roots differ.
    start = Path(start or _notebook_dir())
    return Path(os.path.relpath(path, start=start)).as_posix()


def _json_safe_value(value):
    """Convert notebook result values into JSON-serializable primitives."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be serialized as experiment results.")
        # Convert 0-d / scalar tensors into plain Python numbers before the JSON
        # dump so experiment caches stay human-readable and portable.
        value = value.item()

    if isinstance(value, np.generic):
        # NumPy scalar dtypes (for example `np.float32`) are not serialized by
        # the standard library JSON encoder unless we unwrap them first.
        value = value.item()

    return value


def _restore_case_key(case):
    """Normalize JSON-loaded case keys back to the original case type."""
    if isinstance(case, torch.Tensor):
        if case.numel() != 1:
            raise ValueError("Experiment cases must be scalar values.")
        return case.item()

    if isinstance(case, np.generic):
        return case.item()

    return case


def _restore_cached_experiment_results(payload, cases):
    """Map cached JSON keys back to the same key types passed in via `cases`."""
    if not isinstance(payload, dict):
        raise ValueError("Cached experiment payload must be a JSON object.")

    restored = {}
    # Normalize the requested cases once up front so we can match them against
    # JSON object keys, which are always stored as strings on disk.
    normalized_cases = [_restore_case_key(case) for case in cases]
    for case in normalized_cases:
        case_key = str(case)
        if case_key in payload:
            restored[case] = _json_safe_value(payload[case_key])

    # Preserve any extra keys the caller did not explicitly request so cached
    # results remain inspectable if the case list changes later.
    for key, value in payload.items():
        if key not in {str(case) for case in normalized_cases}:
            restored[key] = _json_safe_value(value)

    return restored


def _contains_tensor(item):
    """Return whether one nested sample structure contains at least one tensor."""
    if torch.is_tensor(item):
        return True
    if isinstance(item, dict):
        # Recursively inspect mapping values because many datasets return
        # dictionaries like `{"image": tensor, "label": int}`.
        return any(_contains_tensor(value) for value in item.values())
    if isinstance(item, (list, tuple)):
        # Lists/tuples cover the common `(inputs, targets)` dataset pattern.
        return any(_contains_tensor(value) for value in item)
    return False


def _prepare_loader_for_iteration(loader):
    """Validate that the DataLoader yields tensors produced by a transform pipeline."""
    dataset = getattr(loader, "dataset", None)
    # Empty loaders are allowed here; the downstream timing helpers handle the
    # "no batches produced" case separately with task-specific errors.
    if dataset is None or len(dataset) == 0:
        return loader

    sample = dataset[0]
    if not _contains_tensor(sample):
        raise TypeError(
            "The benchmark helpers expect the dataset to return tensors. "
            "Pass the same transform pipeline used in training so the DataLoader "
            "yields tensor batches instead of raw PIL images."
        )
    return loader


def _move_batch_to_device(batch, device):
    """Recursively move tensor batches to the requested device for fair timing."""
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        # Preserve the original container shape so downstream code sees the same
        # batch structure after the device transfer.
        return {key: _move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(value, device) for value in batch)
    return batch


def _synchronize_device(device):
    """Wait for asynchronous accelerator work to finish before stopping a timer."""
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        # `torch.mps.synchronize()` is guarded because some PyTorch builds expose
        # MPS availability checks without the synchronization helper.
        torch.mps.synchronize()


def _iter_epoch_progress(num_epochs):
    """Yield epoch indices with a tqdm progress bar when tqdm is available."""
    if tqdm is None:
        # Fall back to a plain generator so the timing helpers still work in
        # minimal Python environments without notebook-friendly progress bars.
        for epoch_idx in range(1, num_epochs + 1):
            yield epoch_idx, None
        return

    progress_bar = tqdm(range(1, num_epochs + 1), desc="Overall Progress", total=num_epochs)
    try:
        for epoch_idx in progress_bar:
            yield epoch_idx, progress_bar
    finally:
        progress_bar.close()


def _write_progress_line(progress_bar, message):
    """Print epoch timing lines without corrupting the tqdm bar."""
    if progress_bar is not None:
        # `progress_bar.write(...)` prints above the bar without overwriting it.
        progress_bar.write(message)
    else:
        print(message)


# ---------------------------------------------------------------------------
# Notebook 01 helpers: dataset inspection
# ---------------------------------------------------------------------------

def _resolve_dataset_root(data_dir):
    """Return the Food-101 dataset root from either the root itself or its parent."""
    data_dir = Path(data_dir)

    # Support callers passing either the exact Food-101 root or a parent folder
    # that contains one extracted dataset directory.
    if (data_dir / "meta" / "train.txt").exists():
        return data_dir

    for child in sorted(data_dir.iterdir()):
        # Sorting makes the search deterministic if the parent directory happens
        # to contain multiple candidate subfolders.
        if child.is_dir() and (child / "meta" / "train.txt").exists():
            return child

    raise FileNotFoundError(f"No Food-101 dataset root found under: {data_dir}")


def _count_from_meta_split(dataset_root, split_name):
    """Count images per class from a metadata split file such as `train.txt`."""
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    # Food-101 metadata lines look like `class_name/image_id`, so counting the
    # first path segment gives us the number of samples per class.
    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    counts = Counter(line.split("/")[0] for line in lines)
    return dict(sorted(counts.items()))


def _resolve_meta_image_path(images_root, relative_stem):
    """Resolve one metadata entry to an on-disk image path across known extensions."""
    # The metadata stores paths without file extensions, so probe the known image
    # suffixes until we find the real file on disk.
    for extension in IMAGE_EXTENSIONS:
        image_path = images_root / f"{relative_stem}{extension}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"Image file not found for metadata entry: {relative_stem}")


def _image_map_from_meta_split(dataset_root, split_name):
    """Build a `{class_name: [image_path, ...]}` mapping for a metadata split."""
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    image_map = {}
    images_root = dataset_root / "images"
    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    for relative_stem in lines:
        class_name = relative_stem.split("/")[0]
        # Keep the original split ordering within each class so repeated runs are
        # deterministic before the later random sampling step.
        image_map.setdefault(class_name, []).append(
            _resolve_meta_image_path(images_root, relative_stem)
        )

    # Sort keys once here so all downstream displays and sampling use a stable
    # class ordering across notebook runs.
    return dict(sorted(image_map.items()))


def display_dataset_count(data_dir):
    """Print per-class image counts for the available Food-101 metadata splits."""
    dataset_root = _resolve_dataset_root(data_dir)
    split_candidates = [
        ("train", "Train Set"),
        ("test", "Test Set"),
    ]

    # Track whether at least one known split was found so we can raise a helpful
    # error instead of silently printing nothing.
    found_any_split = False

    for split_key, split_title in split_candidates:
        counts = _count_from_meta_split(dataset_root, split_key)
        if counts is None:
            continue

        found_any_split = True
        total = sum(counts.values())
        # Align the console output so class counts remain readable even with
        # varying class-name lengths.
        max_name_len = max(len(name) for name in counts)

        print(f"--- {split_title} ---")
        for class_name, num_images in counts.items():
            print(f"- {class_name:<{max_name_len}} : {num_images} images")
        print("-" * (max_name_len + 20))
        print(f"Total: {total} images\n")

    if not found_any_split:
        raise ValueError(f"No Food-101 metadata splits found in: {dataset_root}")


def display_random_images(data_dir, num_classes=3, images_per_class=2, random_seed=None):
    """Display a small random gallery from the Food-101 training split."""
    dataset_root = _resolve_dataset_root(data_dir)
    image_map = _image_map_from_meta_split(dataset_root, "train")
    if image_map is None:
        raise ValueError(f"No readable Food-101 training split found in: {dataset_root}")

    # Sample only from classes that can fill the requested row width.
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
    # Sample classes first, then sample images within each class, so each row
    # represents one class and each column shows different examples from it.
    selected_classes = rng.sample(eligible_classes, num_classes)

    # Force a 2D axes array so the indexing logic stays identical for 1-row and
    # multi-row layouts.
    fig, axes = plt.subplots(
        nrows=num_classes,
        ncols=images_per_class,
        figsize=(images_per_class * 5, num_classes * 5),
        layout="constrained",
        squeeze=False,
    )

    for row_index, class_name in enumerate(selected_classes):
        # Each row corresponds to one sampled class from the training split.
        selected_images = rng.sample(image_map[class_name], images_per_class)
        for col_index, image_path in enumerate(selected_images):
            axis = axes[row_index, col_index]
            # `plt.imread(...)` is sufficient here because the notebook only
            # needs a quick qualitative preview, not a training transform.
            axis.imshow(plt.imread(image_path))
            axis.set_title(class_name, fontsize=16)
            axis.axis("off")

    plt.show()


# ---------------------------------------------------------------------------
# Notebook 01 helpers: MLflow, prediction inspection, and report figures
# ---------------------------------------------------------------------------

def start_mlflow_ui(tracking_dir=None, host="127.0.0.1", port=5000, timeout=15):
    """Start the MLflow UI used in notebook 01, reusing the shared implementation below."""
    return _start_mlflow_ui_impl(
        tracking_dir=tracking_dir,
        host=host,
        port=port,
        timeout=timeout,
    )


# Internal visualization helpers for notebook 01 live below the public MLflow
# wrapper so the notebook-facing API remains grouped in reading order.
def _unwrap_dataset(dataset):
    """Peel nested dataset wrappers until the base dataset is reached."""
    # `Subset` and other wrapper datasets typically expose the wrapped dataset on
    # a `.dataset` attribute. Follow that chain until the real base dataset.
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _resolve_class_names(dataset):
    """Extract class names from a dataset or subset backed by torchvision datasets."""
    base_dataset = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        # Torchvision datasets usually expose human-readable labels on `.classes`.
        return list(base_dataset.classes)
    raise ValueError("Cannot resolve class names from the provided dataset.")


def _denormalize_image(image_tensor, mean, std):
    """Undo channel-wise normalization so a tensor can be plotted correctly."""
    # Broadcast mean/std from `(3,)` to `(3, 1, 1)` so the inverse transform can
    # be applied channel-wise across the entire image tensor.
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std_tensor + mean_tensor).clamp(0, 1)


def _remember_figure(fig):
    """Cache the latest rendered figure so later save calls can reuse it."""
    global _LAST_RENDERED_FIGURE
    # Return the same figure for convenience so callers can chain/use it directly.
    _LAST_RENDERED_FIGURE = fig
    return fig


def _resolve_figure_for_saving(fig=None):
    """Return the explicit figure, the cached figure, or the current active figure."""
    if fig is not None:
        # Respect an explicit figure first so callers can save non-active plots.
        return fig

    global _LAST_RENDERED_FIGURE
    if _LAST_RENDERED_FIGURE is not None:
        try:
            # The cached handle may point to a figure that has already been
            # closed, so confirm it still exists before reusing it.
            if plt.fignum_exists(_LAST_RENDERED_FIGURE.number):
                return _LAST_RENDERED_FIGURE
        except Exception:
            pass

    current_fig = plt.gcf()
    # `plt.gcf()` always returns a figure object, even when nothing has been
    # drawn yet. Require at least one axis so save calls do not create blanks.
    if current_fig.axes:
        return current_fig

    raise ValueError(
        "No plotted Matplotlib figure is available to save. "
        "Pass `fig=...` explicitly or rerun the plotting cell first."
    )


def _require_inference_model(model):
    """Validate that the supplied object behaves like a PyTorch inference model."""
    if hasattr(model, "eval") and hasattr(model, "parameters"):
        return model

    # A frequent notebook mistake is passing the Trainer instead of the trained
    # module. Catch it here and give a task-specific error message.
    if hasattr(model, "lightning_module") or model.__class__.__name__ == "Trainer":
        raise TypeError(
            "`show_random_validation_predictions` expects a trained model, not a Trainer. "
            "Pass the loaded `best_model` checkpoint or `trainer.lightning_module`."
        )

    raise TypeError(
        "`show_random_validation_predictions` expects a PyTorch/Lightning model with "
        "`eval()` and `parameters()` methods."
    )


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
    """Plot random validation samples with predicted and ground-truth labels."""
    if getattr(data_module, "val_dataset", None) is None:
        # Lazily initialize the validation split so callers can pass in a fresh
        # data module without remembering to call `setup("fit")` first.
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
    # Sample unique indices so the figure shows distinct validation examples.
    selected_indices = rng.sample(range(dataset_size), num_images)

    model = _require_inference_model(model)
    was_training = getattr(model, "training", False)
    # Switch to eval mode for deterministic inference, but restore the original
    # mode on exit so this helper does not leave side effects behind.
    model.eval()

    try:
        # Models without parameters are rare, but defaulting to CPU keeps the
        # helper usable for lightweight wrapper modules.
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    rows = math.ceil(num_images / max_cols)
    cols = min(num_images, max_cols)
    if figsize is None:
        # Scale the default canvas from the chosen grid size so labels stay legible.
        figsize = (cols * 4.5, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    # Flatten once so the plotting loop does not need separate row/column math.
    flat_axes = axes.flatten()

    try:
        # Disable autograd to reduce overhead during notebook visualization.
        with torch.inference_mode():
            for axis, sample_idx in zip(flat_axes, selected_indices):
                # Pull one transformed validation sample directly from the dataset.
                image_tensor, true_label = val_dataset[sample_idx]
                # Add a batch dimension because models expect shape `(N, C, H, W)`.
                logits = model(image_tensor.unsqueeze(0).to(device))
                pred_label = int(logits.argmax(dim=1).item())

                # Move back to CPU and undo dataset normalization so matplotlib
                # receives a standard HWC image in the `[0, 1]` range.
                display_image = _denormalize_image(
                    image_tensor.detach().cpu(),
                    denormalize_mean,
                    denormalize_std,
                ).permute(1, 2, 0)

                axis.imshow(display_image)
                axis.axis("off")

                # Use title color as a quick correctness cue when scanning many samples.
                is_correct = pred_label == int(true_label)
                title_color = "green" if is_correct else "red"
                axis.set_title(
                    f"Pred: {class_names[pred_label]}\nTrue: {class_names[int(true_label)]}",
                    fontsize=11,
                    color=title_color,
                )

            # Hide any unused subplot slots when `num_images` is not a multiple
            # of `max_cols`.
            for axis in flat_axes[num_images:]:
                axis.axis("off")
    finally:
        if was_training:
            # Restore the original mode so this visualization helper is safe to
            # call in the middle of training/debugging sessions.
            model.train()

    plt.tight_layout()
    # Remember the rendered figure so a later `save_figure_to_artifacts()` call
    # can find it automatically.
    _remember_figure(fig)
    plt.show()
    return fig, axes


def show_test_prediction_examples(
    model,
    test_dataset,
    *,
    num_correct=2,
    num_incorrect=2,
    batch_size=64,
    num_workers=0,
    random_seed=None,
    class_names=None,
    denormalize_mean=(0.485, 0.456, 0.406),
    denormalize_std=(0.229, 0.224, 0.225),
    figsize=None,
):
    """Plot random correct and incorrect predictions from a test dataset."""
    if test_dataset is None:
        raise ValueError("`test_dataset` cannot be None.")

    # Fail fast with a clear message instead of constructing an empty DataLoader.
    dataset_size = len(test_dataset)
    if dataset_size == 0:
        raise ValueError("Test dataset is empty.")

    # Reuse dataset metadata when the caller does not pass explicit class names.
    if class_names is None:
        class_names = _resolve_class_names(test_dataset)

    # Force the model into a safe inference configuration for visualization.
    model = _require_inference_model(model)
    was_training = getattr(model, "training", False)
    model.eval()

    try:
        # Match the loader output to the device that already holds the model weights.
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Store examples separately so the final figure can show one correctness group
    # per column.
    correct_examples = []
    incorrect_examples = []

    # Shuffle deterministically when a seed is provided, but keep DataLoader
    # shuffling disabled so the sampled order is reproducible.
    rng = random.Random(random_seed)
    shuffled_indices = list(range(dataset_size))
    rng.shuffle(shuffled_indices)
    sampled_subset = torch.utils.data.Subset(test_dataset, shuffled_indices)
    loader = torch.utils.data.DataLoader(
        sampled_subset,
        batch_size=batch_size,
        shuffle=False,
        # The subset itself was already shuffled, so keeping the loader ordered
        # preserves deterministic sampling under the chosen random seed.
        num_workers=num_workers,
    )

    try:
        # Disable autograd because this helper only needs forward passes.
        with torch.inference_mode():
            for image_batch, true_labels in loader:
                logits = model(image_batch.to(device))
                # Move predictions back to CPU immediately because the example
                # records stored below are plain visualization metadata.
                pred_labels = logits.argmax(dim=1).detach().cpu()

                for image_tensor, true_label, pred_label in zip(
                    image_batch,
                    true_labels,
                    pred_labels,
                ):
                    example = {
                        "image_tensor": image_tensor.detach().cpu(),
                        "true_label": int(true_label),
                        "pred_label": int(pred_label),
                    }

                    # Keep collecting examples until each requested bucket is full.
                    if int(pred_label) == int(true_label):
                        if len(correct_examples) < num_correct:
                            correct_examples.append(example)
                    else:
                        if len(incorrect_examples) < num_incorrect:
                            incorrect_examples.append(example)

                    # Stop early as soon as we have enough samples from both groups.
                    if (
                        len(correct_examples) >= num_correct
                        and len(incorrect_examples) >= num_incorrect
                    ):
                        break

                if (
                    len(correct_examples) >= num_correct
                    and len(incorrect_examples) >= num_incorrect
                ):
                    break
    finally:
        if was_training:
            # Restore the original mode so calling this helper does not have
            # side effects on subsequent training code.
            model.train()

    # Surface an explicit error when the dataset does not contain enough samples
    # of either type, instead of silently returning a partial figure.
    if len(correct_examples) < num_correct:
        raise ValueError(
            f"Only found {len(correct_examples)} correct predictions, fewer than "
            f"the requested {num_correct}."
        )

    if len(incorrect_examples) < num_incorrect:
        raise ValueError(
            f"Only found {len(incorrect_examples)} incorrect predictions, fewer than "
            f"the requested {num_incorrect}."
        )

    rows = max(len(correct_examples), len(incorrect_examples))
    if figsize is None:
        figsize = (12, rows * 4.5)

    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    # Keep the layout consistent: correct samples on the left, mistakes on the right.
    column_specs = [
        ("Correct Predictions", correct_examples, "green"),
        ("Incorrect Predictions", incorrect_examples, "red"),
    ]

    for col_idx, (group_title, examples, title_color) in enumerate(column_specs):
        for row_idx in range(rows):
            axis = axes[row_idx, col_idx]

            if row_idx >= len(examples):
                axis.axis("off")
                continue

            example = examples[row_idx]
            # Undo normalization so matplotlib receives a standard RGB image.
            display_image = _denormalize_image(
                example["image_tensor"],
                denormalize_mean,
                denormalize_std,
            ).permute(1, 2, 0)

            axis.imshow(display_image)
            axis.axis("off")
            # Color the title to reinforce the correctness grouping at a glance.
            axis.set_title(
                (
                    f"Pred: {class_names[example['pred_label']]}\n"
                    f"True: {class_names[example['true_label']]}"
                ),
                fontsize=11,
                color=title_color,
            )

    plt.tight_layout()
    # Cache the latest figure so notebook code can save it without threading the
    # figure object through every call site.
    _remember_figure(fig)
    plt.show()
    return fig, axes, {
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
    }


def show_test_prediction_gradcam_examples(
    examples,
    *,
    class_names,
    figsize=None,
):
    """Plot already-computed Grad-CAM overlays for correct and incorrect samples."""
    if "correct_examples" not in examples or "incorrect_examples" not in examples:
        raise ValueError(
            "`examples` must contain `correct_examples` and `incorrect_examples`."
        )

    # Copy into lists so callers can pass any iterable-like containers.
    correct_examples = list(examples["correct_examples"])
    incorrect_examples = list(examples["incorrect_examples"])
    rows = max(len(correct_examples), len(incorrect_examples))
    if rows == 0:
        raise ValueError("`examples` does not contain any samples to visualize.")

    if figsize is None:
        figsize = (12, rows * 4.5)

    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
    # Mirror the plain prediction plot so the user can compare both figures easily.
    column_specs = [
        ("Correct Predictions", correct_examples, "green"),
        ("Incorrect Predictions", incorrect_examples, "red"),
    ]

    for col_idx, (group_title, grouped_examples, title_color) in enumerate(column_specs):
        for row_idx in range(rows):
            axis = axes[row_idx, col_idx]
            if row_idx >= len(grouped_examples):
                axis.axis("off")
                continue

            example = grouped_examples[row_idx]
            # Validate required metadata here so missing Grad-CAM outputs fail with
            # a precise error message.
            if "cam_image" not in example:
                raise ValueError("Each Grad-CAM example must include a `cam_image` entry.")
            if "target_label" not in example or "target_label_name" not in example:
                raise ValueError(
                    "Each Grad-CAM example must include `target_label` and `target_label_name`."
                )

            axis.imshow(example["cam_image"])
            axis.axis("off")
            # Build the title incrementally so optional CAM-target metadata can
            # be inserted without duplicating the common Pred/True lines.
            title_lines = []
            # Only add the CAM target line when the heatmap is not already based on
            # the predicted class, which keeps the common case visually compact.
            if str(example["target_label_name"]).strip().lower() != "pred":
                title_lines.append(
                    f"CAM: {example['target_label_name']} "
                    f"({class_names[example['target_label']]})"
                )
            title_lines.append(f"Pred: {class_names[example['pred_label']]}")
            title_lines.append(f"True: {class_names[example['true_label']]}")
            axis.set_title(
                "\n".join(title_lines),
                fontsize=11,
                color=title_color,
            )

    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return fig, axes


def _load_food101_class_names_from_metadata():
    """Best-effort fallback for Food-101 class names when no dataset is passed in."""
    classes_path = _project_root() / "data" / "food-101" / "meta" / "classes.txt"
    if not classes_path.exists():
        return None

    # Strip blank lines so accidental trailing newlines in the metadata file do
    # not create empty class labels.
    class_names = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    return class_names or None


def show_prediction_grid(
    input_data,
    true_labels,
    predictions,
    *,
    class_names=None,
    denormalize_mean=(0.485, 0.456, 0.406),
    denormalize_std=(0.229, 0.224, 0.225),
    max_cols=4,
    figsize=None,
):
    """Plot ONNX prediction results for a batch of normalized image tensors."""
    input_array = np.asarray(input_data)
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)

    if input_array.ndim != 4:
        raise ValueError(
            "`input_data` must have shape `(batch, channels, height, width)`."
        )

    num_images = input_array.shape[0]
    if num_images == 0:
        raise ValueError("`input_data` does not contain any images to display.")

    if len(true_labels) != num_images:
        raise ValueError(
            "`true_labels` must have the same length as the batch dimension in `input_data`."
        )

    if predictions.ndim == 2:
        # Treat 2D predictions as classifier logits/probabilities and reduce
        # them to one label id per image.
        if predictions.shape[0] != num_images:
            raise ValueError(
                "When `predictions` contains logits, its batch dimension must match `input_data`."
            )
        pred_labels = predictions.argmax(axis=1)
        num_classes = predictions.shape[1]
    elif predictions.ndim == 1:
        # Treat 1D predictions as precomputed label ids and infer the class
        # count from the largest label seen in predictions or ground truth.
        if predictions.shape[0] != num_images:
            raise ValueError(
                "When `predictions` contains label ids, its length must match `input_data`."
            )
        pred_labels = predictions.astype(int)
        num_classes = int(max(np.max(pred_labels), np.max(true_labels)) + 1)
    else:
        raise ValueError(
            "`predictions` must either be a 1D array of label ids or a 2D array of logits."
        )

    if class_names is None:
        class_names = _load_food101_class_names_from_metadata()
    if class_names is None:
        # Fall back to generic labels so visualization still works for exported
        # batches even when Food-101 metadata is unavailable locally.
        class_names = [f"Class {idx}" for idx in range(num_classes)]
    else:
        class_names = list(class_names)

    if len(class_names) < num_classes:
        raise ValueError(
            f"`class_names` only contains {len(class_names)} entries, but at least "
            f"{num_classes} are required."
        )

    rows = math.ceil(num_images / max_cols)
    cols = min(num_images, max_cols)
    if figsize is None:
        figsize = (cols * 4.5, rows * 4.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    # Flattening keeps the plotting loop simple even when the grid has multiple rows.
    flat_axes = axes.flatten()

    for axis, image_array, true_label, pred_label in zip(
        flat_axes,
        input_array,
        true_labels,
        pred_labels,
    ):
        # Convert each sample back to a tensor so the same denormalization helper
        # can be reused for both PyTorch and ONNX notebook outputs.
        image_tensor = torch.as_tensor(image_array).detach().cpu()
        display_image = _denormalize_image(
            image_tensor,
            denormalize_mean,
            denormalize_std,
        ).permute(1, 2, 0)

        axis.imshow(display_image)
        axis.axis("off")
        is_correct = int(pred_label) == int(true_label)
        axis.set_title(
            f"Pred: {class_names[int(pred_label)]}\nTrue: {class_names[int(true_label)]}",
            fontsize=11,
            color="green" if is_correct else "red",
        )

    for axis in flat_axes[num_images:]:
        axis.axis("off")

    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return fig, axes


def save_figure_to_artifacts(
    filename,
    fig=None,
    artifact_subdir="figures",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
):
    """Save a Matplotlib figure under `artifacts/<artifact_subdir>/`."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "artifacts" / artifact_subdir
    # Create the artifact folder lazily so notebook users can call the helper
    # without preparing directories manually.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(filename)
    if output_path.suffix == "":
        # Default to PNG so notebook users can pass a bare stem like `loss_curve`.
        output_path = output_path.with_suffix(".png")

    fig = _resolve_figure_for_saving(fig)

    # Save only the file name inside the chosen artifact subdirectory to prevent
    # accidental writes outside the project artifact tree.
    save_path = output_dir / output_path.name
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
    )
    return save_path


def compare_stage_training_runs(
    *,
    stage1_experiment_name,
    stage2_experiment_name,
    stage1_run_name,
    stage2_run_name,
    tracking_uri=None,
    stage1_name="Stage 1",
    stage2_name="Stage 2",
    one_based_epoch=True,
    print_summary=False,
    plot=False,
    return_details=False,
):
    """Compare notebook 01 stage-training runs using the shared MLflow implementation."""
    return _compare_stage_training_runs_impl(
        stage1_experiment_name=stage1_experiment_name,
        stage2_experiment_name=stage2_experiment_name,
        stage1_run_name=stage1_run_name,
        stage2_run_name=stage2_run_name,
        tracking_uri=tracking_uri,
        stage1_name=stage1_name,
        stage2_name=stage2_name,
        one_based_epoch=one_based_epoch,
        print_summary=print_summary,
        plot=plot,
        return_details=return_details,
    )


# ---------------------------------------------------------------------------
# Notebook 02 helpers: DataLoader benchmarking and efficiency analysis
# ---------------------------------------------------------------------------

def compute_accuracy(model, loader, device):
    """Compute classification accuracy for one model over one evaluation DataLoader."""
    return _compute_accuracy_impl(model, loader, device)


def compute_onnx_accuracy(session, loader):
    """Compute classification accuracy for one ONNX Runtime session over one DataLoader."""
    return _compute_onnx_accuracy_impl(session, loader)


def sparsity_report(model):
    """Summarize weight sparsity across the prunable layers of one model."""
    return _sparsity_report_impl(model)


def bench(
    model,
    device,
    *,
    batch_size=32,
    image_size=224,
    num_warmup=10,
    num_iterations=50,
):
    """Benchmark average forward time per synthetic image batch."""
    return _bench_impl(
        model,
        device,
        batch_size=batch_size,
        image_size=image_size,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def measure_average_epoch_time(loader, device, num_epochs=5, num_warmup_epochs=2):
    """Measure the average iteration time of one DataLoader configuration across epochs."""
    return _measure_average_epoch_time_impl(
        loader,
        device,
        num_epochs=num_epochs,
        num_warmup_epochs=num_warmup_epochs,
    )


def run_experiment(
    *,
    experiment_name,
    experiment_fcn,
    cases,
    rerun=False,
    checkpoint_dir=None,
    **experiment_kwargs,
):
    """Run one cached notebook experiment and return the ordered case-to-result mapping."""
    return _run_experiment_impl(
        experiment_name=experiment_name,
        experiment_fcn=experiment_fcn,
        cases=cases,
        rerun=rerun,
        checkpoint_dir=checkpoint_dir,
        **experiment_kwargs,
    )


def plot_performance_summary(
    performance_by_case,
    title="Performance Summary",
    xlabel="Case",
    ylabel="Average Time per Epoch (milliseconds)",
    *,
    convert_to_milliseconds=True,
    figsize=(12, 6),
    line_color="#2b8ead",
    marker="o",
    linestyle="--",
    linewidth=2.5,
    markersize=9,
    annotation_decimals=2,
):
    """Plot one minimalist performance curve for any ordered set of benchmark cases."""
    return _plot_performance_summary_impl(
        performance_by_case,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        convert_to_milliseconds=convert_to_milliseconds,
        figsize=figsize,
        line_color=line_color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        annotation_decimals=annotation_decimals,
    )


def visualize_dataloader_efficiency(
    loaders_to_compare,
    device,
    *,
    num_batches=30,
    num_warmup_batches=5,
    title="DataLoader Performance Comparison (Efficiency)",
    xlabel="DataLoader Configuration",
    ylabel="Percentage of Average Time per Batch (%)",
    active_label="GPU Active Time",
    idle_label="GPU Idle / Waiting Time",
    active_color="#2563A6",
    idle_color="#D9E8F5",
    edge_color="#D0DCE8",
    figsize=(12, 8),
):
    """Plot the active-vs-idle batch-time split using a minimalist blue stacked bar chart."""
    return _visualize_dataloader_efficiency_impl(
        loaders_to_compare,
        device,
        num_batches=num_batches,
        num_warmup_batches=num_warmup_batches,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        active_label=active_label,
        idle_label=idle_label,
        active_color=active_color,
        idle_color=idle_color,
        edge_color=edge_color,
        figsize=figsize,
    )


def get_onnx_artifact_size_mb(path):
    """Return one ONNX artifact size in megabytes, including sidecar weight files."""
    return _get_onnx_artifact_size_mb_impl(path)


def bench_onnx_session(
    session,
    *,
    batch_size=32,
    image_size=224,
    num_warmup=10,
    num_iterations=50,
):
    """Benchmark average forward time per synthetic ONNX Runtime image batch."""
    return _bench_onnx_session_impl(
        session,
        batch_size=batch_size,
        image_size=image_size,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )


def build_onnx_quantization_comparison_df(
    fp32_onnx_path,
    quantized_onnx_path,
    *,
    fp32_time_s,
    quantized_time_s,
    fp32_acc=None,
    quantized_acc=None,
    fp32_label="Baseline ONNX",
    quantized_label="Quantized ONNX (Static)",
):
    """Build a comparison table for baseline vs. quantized ONNX artifact size, latency, and accuracy."""
    return _build_onnx_quantization_comparison_df_impl(
        fp32_onnx_path,
        quantized_onnx_path,
        fp32_time_s=fp32_time_s,
        quantized_time_s=quantized_time_s,
        fp32_acc=fp32_acc,
        quantized_acc=quantized_acc,
        fp32_label=fp32_label,
        quantized_label=quantized_label,
    )


def _plot_performance_summary_impl(
    performance_by_case,
    title="Performance Summary",
    xlabel="Case",
    ylabel="Average Time per Epoch (milliseconds)",
    *,
    convert_to_milliseconds=True,
    figsize=(12, 6),
    line_color="#2b8ead",
    marker="o",
    linestyle="--",
    linewidth=2.5,
    markersize=9,
    annotation_decimals=2,
):
    """Implementation for plotting one performance curve with value labels above each point."""
    if not performance_by_case:
        raise ValueError("`performance_by_case` cannot be empty.")

    ordered_items = [
        (_restore_case_key(case), value) for case, value in performance_by_case.items()
    ]
    # Build x/y arrays manually so invalid cached values can be skipped while
    # still preserving the original case ordering in the notebook plot.

    x_labels = []
    y_values = []
    invalid_cases = []

    for case, value in ordered_items:
        value = _json_safe_value(value)
        if value is None or not np.isfinite(value):
            invalid_cases.append(case)
            continue

        x_labels.append(str(case))
        y_values.append(float(value) * 1000.0 if convert_to_milliseconds else float(value))

    if not x_labels:
        raise ValueError("No finite values are available to plot.")

    x_positions = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x_positions,
        y_values,
        color=line_color,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
    )

    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="-", linewidth=1, alpha=0.45)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)

    y_min = min(y_values)
    y_max = max(y_values)
    # Keep a small top margin above the highest point so annotations do not clip.
    y_span = max(y_max - y_min, y_max * 0.08, 1.0)
    annotation_offset = y_span * 0.06
    ax.set_ylim(bottom=0, top=y_max + annotation_offset * 2.0)

    for x_position, y_value in zip(x_positions, y_values):
        ax.annotate(
            f"{y_value:.{annotation_decimals}f}",
            xy=(x_position, y_value),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            color=line_color,
        )

    if invalid_cases:
        print(
            "Skipped non-finite values for cases: "
            + ", ".join(str(case) for case in invalid_cases)
        )

    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return fig, ax


def _get_onnx_artifact_size_mb_impl(path):
    """Implementation for returning one ONNX artifact size including sidecar weight files."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file does not exist: {path}")

    artifact_paths = {path}
    # ONNX external-data exports may split tensor weights into sidecar files, so
    # include them to reflect the actual deployable artifact size.
    artifact_paths.update(
        sidecar_path
        for sidecar_path in path.parent.glob(f"{path.name}.data*")
        if sidecar_path.is_file()
    )

    total_size_bytes = sum(artifact_path.stat().st_size for artifact_path in artifact_paths)
    return total_size_bytes / (1024 ** 2)


def _bench_onnx_session_impl(
    session,
    *,
    batch_size=32,
    image_size=224,
    num_warmup=10,
    num_iterations=50,
):
    """Implementation for timing average ONNX Runtime latency on synthetic image batches."""
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if image_size <= 0:
        raise ValueError("`image_size` must be a positive integer.")
    if num_warmup < 0:
        raise ValueError("`num_warmup` cannot be negative.")
    if num_iterations <= 0:
        raise ValueError("`num_iterations` must be a positive integer.")

    if session is None:
        raise ValueError("`session` cannot be None.")

    input_metadata = session.get_inputs()
    if not input_metadata:
        raise ValueError("The ONNX Runtime session does not expose any inputs.")

    input_name = input_metadata[0].name
    rng = np.random.default_rng(0)
    # Use a fixed synthetic batch so repeated latency measurements stay comparable
    # across candidate ONNX exports inside the notebook.
    synthetic_batch = np.ascontiguousarray(
        rng.standard_normal((batch_size, 3, image_size, image_size), dtype=np.float32)
    )

    for _ in range(num_warmup):
        # Warm-up runs let ORT finish one-time graph/runtime setup before timing.
        _ = session.run(None, {input_name: synthetic_batch})

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: synthetic_batch})
    elapsed_seconds = time.perf_counter() - start_time

    return elapsed_seconds / num_iterations


def _compute_onnx_accuracy_impl(session, loader):
    """Implementation for evaluating one ONNX Runtime classifier over a labeled DataLoader."""
    if session is None:
        raise ValueError("`session` cannot be None.")
    if loader is None:
        raise ValueError("`loader` cannot be None.")

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if total_batches == 0:
        raise ValueError("`loader` must contain at least one batch.")

    input_metadata = session.get_inputs()
    if not input_metadata:
        raise ValueError("The ONNX Runtime session does not expose any inputs.")

    input_name = input_metadata[0].name
    correct_predictions = 0
    total_examples = 0
    progress_bar = None

    try:
        if tqdm is not None:
            progress_bar = tqdm(loader, desc="Computing ONNX Accuracy", total=total_batches)
            batch_iterator = progress_bar
        else:
            batch_iterator = loader

        for batch in batch_iterator:
            inputs, targets = _extract_inputs_and_targets(batch)

            if not torch.is_tensor(inputs):
                # Accept list/NumPy batches too, then normalize them to tensors
                # before converting to contiguous NumPy arrays for ORT.
                inputs = torch.as_tensor(inputs)
            if not torch.is_tensor(targets):
                targets = torch.as_tensor(targets)

            ort_inputs = {
                input_name: np.ascontiguousarray(
                    # ORT inference in this notebook expects FP32 image tensors.
                    inputs.detach().cpu().numpy().astype(np.float32)
                )
            }
            logits = session.run(None, ort_inputs)[0]
            predictions = np.asarray(logits).argmax(axis=1)
            targets_np = targets.detach().cpu().numpy()

            correct_predictions += int((predictions == targets_np).sum())
            total_examples += int(targets_np.size)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if total_examples == 0:
        raise ValueError("`loader` did not yield any labeled examples.")

    return correct_predictions / total_examples


def _build_onnx_quantization_comparison_df_impl(
    fp32_onnx_path,
    quantized_onnx_path,
    *,
    fp32_time_s,
    quantized_time_s,
    fp32_acc=None,
    quantized_acc=None,
    fp32_label="Baseline ONNX",
    quantized_label="Quantized ONNX (Static)",
):
    """Implementation for comparing ONNX artifact size, latency, and optional accuracy."""
    fp32_size_mb = _get_onnx_artifact_size_mb_impl(fp32_onnx_path)
    quantized_size_mb = _get_onnx_artifact_size_mb_impl(quantized_onnx_path)

    baseline_values = [
        fp32_size_mb,
        fp32_time_s * 1e3,
    ]
    quantized_values = [
        quantized_size_mb,
        quantized_time_s * 1e3,
    ]
    change_values = [
        fp32_size_mb - quantized_size_mb,
        (fp32_time_s - quantized_time_s) * 1e3,
    ]
    row_index = [
        "Model Size (MB)",
        "Inference Latency (ms)",
    ]

    if fp32_acc is not None and quantized_acc is not None:
        # Accuracy is optional because some notebook cells compare artifacts
        # before the final end-to-end evaluation has finished running.
        baseline_values.append(fp32_acc * 100)
        quantized_values.append(quantized_acc * 100)
        change_values.append((quantized_acc - fp32_acc) * 100)
        row_index.append("Accuracy (%)")

    return pd.DataFrame(
        {
            fp32_label: baseline_values,
            quantized_label: quantized_values,
            "Change": change_values,
        },
        index=row_index,
    )


def _extract_inputs_and_targets(batch):
    """Normalize common batch structures into an `(inputs, targets)` pair."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        # Covers the usual `(images, labels)` dataset / DataLoader convention.
        return batch[0], batch[1]

    if isinstance(batch, dict):
        # Support a handful of common naming conventions so helper functions can
        # work with custom datasets without per-notebook adapter code.
        for input_key, target_key in (
            ("image", "label"),
            ("images", "labels"),
            ("input", "target"),
            ("inputs", "targets"),
            ("x", "y"),
        ):
            if input_key in batch and target_key in batch:
                return batch[input_key], batch[target_key]

    raise TypeError(
        "Expected each batch to contain inputs and targets, such as `(images, labels)`."
    )


def _extract_prediction_logits(model_output):
    """Extract logits from common classifier output structures."""
    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, dict) and "logits" in model_output:
        # Hugging Face style outputs often expose logits under a named key.
        return model_output["logits"]

    if isinstance(model_output, (list, tuple)) and model_output:
        first_item = model_output[0]
        if torch.is_tensor(first_item):
            return first_item

    raise TypeError("Model output must be a tensor or contain logits in a standard structure.")


def _iter_prunable_weight_modules(model):
    """Yield named Conv/Linear modules whose weights participate in pruning."""
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, "weight"):
            weight = getattr(module, "weight", None)
            if weight is not None:
                # Yield both the name and module so callers can build readable reports.
                yield module_name, module


def _count_zero_weights(weight_tensor):
    """Return the number of exact zeros in one weight tensor."""
    return int((weight_tensor == 0).sum().item())


def _resolve_floating_dtype(model):
    """Infer a floating-point dtype from the model parameters or buffers."""
    for tensor in model.parameters():
        if tensor.is_floating_point():
            # Match the synthetic benchmark batch dtype to the model weights to
            # avoid accidental dtype promotion during timing.
            return tensor.dtype
    for tensor in model.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def _sparsity_report_impl(model):
    """Implementation for summarizing model sparsity after optional pruning masks."""
    model = _require_inference_model(model)

    module_rows = []
    total_params = 0
    total_zero_params = 0

    for module_name, module in _iter_prunable_weight_modules(model):
        # Read the realized weight tensor directly so pruning masks that have
        # already been materialized into zeros are reflected in the report.
        weight = module.weight.detach()
        param_count = int(weight.numel())
        zero_count = _count_zero_weights(weight)
        sparsity = zero_count / param_count if param_count > 0 else 0.0

        total_params += param_count
        total_zero_params += zero_count
        module_rows.append(
            {
                "module": module_name,
                "type": module.__class__.__name__,
                "params": param_count,
                "zero_params": zero_count,
                "sparsity": sparsity,
            }
        )

    if total_params == 0:
        raise ValueError("No prunable Conv2d/Linear weights were found in the provided model.")

    return {
        # Return both a compact global metric and a per-layer table so notebook
        # cells can print one summary line or inspect detailed sparsity later.
        "global_sparsity": total_zero_params / total_params,
        "total_params": total_params,
        "zero_params": total_zero_params,
        "module_sparsity": pd.DataFrame(module_rows),
    }


def _bench_impl(
    model,
    device,
    *,
    batch_size=32,
    image_size=224,
    num_warmup=10,
    num_iterations=50,
):
    """Implementation for timing average forward latency on synthetic image batches."""
    if batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    if image_size <= 0:
        raise ValueError("`image_size` must be a positive integer.")
    if num_warmup < 0:
        raise ValueError("`num_warmup` cannot be negative.")
    if num_iterations <= 0:
        raise ValueError("`num_iterations` must be a positive integer.")

    model = _require_inference_model(model)
    device = torch.device(device)
    was_training = getattr(model, "training", False)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    if model_device != device:
        # Time execution on the caller-requested device even if the supplied
        # model checkpoint currently lives elsewhere.
        model = model.to(device)

    model.eval()

    input_dtype = _resolve_floating_dtype(model)
    # Use synthetic data so latency comparisons are not confounded by I/O.
    synthetic_batch = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
        dtype=input_dtype,
    )

    try:
        with torch.inference_mode():
            for _ in range(num_warmup):
                # Warm-up runs absorb one-time kernel selection / compilation cost.
                _ = model(synthetic_batch)

            _synchronize_device(device)
            start_time = time.perf_counter()

            for _ in range(num_iterations):
                _ = model(synthetic_batch)

            _synchronize_device(device)
            elapsed_seconds = time.perf_counter() - start_time
    finally:
        if was_training:
            model.train()

    return elapsed_seconds / num_iterations


def _compute_accuracy_impl(model, loader, device):
    """Implementation for evaluating one classifier over a labeled DataLoader."""
    model = _require_inference_model(model)
    device = torch.device(device)

    if loader is None:
        raise ValueError("`loader` cannot be None.")

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if total_batches == 0:
        raise ValueError("`loader` must contain at least one batch.")

    was_training = getattr(model, "training", False)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    if model_device != device:
        model = model.to(device)

    model.eval()

    correct_predictions = 0
    total_examples = 0
    progress_bar = None

    try:
        if tqdm is not None:
            progress_bar = tqdm(loader, desc="Computing Accuracy", total=total_batches)
            batch_iterator = progress_bar
        else:
            batch_iterator = loader

        with torch.inference_mode():
            for batch in batch_iterator:
                inputs, targets = _extract_inputs_and_targets(batch)
                inputs = _move_batch_to_device(inputs, device)

                if torch.is_tensor(targets):
                    # Keep targets on the same device as predictions so equality
                    # checks do not trigger implicit copies.
                    targets = targets.to(device)
                else:
                    targets = torch.as_tensor(targets, device=device)

                logits = _extract_prediction_logits(model(inputs))
                predictions = logits.argmax(dim=1)

                correct_predictions += (predictions == targets).sum().item()
                total_examples += targets.numel()
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if was_training:
            model.train()

    if total_examples == 0:
        raise ValueError("No labeled examples were available to compute accuracy.")

    return correct_predictions / total_examples


def _measure_loader_efficiency(
    loader,
    device,
    *,
    num_batches=30,
    num_warmup_batches=5,
):
    """Estimate active vs. waiting time percentages for one DataLoader."""
    if num_batches <= 0:
        raise ValueError("`num_batches` must be a positive integer.")
    if num_warmup_batches < 0:
        raise ValueError("`num_warmup_batches` cannot be negative.")

    loader = _prepare_loader_for_iteration(loader)
    loader_iter = iter(loader)

    # Separate "idle" time (waiting for the next batch to be produced) from
    # "active" time (moving the finished batch onto the target device).
    active_times = []
    idle_times = []
    total_steps = num_warmup_batches + num_batches

    with torch.inference_mode():
        for step_idx in range(total_steps):
            idle_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Short datasets are still valid; just measure however many
                # batches we were able to observe before exhaustion.
                break
            idle_seconds = time.perf_counter() - idle_start

            active_start = time.perf_counter()
            batch = _move_batch_to_device(batch, device)
            # Device copies can be asynchronous on accelerators, so synchronize
            # before stopping the timer to avoid under-reporting active time.
            _synchronize_device(device)
            active_seconds = time.perf_counter() - active_start
            del batch

            if step_idx >= num_warmup_batches:
                # Ignore warm-up iterations so cache fills / worker spin-up do
                # not skew the steady-state percentages.
                idle_times.append(idle_seconds)
                active_times.append(active_seconds)

    if not active_times:
        raise ValueError("Not enough batches were available to measure DataLoader efficiency.")

    total_active = sum(active_times)
    total_idle = sum(idle_times)
    total_time = total_active + total_idle
    active_pct = (total_active / total_time) * 100.0 if total_time > 0 else 0.0
    idle_pct = 100.0 - active_pct

    return {
        "active_pct": active_pct,
        "idle_pct": idle_pct,
        "avg_active_ms": (total_active / len(active_times)) * 1000.0,
        "avg_idle_ms": (total_idle / len(idle_times)) * 1000.0,
        "measured_batches": len(active_times),
    }

# Private implementations for the notebook 02 public wrappers above.
def _visualize_dataloader_efficiency_impl(
    loaders_to_compare,
    device,
    *,
    num_batches=30,
    num_warmup_batches=5,
    title="DataLoader Performance Comparison (Efficiency)",
    xlabel="DataLoader Configuration",
    ylabel="Percentage of Average Time per Batch (%)",
    active_label="GPU Active Time",
    idle_label="GPU Idle / Waiting Time",
    active_color="#2563A6",
    idle_color="#D9E8F5",
    edge_color="#D0DCE8",
    figsize=(12, 8),
):
    """Implementation for the stacked-bar DataLoader efficiency visualization."""
    if not loaders_to_compare:
        raise ValueError("`loaders_to_compare` cannot be empty.")

    summary_rows = []
    for loader_name, loader in loaders_to_compare.items():
        # Measure each candidate independently so the returned summary table can
        # also be reused outside the plotted figure.
        metrics = _measure_loader_efficiency(
            loader,
            device,
            num_batches=num_batches,
            num_warmup_batches=num_warmup_batches,
        )
        summary_rows.append(
            {
                "label": str(loader_name),
                **metrics,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=figsize)
    x_positions = np.arange(len(summary_df))

    active_bars = ax.bar(
        x_positions,
        summary_df["active_pct"],
        color=active_color,
        width=0.8,
        label=active_label,
    )
    idle_bars = ax.bar(
        x_positions,
        summary_df["idle_pct"],
        bottom=summary_df["active_pct"],
        color=idle_color,
        edgecolor=edge_color,
        linewidth=1.0,
        width=0.8,
        label=idle_label,
    )

    ax.set_title(title, fontsize=22, pad=16, weight="semibold")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary_df["label"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=1, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    for bar, active_pct in zip(active_bars, summary_df["active_pct"]):
        # Label the active portion directly because that is the quantity users
        # typically compare when choosing a DataLoader configuration.
        ax.annotate(
            f"{active_pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, active_pct),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="semibold",
            color=active_color,
        )

    plt.tight_layout()
    _remember_figure(fig)
    plt.show()
    return summary_df, fig, ax


def _measure_average_epoch_time_impl(loader, device, num_epochs=5, num_warmup_epochs=2):
    """Implementation for measuring one DataLoader configuration across several epochs."""
    if num_epochs <= 0:
        raise ValueError("`num_epochs` must be a positive integer.")
    if num_warmup_epochs < 0 or num_warmup_epochs >= num_epochs:
        raise ValueError("`num_warmup_epochs` must be between 0 and `num_epochs - 1`.")

    loader = _prepare_loader_for_iteration(loader)
    epoch_times = []

    for epoch_idx, progress_bar in _iter_epoch_progress(num_epochs):
        # Synchronize both before and after the loop so each epoch includes the
        # full cost of outstanding accelerator work and host-to-device copies.
        _synchronize_device(device)
        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch in loader:
                batch = _move_batch_to_device(batch, device)
                del batch

        _synchronize_device(device)
        elapsed_seconds = time.perf_counter() - start_time
        epoch_times.append(elapsed_seconds)

        epoch_message = f"Epoch {epoch_idx}/{num_epochs} | Time: {elapsed_seconds:.2f} seconds"
        if epoch_idx <= num_warmup_epochs:
            epoch_message += " (warm-up)"
        _write_progress_line(progress_bar, epoch_message)

    # Drop the warm-up epochs from the reported average so notebook comparisons
    # focus on the stabilized throughput of each loader configuration.
    measured_times = epoch_times[num_warmup_epochs:]
    average_time = sum(measured_times) / len(measured_times)
    print()
    print(
        f"Average execution time (avg of last {len(measured_times)}): "
        f"{average_time:.2f} seconds"
    )
    print()
    return average_time


def _run_experiment_impl(
    *,
    experiment_name,
    experiment_fcn,
    cases,
    rerun=False,
    checkpoint_dir=None,
    **experiment_kwargs,
):
    """Implementation for running, caching, and restoring notebook experiment results."""
    checkpoint_dir = Path(checkpoint_dir or _experiment_checkpoint_dir())
    # Keep all notebook experiment caches in one shared directory so reruns can
    # reuse previous timing sweeps across sessions.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{experiment_name}.json"
    relative_checkpoint_path = _display_relative_path(checkpoint_path)

    if checkpoint_path.exists() and not rerun:
        # Fast path: reuse the on-disk cache when the caller did not explicitly
        # request a fresh run of the experiment.
        print(
            f"Results for experiment '{experiment_name}' found. "
            f"Loading from {relative_checkpoint_path}"
        )
        cached_results = json.loads(checkpoint_path.read_text())
        return _restore_cached_experiment_results(cached_results, cases)

    if checkpoint_path.exists() and rerun:
        print(
            f"Results for experiment '{experiment_name}' already exist, "
            "but `rerun=True`, so the cache will be overwritten."
        )
    else:
        print(f"Results for experiment '{experiment_name}' not found. Running experiment.")

    print(f"Executing experiment '{experiment_name}'...")
    experiment_results = experiment_fcn(cases, **experiment_kwargs)
    # The experiment function is expected to return a mapping keyed by `cases`.

    # Stringify keys on write because JSON object keys must be strings, while
    # notebook experiments often use ints / floats / NumPy scalars as cases.
    serializable_results = {
        str(_restore_case_key(case)): _json_safe_value(value)
        for case, value in experiment_results.items()
    }
    checkpoint_path.write_text(json.dumps(serializable_results, indent=2, allow_nan=True))

    print(f"Results for experiment '{experiment_name}' saved to {relative_checkpoint_path}")
    return experiment_results


# ---------------------------------------------------------------------------
# Notebook 01 private MLflow implementations
# ---------------------------------------------------------------------------

def _build_epoch_history(df, one_based_epoch=True):
    """Collapse MLflow metric rows into one row per completed epoch."""
    history = (
        # MLflow logs the same metric multiple times within one epoch; keeping the
        # max value after sorting effectively preserves the last logged point.
        df.groupby("epoch", as_index=False)
        .agg({
            "train_loss": "max",
            "val_loss": "max",
            "val_acc": "max",
        })
        .sort_values("epoch")
    )
    # Drop rows for epochs that never produced a training loss, since those are
    # usually incomplete or non-training bookkeeping rows.
    history = history[history["train_loss"].notna()].copy()

    if history.empty:
        raise ValueError("No training epochs with `train_loss` were found in the provided metrics.")

    # Many notebook plots/readouts are easier to interpret with 1-based epoch
    # numbering even though MLflow stores epochs starting at 0.
    epoch_col = "epoch_num" if one_based_epoch else "epoch"
    history[epoch_col] = history["epoch"] + 1 if one_based_epoch else history["epoch"]
    return history, epoch_col


def _default_mlflow_tracking_uri():
    """Return the default local MLflow tracking URI used by this project."""
    project_root = Path(__file__).resolve().parent.parent
    # Support both the legacy `mlruns/` location and the newer
    # `artifacts/mlruns/` layout used by the project notebooks.
    for candidate in (project_root / "mlruns", project_root / "artifacts" / "mlruns"):
        if candidate.exists():
            # Return the first existing store so older and newer notebook layouts
            # both keep working without manual path changes.
            return candidate.as_uri()
    return (project_root / "mlruns").as_uri()


def _load_mlflow_metrics_df(
    *,
    experiment_name,
    run_name,
    tracking_uri=None,
    metric_names=("epoch", "train_loss", "val_loss", "val_acc"),
):
    """Load epoch-level metric histories for one MLflow run into a merged DataFrame."""
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "Reading training metrics from MLflow requires the `mlflow` package to be installed."
        ) from exc

    tracking_uri = tracking_uri or _default_mlflow_tracking_uri()
    # Configure MLflow globally for this process before any experiment lookups.
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name!r}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        # If several runs reused the same run name, compare against the latest one.
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if runs_df.empty:
        raise ValueError(
            f"MLflow run not found for experiment={experiment_name!r}, run_name={run_name!r}."
        )

    run_id = runs_df.iloc[0]["run_id"]
    # Use the low-level client because metric history retrieval is not exposed as
    # a single tidy dataframe in the high-level search API.
    client = mlflow.tracking.MlflowClient()

    metric_frames = []
    for metric_name in metric_names:
        history = client.get_metric_history(run_id, metric_name)
        if not history:
            continue

        # Build one frame per metric and merge them on MLflow step afterwards,
        # because MLflow stores each metric history independently.
        metric_df = pd.DataFrame(
            {
                "step": [metric.step for metric in history],
                metric_name: [metric.value for metric in history],
            }
        )
        # Keep the last value logged for a given step in case callbacks emitted
        # duplicates during the same training iteration.
        metric_df = metric_df.drop_duplicates(subset="step", keep="last")
        metric_frames.append(metric_df)

    if not metric_frames:
        raise ValueError(
            f"No metric history was found for MLflow run={run_name!r} in experiment={experiment_name!r}."
        )

    merged_df = metric_frames[0]
    for metric_df in metric_frames[1:]:
        # Outer merge preserves steps where only a subset of metrics were logged.
        merged_df = merged_df.merge(metric_df, on="step", how="outer")

    merged_df = merged_df.sort_values("step").reset_index(drop=True)
    if "epoch" not in merged_df.columns:
        raise ValueError("MLflow metric history does not contain the required `epoch` metric.")

    # Some metrics are logged on steps where `epoch` itself is omitted. Forward
    # filling reconstructs the epoch label for those rows before aggregation.
    merged_df["epoch"] = merged_df["epoch"].ffill()
    merged_df = merged_df[merged_df["epoch"].notna()].copy()
    merged_df["epoch"] = merged_df["epoch"].astype(int)
    return merged_df


def _compare_stage_training_runs_impl(
    *,
    stage1_experiment_name,
    stage2_experiment_name,
    stage1_run_name,
    stage2_run_name,
    tracking_uri=None,
    stage1_name="Stage 1",
    stage2_name="Stage 2",
    one_based_epoch=True,
    print_summary=False,
    plot=False,
    return_details=False,
):
    """Implementation for comparing Stage 1 and Stage 2 MLflow runs."""
    # Load both runs through the same normalization path so the comparison table
    # can be built from aligned epoch-level histories.
    stage1_metrics_df = _load_mlflow_metrics_df(
        experiment_name=stage1_experiment_name,
        run_name=stage1_run_name,
        tracking_uri=tracking_uri,
    )
    stage2_metrics_df = _load_mlflow_metrics_df(
        experiment_name=stage2_experiment_name,
        run_name=stage2_run_name,
        tracking_uri=tracking_uri,
    )

    stage1_history, epoch_col = _build_epoch_history(
        stage1_metrics_df,
        one_based_epoch=one_based_epoch,
    )
    stage2_history, _ = _build_epoch_history(
        stage2_metrics_df,
        one_based_epoch=one_based_epoch,
    )
    epoch_label = "Epoch (1-based)" if one_based_epoch else "Epoch (0-based)"

    # Compare both the best validation checkpoint and the final epoch because
    # they answer different questions about model quality and training stability.
    stage1_best = stage1_history.loc[stage1_history["val_acc"].idxmax()]
    stage2_best = stage2_history.loc[stage2_history["val_acc"].idxmax()]
    stage1_final = stage1_history.iloc[-1]
    stage2_final = stage2_history.iloc[-1]

    summary_df = pd.DataFrame(
        [
            {
                # Row 1: absolute summary for Stage 1.
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
                # Row 2: absolute summary for Stage 2.
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
                # Row 3: Stage 2 minus Stage 1, which makes improvement/regression
                # visible without manually subtracting columns in the notebook.
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

    fig = None
    axes = None
    if plot:
        # Build the optional figure only when requested because some notebook
        # cells only need the summary table for reporting.
        max_epoch = int(max(stage1_history[epoch_col].max(), stage2_history[epoch_col].max()))
        # Use a shared integer tick grid so both training stages line up visually.
        xticks = list(range(1, max_epoch + 1))

        # Panel order: train loss, val loss, val accuracy, then compact summary.
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

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
            # Mark the epoch with the best validation accuracy on the val-loss
            # curve so the two selection criteria can be inspected together.
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
        # Use metric names as the index so pandas can label the grouped bars directly.
        comparison_plot_df = comparison_plot_df.set_index("metric")
        # The final panel compresses the most important summary numbers into one
        # glanceable bar chart for notebook reports.
        comparison_plot_df.plot(kind="bar", ax=axes[3], width=0.75)
        axes[3].set_title("Best / Final Metric Summary")
        axes[3].set_xlabel("")
        axes[3].grid(True, axis="y", linestyle="--", alpha=0.4)
        axes[3].legend()

        plt.suptitle(f"{stage1_name} vs {stage2_name}", fontsize=15)
        plt.tight_layout()
        # Cache the comparison figure too, since this helper often feeds report exports.
        _remember_figure(fig)
        plt.show()

    if print_summary:
        # `to_string(index=False)` keeps the notebook output compact and aligned.
        print("=== Stage Training Summary ===")
        print(summary_df.to_string(index=False))

    if return_details:
        # Return raw histories in addition to the summary so notebooks can build
        # custom follow-up plots without re-querying MLflow.
        return {
            "summary": summary_df,
            "stage1_history": stage1_history,
            "stage2_history": stage2_history,
            "fig": fig,
            "axes": axes,
        }

    return summary_df


def _is_port_open(host, port):
    """Return whether a TCP port is already accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        # `connect_ex` returns 0 on success instead of raising, which makes it a
        # compact readiness check for a background server.
        return sock.connect_ex((host, port)) == 0


def _start_mlflow_ui_impl(tracking_dir=None, host="127.0.0.1", port=5000, timeout=15):
    """Implementation for starting a background MLflow UI process."""
    project_root = Path(__file__).resolve().parent.parent
    # Default to the project's artifact-backed MLflow store when callers do not
    # specify a directory explicitly.
    mlruns_dir = Path(tracking_dir or (project_root / "artifacts" / "mlruns")).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    if _is_port_open(host, port):
        url = f"http://{host}:{port}"
        # Reuse an existing server instead of spawning a duplicate process on the same port.
        print("MLflow UI appears to already be running.")
        if display is not None and HTML is not None:
            display(HTML(f'Access from: <a href="{url}" target="_blank">Open MLflow UI</a>'))
        else:
            print(f"Access from: {url}")
        return None

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        # Point MLflow explicitly at the project run store so launching from a
        # notebook kernel does not depend on the current working directory.
        "--backend-store-uri",
        f"file:{mlruns_dir}",
        "--host",
        host,
        "--port",
        str(port),
    ]

    print("Starting MLflow UI...")
    mlflow_process = subprocess.Popen(
        command,
        # Silence the background process in notebook output; users only need the URL.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Waiting for server to start...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if mlflow_process.poll() is not None:
            # If the child exits before the port opens, treat that as a startup failure.
            raise RuntimeError("MLflow UI failed to start. Check that `mlflow` is installed correctly.")
        if _is_port_open(host, port):
            break
        # Poll briefly until the TCP port starts accepting connections.
        time.sleep(0.25)
    else:
        # Clean up the child process if the server never became reachable.
        mlflow_process.terminate()
        raise TimeoutError("Timed out while waiting for the MLflow UI server to start.")

    url = f"http://{host}:{port}"
    divider = "=" * 58

    print()
    print(divider)
    print("MLflow UI is running in the background!")
    print(divider)

    if display is not None and HTML is not None:
        display(HTML(f'Access from: <a href="{url}" target="_blank">Open MLflow UI</a>'))
    else:
        print(f"Access from: {url}")

    print()
    print("The server is running in the background. You can continue with the notebook.")
    print("To stop the server later, run: mlflow_process.terminate()")
    print(divider)

    return mlflow_process
