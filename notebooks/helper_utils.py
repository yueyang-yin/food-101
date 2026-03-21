"""Helper utilities used by the Food-101 notebooks."""

from collections import Counter
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

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
            axis.imshow(plt.imread(image_path))
            axis.set_title(class_name, fontsize=16)
            axis.axis("off")

    plt.show()


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
    """Compare Stage 1 and Stage 2 MLflow runs in a table and optional plots."""
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


def start_mlflow_ui(tracking_dir=None, host="127.0.0.1", port=5000, timeout=15):
    """Start a background MLflow UI process if one is not already running."""
    project_root = Path(__file__).resolve().parent.parent
    # Default to the project's artifact-backed MLflow store when callers do not
    # specify a directory explicitly.
    mlruns_dir = Path(tracking_dir or (project_root / "artifacts" / "mlruns")).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    if _is_port_open(host, port):
        url = f"http://{host}:{port}"
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
