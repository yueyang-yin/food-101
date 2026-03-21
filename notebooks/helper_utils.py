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
    # Fall back to plain-text messages when IPython rich display is unavailable.
    HTML = None
    display = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
_LAST_RENDERED_FIGURE = None


def _resolve_dataset_root(data_dir):
    """Return the Food-101 dataset root from either the root itself or its parent."""
    data_dir = Path(data_dir)

    if (data_dir / "meta" / "train.txt").exists():
        return data_dir

    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and (child / "meta" / "train.txt").exists():
            return child

    raise FileNotFoundError(f"No Food-101 dataset root found under: {data_dir}")


def _count_from_meta_split(dataset_root, split_name):
    """Count images per class from a metadata split file such as `train.txt`."""
    split_file = dataset_root / "meta" / f"{split_name}.txt"
    if not split_file.exists():
        return None

    lines = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    counts = Counter(line.split("/")[0] for line in lines)
    return dict(sorted(counts.items()))


def _resolve_meta_image_path(images_root, relative_stem):
    """Resolve one metadata entry to an on-disk image path across known extensions."""
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
        image_map.setdefault(class_name, []).append(
            _resolve_meta_image_path(images_root, relative_stem)
        )

    return dict(sorted(image_map.items()))


def display_dataset_count(data_dir):
    """Print per-class image counts for the available Food-101 metadata splits."""
    dataset_root = _resolve_dataset_root(data_dir)
    split_candidates = [
        ("train", "Train Set"),
        ("test", "Test Set"),
    ]

    found_any_split = False

    for split_key, split_title in split_candidates:
        counts = _count_from_meta_split(dataset_root, split_key)
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
        raise ValueError(f"No Food-101 metadata splits found in: {dataset_root}")


def display_random_images(data_dir, num_classes=3, images_per_class=2, random_seed=None):
    """Display a small random gallery from the Food-101 training split."""
    dataset_root = _resolve_dataset_root(data_dir)
    image_map = _image_map_from_meta_split(dataset_root, "train")
    if image_map is None:
        raise ValueError(f"No readable Food-101 training split found in: {dataset_root}")

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
    """Peel nested dataset wrappers until the base dataset is reached."""
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _resolve_class_names(dataset):
    """Extract class names from a dataset or subset backed by torchvision datasets."""
    base_dataset = _unwrap_dataset(dataset)
    if hasattr(base_dataset, "classes"):
        return list(base_dataset.classes)
    raise ValueError("Cannot resolve class names from the provided dataset.")


def _denormalize_image(image_tensor, mean, std):
    """Undo channel-wise normalization so a tensor can be plotted correctly."""
    mean_tensor = torch.tensor(mean, device=image_tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, device=image_tensor.device).view(3, 1, 1)
    return (image_tensor * std_tensor + mean_tensor).clamp(0, 1)


def _remember_figure(fig):
    """Cache the latest rendered figure so later save calls can reuse it."""
    global _LAST_RENDERED_FIGURE
    _LAST_RENDERED_FIGURE = fig
    return fig


def _resolve_figure_for_saving(fig=None):
    """Return the explicit figure, the cached figure, or the current active figure."""
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


def _require_inference_model(model):
    """Validate that the supplied object behaves like a PyTorch inference model."""
    if hasattr(model, "eval") and hasattr(model, "parameters"):
        return model

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

    model = _require_inference_model(model)
    was_training = getattr(model, "training", False)
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


def _build_epoch_history(df, one_based_epoch=True):
    """Collapse MLflow metric rows into one row per completed epoch."""
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
    return history, epoch_col


def _default_mlflow_tracking_uri():
    """Return the default local MLflow tracking URI used by this project."""
    project_root = Path(__file__).resolve().parent.parent
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
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name!r}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if runs_df.empty:
        raise ValueError(
            f"MLflow run not found for experiment={experiment_name!r}, run_name={run_name!r}."
        )

    run_id = runs_df.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()

    metric_frames = []
    for metric_name in metric_names:
        history = client.get_metric_history(run_id, metric_name)
        if not history:
            continue

        metric_df = pd.DataFrame(
            {
                "step": [metric.step for metric in history],
                metric_name: [metric.value for metric in history],
            }
        )
        metric_df = metric_df.drop_duplicates(subset="step", keep="last")
        metric_frames.append(metric_df)

    if not metric_frames:
        raise ValueError(
            f"No metric history was found for MLflow run={run_name!r} in experiment={experiment_name!r}."
        )

    merged_df = metric_frames[0]
    for metric_df in metric_frames[1:]:
        merged_df = merged_df.merge(metric_df, on="step", how="outer")

    merged_df = merged_df.sort_values("step").reset_index(drop=True)
    if "epoch" not in merged_df.columns:
        raise ValueError("MLflow metric history does not contain the required `epoch` metric.")

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

    stage1_best = stage1_history.loc[stage1_history["val_acc"].idxmax()]
    stage2_best = stage2_history.loc[stage2_history["val_acc"].idxmax()]
    stage1_final = stage1_history.iloc[-1]
    stage2_final = stage2_history.iloc[-1]

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

    fig = None
    axes = None
    if plot:
        max_epoch = int(max(stage1_history[epoch_col].max(), stage2_history[epoch_col].max()))
        xticks = list(range(1, max_epoch + 1))

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

    if return_details:
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
        return sock.connect_ex((host, port)) == 0


def start_mlflow_ui(tracking_dir=None, host="127.0.0.1", port=5000, timeout=15):
    """Start a background MLflow UI process if one is not already running."""
    project_root = Path(__file__).resolve().parent.parent
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
        time.sleep(0.25)
    else:
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
