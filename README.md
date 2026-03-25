# Food-101 🍜

Food-101 classification project built around `MobileNetV3-Large`, with a full workflow that goes from dataset sanity checks to training, analysis, and deployment-oriented export.

This repo focuses on two things:

- getting a compact model to perform well on `Food-101`
- treating efficiency and deployment as part of the project, not as an afterthought

## Overview 🔍

The project includes:

- dataset loading and sanity checks
- a baseline training pipeline in `PyTorch Lightning`
- two-stage transfer learning on top of `MobileNetV3-Large`
- `DataLoader` efficiency benchmarking
- qualitative error analysis with `Grad-CAM`
- pruning experiments
- ONNX export and selective static INT8 quantization with `onnxruntime`

## Results 📊

| Item | Result |
| --- | --- |
| Backbone | `MobileNetV3-Large` (ImageNet pretrained) |
| Best validation accuracy | `0.7836` |
| Final test accuracy | `82.76%` |
| Best deployment artifact | Selective static INT8 ONNX (`MatMul + Gemm Only`) |
| ONNX size change | `16.85 MB -> 12.95 MB` |
| Quantized accuracy change | `82.764% -> 82.745%` |
| Pruning | Not kept in the final path |

## Project Structure 🗂️

```text
food101-efficient-cv/
├── README.md                            # project overview, setup, and reproduction guide
├── LICENSE                              # repository license
├── environment.yml                      # recommended Conda environment definition
├── requirements.txt                     # environment snapshot from the working setup
├── notebooks/
│   ├── 00_sanity_check.ipynb            # dataset download and input-pipeline sanity checks
│   ├── 01_baseline_and_transfer_learning.ipynb
│   │                                       # baseline training and two-stage transfer learning
│   ├── 02_analysis_and_optimization.ipynb
│   │                                       # efficiency analysis, Grad-CAM, pruning, quantization
│   └── helper_utils.py                  # shared utility functions used by the notebooks
├── artifacts/
│   ├── checkpoints/
│   │   └── food101_mobilenetv3_stage2_best.ckpt
│   │                                       # final selected Lightning checkpoint
│   ├── weights/
│   │   └── food101_mobilenetv3_stage2_best_state_dict.pth
│   │                                       # exported PyTorch state_dict
│   ├── onnx/
│   │   ├── food101_model_FP32.onnx
│   │   ├── food101_model_FP32.onnx.data
│   │   └── food101_model_INT8_matmul_gemm_only.onnx
│   │                                       # deployment-oriented ONNX artifacts
│   └── figures/
│       └── ...                            # figures kept for the report and README
└── reports/
    └── final_report.md                 # final project report
```

## Setup ⚙️

### 1. Clone the repository

```bash
git clone <repo-url>
cd food101-efficient-cv
```

### 2. Create the environment

Recommended:

```bash
conda env create -f environment.yml
conda activate food101
```

`requirements.txt` is kept mainly as an environment snapshot. For a clean reproduction on another machine, `environment.yml` is the safer option.

## Reproduction 🧪

Launch Jupyter from the project root:

```bash
jupyter lab
```

Then run the notebooks in order.

### `notebooks/00_sanity_check.ipynb`

Purpose:

- detect the active device
- download `Food-101` on first run
- verify transforms, labels, and one `DataLoader` batch

### `notebooks/01_baseline_and_transfer_learning.ipynb`

Purpose:

- inspect the dataset
- build the train / validation / test pipeline
- train a baseline model
- train Stage 1 transfer learning
- train Stage 2 partial fine-tuning
- evaluate the final checkpoint on the held-out test split
- export stable checkpoint and weight artifacts

Main outputs:

- `artifacts/checkpoints/food101_mobilenetv3_stage2_best.ckpt`
- `artifacts/weights/food101_mobilenetv3_stage2_best_state_dict.pth`

### `notebooks/02_analysis_and_optimization.ipynb`

Purpose:

- benchmark `DataLoader` settings
- inspect predictions with `Grad-CAM`
- test a pruning configuration
- export ONNX artifacts
- compare selective static quantization candidates
- run a final inference demo

Main outputs:

- ONNX artifacts in `artifacts/onnx/`

## Notes 📝

- The repo structure above reflects the files intended to be kept in Git.
- `data/food-101/` is generated locally after download and is ignored by `.gitignore`.
- `mlruns/` is also a local experiment log directory and is not part of the committed repo snapshot.
- The first notebook uses `download=True`, so it can fetch the dataset automatically.
- Later notebooks assume the dataset is already present and use `download=False`.
- Notebook paths are written relative to the `notebooks/` directory, so the safest workflow is to launch Jupyter from the repo root.
- `DataLoader` benchmark results are machine-dependent.

## Useful Artifacts 📦

- Final report: [reports/final_report.md](reports/final_report.md)
- Best checkpoint: [artifacts/checkpoints/food101_mobilenetv3_stage2_best.ckpt](artifacts/checkpoints/food101_mobilenetv3_stage2_best.ckpt)
- Exported PyTorch weights: [artifacts/weights/food101_mobilenetv3_stage2_best_state_dict.pth](artifacts/weights/food101_mobilenetv3_stage2_best_state_dict.pth)
- FP32 ONNX model: [artifacts/onnx/food101_model_FP32.onnx](artifacts/onnx/food101_model_FP32.onnx)
- Selected INT8 ONNX model: [artifacts/onnx/food101_model_INT8_matmul_gemm_only.onnx](artifacts/onnx/food101_model_INT8_matmul_gemm_only.onnx)

## MLflow 📈

To inspect experiment runs locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Running the code in the notebook in sequence will directly open the MLFlow UI in the Jupyter environment through helper function.

## References 📚

1. Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. *Searching for MobileNetV3*. arXiv 2019.  
   arXiv: [official paper page](https://arxiv.org/abs/1905.02244)  
   PDF: [official paper PDF](https://arxiv.org/pdf/1905.02244)

2. Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. *Food-101 -- Mining Discriminative Components with Random Forests*. ECCV 2014.  
   Project page: [Food-101 dataset page](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)  
   PDF: [official paper PDF](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf)

3. PyTorch / TorchVision. *`torchvision.datasets.Food101` documentation*.  
   Docs: [official TorchVision Food101 docs](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Food101.html)
